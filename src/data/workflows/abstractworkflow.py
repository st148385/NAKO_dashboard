import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import gin
import polars as pl
from transformations import TRANSFORMS, TRANSFORMS_DICT
from utils.constants import DATA_PATH, IGNORE_VALUE, INPUT_ROOT, INPUTS, METADATA_PATH, OPTIONAL_INPUTS, OUTPUT_ROOT


@gin.configurable
class AbstractWorkflow(ABC):
	"""
	Abstract base class for data processing workflows using Polars.

	This class establishes the fundamental structure for a data processing workflow,
	consisting of three primary stages:

	1. **Preprocessing:** Cleaning, transforming, and preparing the raw data.
	2. **Processing:** Executing the core logic of the workflow (e.g., model training).
	3. **Postprocessing:**  Finalizing results or preparing data for output.

	Subclasses are expected to implement concrete methods for each of these stages,
	customizing them to suit specific datasets and tasks.

	**Note:**
		Subclasses must implement the ``preprocess``, ``process``, and ``postprocess`` methods.

	**Warning:**
		This class cannot be instantiated directly.

	:param path_collection: A dictionary mapping data types to file paths. Must include keys "metadata_path" and "data_path".
	:type path_collection: Dict[str, str]
	:param feature_info: A dictionary specifying preprocessing operations for each feature.
	:type feature_info: Dict[str, str], optional

	"""

	def __init__(
		self,
		path_collection: Dict[str, str],
		feature_info: Dict[str, str] = None,
	):
		"""
		Initializes the abstract workflow.

		:param path_collection: A dictionary mapping data types to file paths.
		:type path_collection: Dict[str, str]
		:param feature_info: A dictionary specifying preprocessing operations for each feature.
		:type feature_info: Dict[str, str], optional
		:raises KeyError: If required data paths ("metadata_path" or "data_path") are missing.
		:raises FileNotFoundError: If any specified data file is not found.

		"""
		self._validate_paths(path_collection)
		self.path_collection = path_collection
		self._feature_info = feature_info

	def _filter_by_config(self, data: pl.DataFrame) -> pl.DataFrame:
		"""
		This method defines a general preprocessing strategy applied to all datasets.
		It iterates over features and transformations specified in the `feature_info` dictionary,
		applying transformations from the `TRANSFORMS` enum to each column as needed. It also logs and
		drops features not explicitly listed in the `feature_info`.

		:param data: The data to preprocess.
		:type data: pl.DataFrame
		:return: The preprocessed data.
		:rtype: pl.DataFrame
		:raises ValueError: If an invalid transformation is specified for a column.
		"""
		not_listed_features = data.columns
		for column, feature_info in self._feature_info.items():  # Now using enum values
			if column in not_listed_features:
				not_listed_features.remove(column)
			print(column, feature_info)
			transform_list = feature_info["transforms"]
			if not isinstance(transform_list, list):
				transform_list = [transform_list]
			if transform_list[0] is None:
				continue
			elif set(transform_list).issubset(set(TRANSFORMS)):  # Check if it's a valid enum value
				for transform_enum in transform_list:
					transform_func = TRANSFORMS_DICT[transform_enum]
					data = transform_func(data, column)
			else:
				raise ValueError(f"Invalid transformation for column '{column}': {transform_enum}")

		# drop features which have not been mentioned explicitly
		if not_listed_features:
			logging.warn(f"Dropping not explicilty listed features from data: {not_listed_features}")
		data = data.drop(not_listed_features)
		return data

	@abstractmethod
	def preprocess(self, data: pl.DataFrame) -> pl.DataFrame:
		"""
		Preprocesses the raw data.

		This method defines a general preprocessing strategy applied to all datasets.

		:param data: The raw data to preprocess.
		:type data: pl.DataFrame
		:return: The preprocessed data.
		:rtype: pl.DataFrame
		:raises ValueError: If an invalid transformation is specified for a column.

		"""
		return data

	@abstractmethod
	def process(self, data: pl.DataFrame) -> Any:
		"""
		Performs the core processing logic on the preprocessed data.

		This method should be implemented by subclasses to execute the main steps
		of the workflow (e.g., model training, data analysis).

		:param data: The preprocessed data.
		:type data: pl.DataFrame
		:return: The result of the processing step. The type depends on the specific workflow.
		:rtype: Any

		"""
		pass

	@abstractmethod
	def postprocess(self, data: Any) -> Any:
		"""
		Applies postprocessing steps to the processed data.

		This method should be implemented by subclasses to finalize the results,
		prepare the data for output, or perform any other necessary postprocessing.

		:param data: The processed data.
		:type data: Any
		:return: The postprocessed data. The type depends on the specific workflow.
		:rtype: Any

		"""
		pass

	def run(self):
		"""Executes the entire workflow."""
		self.data = self.preprocess(self.data)
		self.data = self.process(self.data)
		self.data = self.postprocess(self.data)

		return self.data

	@staticmethod
	def _validate_paths(path_collection: Dict[str, str]) -> None:
		"""Validates that the required paths exist and are accessible."""

		# Check for missing required keys in nested dictionaries
		if not {METADATA_PATH, DATA_PATH}.issubset(path_collection[INPUTS].keys()):
			missing_keys = {METADATA_PATH, DATA_PATH} - set(path_collection[INPUTS].keys())
			raise KeyError(f"Required keys missing in {INPUTS}: {missing_keys}")

		# Check if input root exists
		input_root = Path(path_collection[INPUT_ROOT])
		if not input_root.exists():
			raise FileNotFoundError(f"Input root directory not found: {input_root}")

		# Validate paths within INPUTS (mandatory)
		for key, rel_path in path_collection[INPUTS].items():
			path = input_root / rel_path
			if not path.is_file():
				raise FileNotFoundError(f"File not found for {key}: {path}")

		# Validate paths within OPTIONAL_INPUTS (if present)
		if OPTIONAL_INPUTS in path_collection:
			for key, rel_path in path_collection[OPTIONAL_INPUTS].items():
				path = input_root / rel_path
				if not path.is_file():
					logging.warning(f"Optional file not found for {key}: {path}. Skipping...")

		# Validate OUTPUT_ROOT (if present)
		if OUTPUT_ROOT in path_collection:
			output_root = Path(path_collection[OUTPUT_ROOT])
			if not output_root.exists():
				output_root.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
			elif not output_root.is_dir():
				raise NotADirectoryError(f"{OUTPUT_ROOT} is not a directory: {output_root}")

	@staticmethod
	def save_to_csv(path: Path, df: pl.DataFrame, overwrite=False) -> None:
		"""
		Saves a Polars DataFrame to a CSV file.

		Args:
		    path (Path): The path to the output CSV file.
		    df (pl.DataFrame): The Polars DataFrame to save.
		    overwrite (bool): Whether to overwrite the file if it exists. Default is False.
		"""
		if path.exists() and not overwrite:
			raise FileExistsError(f"File '{path}' already exists. Use 'overwrite=True' to overwrite.")

		# Create parent directories if necessary
		path.parent.mkdir(parents=True, exist_ok=True)

		# Use `write_csv` with sensible defaults and optional compression
		df.write_csv(path, separator=";", has_header=True)

		logging.info(f"Saved DataFrame to '{path}'")
