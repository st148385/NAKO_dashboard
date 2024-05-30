import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import gin
import polars as pl
from transformations import TRANSFORMS, TRANSFORMS_DICT
from utils.reading import read_data_with_polars


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

	:param data_paths: A dictionary mapping data types to file paths. Must include keys "metadata_path" and "data_path".
	:type data_paths: Dict[str, str]
	:param preprocess_basis: A dictionary specifying preprocessing operations for each feature.
	:type preprocess_basis: Dict[str, str], optional

	"""

	def __init__(
		self,
		data_paths: Dict[str, str],
		preprocess_basis: Dict[str, str] = None,
	):
		"""
		Initializes the abstract workflow.

		:param data_paths: A dictionary mapping data types to file paths.
		:type data_paths: Dict[str, str]
		:param preprocess_basis: A dictionary specifying preprocessing operations for each feature.
		:type preprocess_basis: Dict[str, str], optional
		:raises KeyError: If required data paths ("metadata_path" or "data_path") are missing.
		:raises FileNotFoundError: If any specified data file is not found.

		"""
		self._validate_paths(data_paths)
		self.data_paths = data_paths
		self._preprocess_basis = preprocess_basis

		# Load metadata and main data
		self.metadata = read_data_with_polars(
			data_paths.pop("metadata_path"),
			separator=";",
			encoding="latin1",
			infer_schema_length=0,
			truncate_ragged_lines=True,
		)
		self.data = read_data_with_polars(data_paths.pop("data_path"), separator=";", encoding="latin1")

		# Merge additional data (if any)
		for data_name, data_path in data_paths.items():
			logging.info(f"Merging '{data_name}' data...")
			additional_data = read_data_with_polars(data_path, separator=";", encoding="latin1")
			self.data = self.data.join(additional_data, on="ID", how="left")  # Assuming "ID" is the join key

	@abstractmethod
	def preprocess(self, data: pl.DataFrame) -> pl.DataFrame:
		"""
		Preprocesses the raw data.

		This method defines a general preprocessing strategy applied to all datasets.
		It iterates over features and transformations specified in the `preprocess_basis` dictionary,
		applying transformations from the `TRANSFORMS` enum to each column as needed. It also logs and
		drops features not explicitly listed in the `preprocess_basis`.

		:param data: The raw data to preprocess.
		:type data: pl.DataFrame
		:return: The preprocessed data.
		:rtype: pl.DataFrame
		:raises ValueError: If an invalid transformation is specified for a column.

		"""

		not_listed_features = data.columns
		for column, transform_list in self._preprocess_basis.items():  # Now using enum values
			if column in not_listed_features:
				not_listed_features.remove(column)
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
	def _validate_paths(data_paths: Dict[str, str]) -> None:
		"""Validates that the required paths exist and are files."""

		check_keys = {"metadata_path", "data_path"}
		target_keys = set(data_paths.keys())

		if not check_keys.issubset(target_keys):
			raise KeyError(f"Keys: {check_keys} not in {target_keys}")

		for data_type, path in data_paths.items():
			path = Path(path)
			if not path.is_file():
				raise FileNotFoundError(f"{data_type} not found at {path}")
