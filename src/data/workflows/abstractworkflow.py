import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import gin
from transformations import TRANSFORMS, TRANSFORMS_DICT
from utils.reading import read_data_with_polars


@gin.configurable
class AbstractWorkflow(ABC):
	"""Abstract base class for processing data using Polars.

	This class defines the structure of a data processing workflow. It outlines
	three main stages: preprocessing, processing, and postprocessing. Subclasses
	must implement concrete methods for each of these stages to tailor the
	workflow to specific datasets and tasks.

	:param file_path: The path to the data file.
	:type file_path: Path

	.. note::
		Subclasses must implement the ``preprocess``, ``process``, and
		``postprocess`` methods.

	.. warning::
		This class cannot be instantiated directly.
	"""

	def __init__(
		self,
		data_paths: Dict[str, str],
		preprocess_basis: Dict[str, str] = None,
	):
		"""_summary_

		:param data_paths: _description_
		:type data_paths: Dict[str, str]
		"""

		self._validate_paths(data_paths)
		self.data_paths = data_paths

		# TODO might outsource to the subclasses...
		# Read metadata and actual data
		self.metadata = read_data_with_polars(
			data_paths.pop("metadata_path"),
			separator=";",
			encoding="latin1",
			infer_schema_length=0,  # TODO this might resolve for new version of data
			truncate_ragged_lines=True,
		)
		self.data = read_data_with_polars(data_paths.pop("data_path"), separator=";", encoding="latin1")
		# Merge additional data (if any)
		for data_name, data_path in data_paths.items():
			logging.info(f"Merging '{data_name}' data...")
			additional_data = read_data_with_polars(data_path, separator=";", encoding="latin1")
			self.data = self.data.join(additional_data, on="ID", how="left")

		# Dict holding operations for each feature. See base config for this.
		self._preprocess_basis = preprocess_basis

	# TODO might reduce to only one process...
	@abstractmethod
	def preprocess(self, data):
		"""Preprocesses the raw data from the CSV file."""

		# General preprocessing.
		# Defining all operations for a given feature inside of the config file
		# e.g. {"basis_age": [%TRANSFORMS.MIN_MAX_NORM, %TRANSFORMS.DROP]}
		# will min max scale basis_age and drop it afterwards (does not make sense, just as example :D)
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
	def process(self, data):
		"""Performs the core processing logic on the data."""
		pass

	@abstractmethod
	def postprocess(self, data):
		"""Applies any post-processing steps to the processed data."""
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
