import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import gin
import polars as pl


@gin.configurable
class BaseDataLoader(ABC):
	"""Base class for data loading.

	Handles loading, merging, and basic validation of data from CSV files.

	Subclasses should implement the `transform` method to tailor data preparation
	for specific ML frameworks (e.g., TensorFlow, scikit-learn).

	Configuration (GIN):

	- **data_paths (dict):**
		- ``metadata_path`` (str): Path to the metadata CSV file.
		- ``data_path`` (str): Path to the main data CSV file.
		- Additional keys (optional): Paths to other datasets to be merged, named as you prefer.
	- **target_feature (str):** The name of the feature to be predicted.
	"""

	def __init__(self, data_paths: Dict[str, str], target_feature: str):
		self.target_feature = target_feature
		self._validate_paths(data_paths)

		# self._validate_target_feature(self.data, self.target_feature)

	@staticmethod
	def _validate_paths(data_paths: Dict[str, str]) -> None:
		"""Validates that the required paths exist and are files."""
		for data_type, path in data_paths.items():
			path = Path(path)
			if not path.is_file():
				raise FileNotFoundError(f"{data_type} not found at {path}")

	@staticmethod
	def _validate_target_feature(data: pl.DataFrame, feature: str) -> None:
		"""Check if the target_feature is even included in data."""
		if feature not in data.columns:
			raise KeyError(f"Target feature: {feature} not included in data")

	@abstractmethod
	def transform(self):
		"""Transforms the data into the desired format.

		This method is abstract and must be implemented by subclasses.
		"""
		pass


@gin.configurable
class TensorflowDataloader(BaseDataLoader):
	"""Data Loader for TensorFlow models."""

	def __init__(self, batch_size: int, **kwargs):
		"""Initializes the TensorFlowLoader.

		Since `TensorFlowLoader` extends `BaseDataLoader`,
		it inherits its configuration parameters.

		:param data_paths: (Inherited) Dictionary containing paths to the data files.
		:param target_feature: (Inherited) Name of the target feature.
		"""
		super().__init__(**kwargs)

		# TODO ONLY EXAMPLE
		self.batch_size = batch_size

	def transform(self):
		"""Transforms the data into TensorFlow tensors."""
		logging.info("Transforming data for TensorFlow...")
		# ... your TensorFlow-specific code here ...


@gin.configurable
class ScikitLearnDataloader(BaseDataLoader):
	"""Data loader for scikit-learn models."""

	def __init__(self, scaler, **kwargs):
		"""Initializes the ScikitLearnDataloader.

		Since `ScikitLearnLoader` extends `BaseDataLoader`,
		it inherits its configuration parameters.

		:param data_paths: (Inherited) Dictionary containing paths to the data files.
		:param target_feature: (Inherited) Name of the target feature.
		"""
		super().__init__(**kwargs)

		# TODO this is only an example
		self.scaler = scaler

	def transform(self):
		"""Transforms the data into NumPy arrays suitable for scikit-learn."""
		logging.info("Transforming data for scikit-learn...")
		# ... your scikit-learn specific code here ...
