import logging
from pathlib import Path
from typing import Dict

import gin
import polars as pl


@gin.configurable
class DataPreprocessorBase:
	"""Base class for data preprocessing in ML pipelines.

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

		# Read CSV files
		self.metadata = pl.read_csv(
			data_paths.pop("metadata_path"),
			separator=";",
			encoding="latin1",
			infer_schema_length=0,  # TODO this might resolve for new version of data
			truncate_ragged_lines=True,
		)
		self.data = pl.read_csv(data_paths.pop("data_path"), separator=";", encoding="latin1")

		# Merge additional data (if any)
		for data_name, data_path in data_paths.items():
			logging.info(f"Merging '{data_name}' data...")
			additional_data = pl.read_csv(data_path, separator=";", encoding="latin1")
			self.data = self.data.join(additional_data, on="ID", how="left")

	@staticmethod
	def _validate_paths(data_paths):
		"""Validates that the required paths exist and are files."""
		for data_type, path in data_paths.items():
			path = Path(path)
			if not path.is_file():
				raise FileNotFoundError(f"{data_type} not found at {path}")

	def transform(self):
		"""Transforms the data into the desired format.

		This method is abstract and must be implemented by subclasses.
		"""
		raise NotImplementedError("Subclasses must implement the transform method.")


@gin.configurable
class TensorFlowPreprocessor(DataPreprocessorBase):
	"""Data preprocessor for TensorFlow models."""

	def __init__(self, batch_size: int, **kwargs):
		"""Initializes the TensorFlowPreprocessor.

		Since `TensorFlowPreprocessor` extends `DataPreprocessorBase`,
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
class ScikitLearnPreprocessor(DataPreprocessorBase):
	"""Data preprocessor for scikit-learn models."""

	def __init__(self, scaler, **kwargs):
		"""Initializes the ScikitLearnPreprocessor.

		Since `ScikitLearnPreprocessor` extends `DataPreprocessorBase`,
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
