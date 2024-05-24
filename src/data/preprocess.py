import logging
from pathlib import Path
from typing import Dict

import gin
import polars as pl


@gin.configurable
class DataPreprocessor:
	"""A configurable data preprocessor for machine learning pipelines.

	This class handles essential preprocessing steps including:

	1. Loading data from CSV files (metadata, main data, and optional additional datasets).
	2. Merging the main dataset with additional datasets using a common unique identifier (``ID``).
	3. (Optional) Performing feature selection or reduction based on configuration.
	4. Preparing data in the desired format for the specified task (e.g., TensorFlow tensors, NumPy arrays).

	Configuration (GIN):

	- **data_paths (dict):**
		- ``metadata_path`` (str): Path to the metadata CSV file.
		- ``data_path`` (str): Path to the main data CSV file.
		- Additional keys (optional): Paths to other datasets to be merged, named as you prefer.
	- **target_feature (str):** The name of the feature to be predicted.
	- **task (str):** The type of machine learning task ("regression" or "classification").
	- **format (str):** The desired output data format (e.g., "tensorflow" or "scikit").

	Example GIN configuration:

	```
	DataPreprocessor.data_paths = {
	    "metadata_path": "/path/to/metadata.csv",
	    "data_path": "/path/to/data.csv",
	    "mri_data_path": "/path/to/mri_data.csv",
	}
	DataPreprocessor.target_feature = "some_label"
	DataPreprocessor.task = "regression"
	DataPreprocessor.format = "tensorflow"
	```

	Note:

	* All CSV files must contain a column named "ID" for merging purposes.

	"""

	def __init__(
		self,
		data_paths: Dict[str, str],
		target_feature: str,
		task: str,
		format: str,
	):
		"""Initializes the DataPreprocessor.

		:param data_paths_unified: Dictionary containing paths to the data files.
		:param target_feature: Name of the target feature.
		:param task: The machine learning task ("regression" or "classification").
		:param format: The desired output data format (e.g., "tensorflow" or "scikit").
		"""
		self.target_feature = target_feature
		self.task = task
		self.format = format

		self._validate_paths(data_paths)

		# Load metadata and main data
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
