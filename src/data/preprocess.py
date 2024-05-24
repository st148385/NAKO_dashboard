import logging
from pathlib import Path
from typing import Dict

import gin
import polars as pl


@gin.configurable
class DataPreprocessor:
	"""Data preprocessor for pipeline.

	This preprocessor is used to
		- load data
		- merge data
		- compute feature selection/feature reduction
		- create correct output structure (e.g. tensor batches or numpy arrays)


	The overall operations shall be predefined using some GIN config.
	The config shall hold different stuff, e.g.
	- data_path: dict which can hold multiple relevant paths
	{
		"metadata": <some_path>,
		"data": <some_path>,
		"additional_data": <some_path> (e.g. MRI)
	}
	metadata and data shall in general be included. addtional data must be handled
	optionally with the UNIQUE KEY inside of the data ID (e.g. merge using pd.merge or polars join)

	- target_feature: a feature which shall be targetted to be predicted
	the idea is to hold the pipeline general e.g. defining the target feature
	is necessary for a reasonable feature selection and building the data/label pairs

	- task: {"regression", "classification"}
	- format: {"tensorflow", "scikit"} (?)
	"""

	def __init__(
		self,
		data_paths_unified: Dict[str, str],
		target_feature: str,
		task: str,
		format: str,
	):
		# Validate if metadata and data is provided.
		assert Path(
			data_paths_unified.get("metadata_path", "")
		).is_file(), f'Metadata not found at {data_paths_unified.get("metadata_path", "")}, check the config file...'
		assert Path(
			data_paths_unified.get("data_path", "")
		).is_file(), f'Data not found at {data_paths_unified.get("data_path", "")}, check the config file...'
		# Get metadata and data paths. If there are remaining paths inside pf data_paths_unified, they are additional
		# Data which must be merged to the "orginal data"
		metadata_path = data_paths_unified.pop("metadata_path")
		data_path = data_paths_unified.pop("data_path")

		# Read the data...
		self.data = pl.read_csv(data_path, encoding="latin1", separator=";")  # sep should be general now...
		# TODO load actual setting of metadata and probably remove infer and truncate...
		self.metadata = pl.read_csv(
			metadata_path, encoding="latin1", separator=";", infer_schema_length=0, truncate_ragged_lines=True
		)  # sep should be general now...

		# Merging the original data with all additonal data given.
		for additional_data_name, additional_data_path in data_paths_unified.items():
			logging.info(f"Original data is merged with: {additional_data_name}")
			assert Path(
				additional_data_path
			).is_file(), f"Data path is not valid: {additional_data_path}. Please check path and config"
			additional_data = pl.read_csv(additional_data_path, separator=";", encoding="latin1")
			self.data = self.data.join(additional_data, on="ID", how="left")

		self.target_feature = target_feature
		self.task = task
		self.format = format
