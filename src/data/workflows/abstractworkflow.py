from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import gin
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

	def __init__(self, data_paths: Dict[str, str]):
		"""_summary_

		:param data_paths: _description_
		:type data_paths: Dict[str, str]
		"""

		self._validate_paths(data_paths)
		self.data_paths = data_paths

	# TODO might reduce to only one process...
	@abstractmethod
	def preprocess(self, data):
		"""Preprocesses the raw data from the CSV file."""
		pass

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
		data = read_data_with_polars(self.data_paths)

		data = self.preprocess(data)
		data = self.process(data)
		data = self.postprocess(data)

		return data

	@staticmethod
	def _validate_paths(data_paths: Dict[str, str]) -> None:
		"""Validates that the required paths exist and are files."""
		for data_type, path in data_paths.items():
			path = Path(path)
			if not path.is_file():
				raise FileNotFoundError(f"{data_type} not found at {path}")
