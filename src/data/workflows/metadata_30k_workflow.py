import logging

import gin
from utils.constants import DATA_PATH, IGNORE_VALUE, METADATA_PATH, OUTPUT_DIR_PATH
from utils.reading import read_data_with_polars

from .abstractworkflow import AbstractWorkflow


@gin.configurable
class Metadata30kWorkflow(AbstractWorkflow):
	"""
	Workflow for processing the Metadata30k dataset.

	This workflow inherits from :class:`AbstractWorkflow` and provides specific implementations
	for preprocessing, processing, and postprocessing steps tailored to the Metadata30k dataset.

	:param \**kwargs: Additional keyword arguments to pass to the parent class (``AbstractWorkflow``).
	"""

	def __init__(self, **kwargs):
		"""
		Initializes the ``Metadata30kWorkflow``.

		:param \**kwargs: Keyword arguments passed to the base class constructor.
		"""
		super().__init__(**kwargs)

		# Load metadata and main data
		self.metadata = read_data_with_polars(
			self.path_collection[METADATA_PATH],
			separator=";",
			encoding="utf-8",
			infer_schema_length=0,
			truncate_ragged_lines=True,
		)
		self.data = read_data_with_polars(self.path_collection[DATA_PATH], separator=";", encoding="latin1")

		# Merge additional data (if any)
		for data_name, data_path in self.path_collection.items():
			if data_name in {DATA_PATH, METADATA_PATH}:
				continue
			logging.info(f"Merging '{data_name}' data...")
			additional_data = read_data_with_polars(data_path, separator=";", encoding="latin1")
			self.data = self.data.join(additional_data, on="ID", how="left")  # Assuming "ID" is the join key

	def preprocess(self, data):
		"""
		Preprocesses the Metadata30k dataset.

		This method first calls the general preprocessing of the parent class and then
		performs additional preprocessing specific to the Metadata30k dataset.

		:param data: The input data to be preprocessed.
		:type data: pl.DataFrame
		:return: The preprocessed data.
		:rtype: pl.DataFrame

		"""

		# General preprocessing
		data = super().preprocess(data)

		# TODO: Add dataset-specific preprocessing steps here.

		return data

	def process(self, data):
		"""
		Processes the preprocessed Metadata30k data.

		This method currently delegates processing to the parent class.

		:param data: The preprocessed data to be processed.
		:type data: Any
		:return: The processed data.
		:rtype: Any

		"""
		super().process(data)
		return data

	def postprocess(self, data):
		"""
		Postprocesses the processed Metadata30k data.

		This method currently delegates postprocessing to the parent class.

		:param data: The processed data to be postprocessed.
		:type data: Any
		:return: The postprocessed data.
		:rtype: Any

		"""
		super().postprocess(data)
		return data
