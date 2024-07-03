import logging
from pathlib import Path

import gin
from utils.constants import DATA_PATH, IGNORE_VALUE, INPUT_ROOT, INPUTS, METADATA_PATH, OPTIONAL_INPUTS, OUTPUT_ROOT
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

		# Use pathlib for cleaner path handling
		metadata_path = Path(self.path_collection[INPUT_ROOT]) / self.path_collection[INPUTS][METADATA_PATH]
		data_path = Path(self.path_collection[INPUT_ROOT]) / self.path_collection[INPUTS][DATA_PATH]
		self.output_dir = Path(self.path_collection.get(OUTPUT_ROOT, "."))  # Default to current dir if not provided

		self.metadata = read_data_with_polars(
			metadata_path, separator=";", encoding="utf-8", infer_schema_length=0, truncate_ragged_lines=True
		)
		self.data = read_data_with_polars(data_path, separator=";", encoding="latin1")

		# Merge additional data (pathlib handling)
		for data_name, rel_path in self.path_collection.get(OPTIONAL_INPUTS, {}).items():
			logging.info(f"Merging '{data_name}' data...")
			data_path = Path(self.path_collection[INPUT_ROOT]) / rel_path
			additional_data = read_data_with_polars(data_path, separator=";", encoding="latin1")
			self.data = self.data.join(additional_data, on="ID", how="left")

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

	def run(self):
		super().run()

		# Save the data optionally if output dir is given
		if OUTPUT_ROOT in self.path_collection:
			output_root = Path(self.path_collection[OUTPUT_ROOT])
			input_root = Path(self.path_collection[INPUT_ROOT])

			for path_key, key in zip([DATA_PATH, METADATA_PATH], ["data", "metadata"]):
				rel_path = input_root / self.path_collection[INPUTS][path_key]
				output_path = output_root / rel_path.relative_to(input_root)
				self.save_to_csv(output_path, getattr(self, key.lower()), overwrite=True)

		return self.data
