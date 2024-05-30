import gin

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
