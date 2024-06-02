import math

import gin
import polars as pl

from data.utils.constants import IGNORE_VALUE
from data.utils.preprocessing_utils import filter_data_by_mapping_dict, get_mapping_from_metadata

from .abstractworkflow import AbstractWorkflow


# Assuming df is your Polars DataFrame containing sa_ogtt0 and sa_ogtt2 columns
def classify_diabtes_severity(row):
	sa_ogtt0, sa_ogtt2 = row["sa_ogtt0"], row["sa_ogtt2"]

	# Due to remapping some values are "nan"
	if (
		sa_ogtt0 is None
		or (isinstance(sa_ogtt0, float) and math.isnan(sa_ogtt0))
		or sa_ogtt2 is None
		or (isinstance(sa_ogtt2, float) and math.isnan(sa_ogtt2))
	):
		return IGNORE_VALUE
	if sa_ogtt0 >= 7 or sa_ogtt2 >= 11.1:
		return 0  # Class 0
	elif 6.1 <= sa_ogtt0 < 7:
		return 1  # Class 1
	elif 7.8 <= sa_ogtt2 < 11.1:
		return 2  # Class 2
	elif sa_ogtt0 < 6.1 and sa_ogtt2 < 7.8:
		return 3  # Class 3
	else:
		raise ValueError(
			"""
			All cases of oGTT values should be caught and a class (0, 1, 2 or 3) determined for them, 
			with the current class definitions. We can't get here! Look for mistakes in the if statement. \n
			The problem occurred for sa_ogtt0 = {sa_ogtt0} and sa_ogtt2 = {sa_ogtt2}"""
		)


@gin.configurable
class Diabetes13kWorkflow(AbstractWorkflow):
	"""
	Workflow for processing the Diabetes13k dataset.

	This workflow inherits from :class:`AbstractWorkflow` and provides specific implementations
	for preprocessing, processing, and postprocessing steps tailored to the Diabetes13k dataset.

	:param \**kwargs: Additional keyword arguments to pass to the parent class (``AbstractWorkflow``).
	"""

	def __init__(self, **kwargs):
		"""
		Initializes the ``Diabetes13kWorkflow``.

		:param \**kwargs: Keyword arguments passed to the base class constructor.
		"""
		super().__init__(**kwargs)

		# Get mapping dict using the metadata
		self.mapping_dict = get_mapping_from_metadata(self.metadata)
		# Replace Missing, unknown not valid data with nans
		self.data = filter_data_by_mapping_dict(self.data, self.mapping_dict)

	def preprocess(self, data):
		"""
		Preprocesses the Diabetes13k dataset.

		This method first calls the general preprocessing of the parent class and then
		performs additional preprocessing specific to the Diabetes13k dataset.

		:param data: The input data to be preprocessed.
		:type data: pl.DataFrame
		:return: The preprocessed data.
		:rtype: pl.DataFrame

		"""

		# General preprocessing
		data = super().preprocess(data)
		# TODO: Add dataset-specific preprocessing steps here.

		########################################################
		####### use WHO definition regarding sa_ogtt to ########
		####### 	create diabetes_severity column		########
		data = data.with_columns(
			pl.struct(["sa_ogtt0", "sa_ogtt2"])
			.apply(lambda row: classify_diabtes_severity(row))
			.alias("diabetes_class")
		)
		# Artifically add clas and update the preprocess_basis
		# otherwise the class will be removed by _filter_by_config
		# Since it is not in there...
		self._preprocess_basis.update({"diabetes_class": None})
		########################################################

		# The manually choosen stuff will be done here then.

		# TODO might need to rethink this, if artifcially adding stuff here
		# this won't appear in the original config.

		data = self._filter_by_config(data)

		return data

	def process(self, data):
		"""
		Processes the preprocessed Diabetes13k data.

		This method currently delegates processing to the parent class.

		:param data: The preprocessed data to be processed.
		:type data: Any
		:return: The processed data.
		:rtype: Any

		"""
		return data

	def postprocess(self, data):
		"""
		Postprocesses the processed Diabetes13k data.

		This method currently delegates postprocessing to the parent class.

		:param data: The processed data to be postprocessed.
		:type data: Any
		:return: The postprocessed data.
		:rtype: Any

		"""
		return data
