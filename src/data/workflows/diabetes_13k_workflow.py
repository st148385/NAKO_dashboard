from typing import Dict

import gin

from .abstractworkflow import AbstractWorkflow


@gin.configurable
class Diabetes13kWorkflow(AbstractWorkflow):
	def __init__(self, feature_selection: Dict[str, str] = None, **kwargs):
		"""Init workflow

		:param feature_selection: Dict in the form of
			{
				feature1: filtermethod1,
				feature2: filtermethod2
			},
			defaults to None
			the idea is to predefine the filtering (e.g. is the data numerical we might min max scale
			if it is binary we might do something different.)
		:type feature_selection: Dict[str, str], optional
		"""
		super().__init__(**kwargs)

		# If feature selection is not specified, all the data without processing will be used.
		self.feature_selection = feature_selection

	def preprocess(self, data):
		return super().preprocess(data)

	def process(self, data):
		return super().process(data)

	def postprocess(self, data):
		return super().postprocess(data)
