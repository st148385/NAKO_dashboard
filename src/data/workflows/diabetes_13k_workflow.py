import gin

from .abstractworkflow import AbstractWorkflow


@gin.configurable
class Diabetes13kWorkflow(AbstractWorkflow):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def preprocess(self, data):
		# General preprocessing
		data = super().preprocess(data)

		# Dataset specific preprocessing.
		return data

	def process(self, data):
		return super().process(data)

	def postprocess(self, data):
		return super().postprocess(data)
