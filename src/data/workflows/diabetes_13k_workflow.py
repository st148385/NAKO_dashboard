from typing import Dict

import gin

from .abstractworkflow import AbstractWorkflow


@gin.configurable
class Diabetes13kWorkflow(AbstractWorkflow):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def preprocess(self, data):
		return super().preprocess(data)

	def process(self, data):
		return super().process(data)

	def postprocess(self, data):
		return super().postprocess(data)
