from keras import layers
from models.base_model import BaseModelTensorflow


class DummyModelTensorflow(BaseModelTensorflow):
	"""
	Class DummyModelTensorflow

	A class representing a dummy model for TensorFlow.

	Inherits from BaseModelTensorflow.

	"""

	def __init__(self, ds_info, **kwargs):
		super().__init__(ds_info, **kwargs)

		num_classes = self.ds_info.get("num_classes")
		self.num_classes = num_classes
		self.hidden_layer = None
		self.output_layer = None

	def build(self, input_shape):
		self.hidden_layer = layers.Dense(32, activation="relu", input_shape=input_shape)
		if self.num_classes:
			self.output_layer = layers.Dense(self.num_classes, activation="linear")
		else:
			self.output_layer = layers.Dense(1, activation="linear")

	def call(self, inputs):
		if self.hidden_layer is None or self.output_layer is None:
			self.build(inputs.shape)
		x = self.hidden_layer(inputs)
		out = self.output_layer(x)
		return out

	def explain(self):
		print("This is a dummy model for Tensorflow")
