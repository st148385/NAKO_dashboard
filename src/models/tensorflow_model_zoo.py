from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import layers


class DummyModelTensorflow(tf.keras.Model):
	def __init__(self, ds_info: Dict[str, Any], **kwargs):
		super().__init__(**kwargs)

		num_classes = ds_info.get("num_classes")
		print(ds_info)
		self.hidden_layer = layers.Dense(32, activation="relu")
		self.output_layer = layers.Dense(1, activation="linear")

		if num_classes:
			self.output_layer = layers.Dense(num_classes, activation="linear")

	def call(self, inputs):
		x = self.hidden_layer(inputs)
		out = self.output_layer(x)
		return out
