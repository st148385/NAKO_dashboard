import tensorflow as tf
from tensorflow.keras import layers


class DummyModelTensorflow(tf.keras.Model):
	def __init__(self, input_shape, output_shape, **kwargs):
		super(DummyModelTensorflow).__init__(**kwargs)

		self.input_layer = layers.Input(shape=input_shape, name="input")
		self.hidden_layer = layers.Dense(32, activation="relu")
		self.output_layer = layers.Dense(output_shape, activation="linear")

	def call(self, inputs):
		x = self.input_layer(inputs)
		x = self.hidden_layer(x)
		out = self.output_layer(x)
		return out
