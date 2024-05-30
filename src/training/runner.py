from typing import Union

import gin
import tensorflow as tf
from data.dataloaders import ScikitLearnDataloader, TensorflowDataloader


@gin.configurable
class Runner:
	def __init__(self, model, dataloader, num_epochs: int = 10, **kwargs):
		self.model = model
		self.dataloader = dataloader
		self.train_ds, self.val_ds = dataloader.get_datasets()

		# Extract kwargs Tensorflow specific
		self.loss_obj = kwargs.get("loss")
		self.optimizer = kwargs.get("optimizer")

		# Extract kwargs SciKitLearn Specific

	def run(self):
		self.train()

	def train(self):
		# Dynamically determine the appropriate train method
		if isinstance(self.dataloader, TensorflowDataloader):
			self._train_tensorflow()
		elif isinstance(self.dataloader, ScikitLearnDataloader):
			self._train_scikitlearn()
		else:
			raise ValueError("Unsupported DataLoader type")

	def _train_tensorflow(self):
		# TensorFlow-specific training logic

		for batch in self.train_ds:
			features, labels = batch["features"], batch["labels"]

	def _train_scikitlearn(self):
		# Scikit-learn training logic
		for batch in self.train_ds:
			features, labels = batch["features"], batch["labels"]

	@tf.function
	def _train_step_tensorflow(self, features, labels):
		with tf.GradientTape() as tape:
			predictions = self.model(features, training=True)
			loss = self.loss_object(labels, predictions)
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		self.train_loss(loss)
		self.train_accuracy(labels, predictions)
