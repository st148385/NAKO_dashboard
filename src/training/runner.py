from typing import Dict, Union

import gin
import sklearn
import tensorflow as tf
from data.dataloaders import ScikitLearnDataloader, TensorflowDataloader


@gin.configurable
class Runner:
	def __init__(
		self,
		model: Union[tf.keras.Model, sklearn.base.BaseEstimator],
		dataloader: Union[ScikitLearnDataloader, TensorflowDataloader],
		run_paths: Dict[str, str],
		num_epochs: int = 10,
		**kwargs,
	):
		self.model = model
		self.dataloader = dataloader
		self.train_ds, self.val_ds, self.ds_info = dataloader.get_datasets()
		self.run_paths = run_paths

		# ------------------ GIN CONFIGURABLES --------------- #

		self.num_epochs = num_epochs

		# Extract kwargs Tensorflow specific
		# Train stuff
		self.loss_object = kwargs.get("loss")
		self.optimizer = kwargs.get("optimizer")
		self.train_loss = kwargs.get("train_loss")
		self.train_accuracy = kwargs.get("train_acc")
		# Val stuff
		self.val_loss = kwargs.get("val_loss")
		self.val_accuracy = kwargs.get("val_acc")

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
			self._train_step_tensorflow(features, labels)

	def _train_scikitlearn(self):
		# Scikit-learn training logic
		for batch in self.train_ds:
			features, labels = batch["features"], batch["labels"]

	# @tf.function
	def _train_step_tensorflow(self, features, labels):
		with tf.GradientTape() as tape:
			predictions = self.model(features, training=True)
			loss = self.loss_object(labels, predictions)
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		self.train_loss(loss)
		self.train_accuracy(labels, predictions)
