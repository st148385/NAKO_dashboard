import logging
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
		log_interval: int = 5000,
		**kwargs,
	):
		self.model = model
		self.dataloader = dataloader
		self.train_ds, self.val_ds, self.ds_info = dataloader.get_datasets()
		self.run_paths = run_paths

		# Assertion if specific info not in ds_info
		assert self.ds_info.get("num_samples"), "num_samples not in ds_info, check your dataloader"
		assert self.ds_info.get("batch_size"), "batch_size not in ds_info, check your dataloader"

		self.batch_size = self.ds_info.get("batch_size")
		self.num_samples = self.ds_info.get("num_samples")
		self.ignore_value = self.ds_info.get("ingore_value")

		# ------------------ GIN CONFIGURABLES --------------- #

		self.num_epochs = num_epochs
		self.steps_per_epoch = self.num_samples // self.batch_size
		self.log_interval = log_interval

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
		logging.info("Start TensorFlow training...")

		for epoch in range(1, self.num_epochs + 1):
			logging.info(f"Epoch: {epoch:5}")

			for step, batch in enumerate(self.train_ds, start=1):
				features, labels = batch["features"], batch["labels"]
				self._train_step_tensorflow(features, labels, self.ignore_value)
				if (step * self.batch_size) % self.log_interval == 0:
					logging.info(f"Processed {step * self.batch_size:8} samples")
				if step >= self.steps_per_epoch:
					break

			for val_batch in self.val_ds:
				self._val_step_tensorflow(val_batch["features"], val_batch["labels"], self.ignore_value)

			template = "Loss: {:.2f}, Accuracy: {:.2f}%, Validation Loss: {:.2f}, Validation Accuracy: {:.2f}%"
			logging.info(
				template.format(
					self.train_loss.result(),
					self.train_accuracy.result() * 100,
					self.val_loss.result(),
					self.val_accuracy.result() * 100,
				)
			)

			logging.info("-" * 40)

	def _train_scikitlearn(self):
		# Scikit-learn training logic
		for batch in self.train_ds:
			features, labels = batch["features"], batch["labels"]

	@tf.function
	def _train_step_tensorflow(self, input, labels, ignore_value):
		with tf.GradientTape() as tape:
			predictions = self.model(input, training=True)
			if ignore_value:
				mask = tf.math.not_equal(labels, ignore_value)
				loss = self.loss_object(labels[mask], predictions[mask])
			else:
				loss = self.loss_object(labels, predictions)
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		self.train_loss(loss)
		self.train_accuracy(labels, predictions)

	@tf.function
	def _val_step_tensorflow(self, input, labels, ignore_value):
		predictions = self.model(input, training=False)
		if ignore_value:
			mask = tf.math.not_equal(labels, ignore_value)
			t_loss = self.loss_object(labels[mask], predictions[mask])
		else:
			t_loss = self.loss_object(labels, predictions)

		self.val_loss(t_loss)
		self.val_accuracy(labels, predictions)
		# self.confusion_matrix.update_state(labels, predictions)
