import logging
from typing import Dict, Union

import gin
import sklearn
import tensorflow as tf
import wandb
from data.dataloaders import ScikitLearnDataloader, TensorflowDataloader
from rich.progress import Progress
from utils.utils_misc import gin_config_to_readable_dictionary, log_dict


@gin.configurable
class Runner:
	def __init__(
		self,
		model: Union[tf.keras.Model, sklearn.base.BaseEstimator],
		dataloader: Union[ScikitLearnDataloader, TensorflowDataloader],
		run_paths: Dict[str, str],
		wandb_api_key: str = None,
		num_epochs: int = 10,
		**kwargs,
	):
		self.model = model
		self.dataloader = dataloader
		self.train_ds, self.val_ds, self.ds_info = dataloader.get_datasets()
		self.run_paths = run_paths
		self.wandb_api_key = wandb_api_key
		self.config_dict = gin_config_to_readable_dictionary(gin.config._CONFIG)

		# Assertion if specific info not in ds_info
		assert self.ds_info.get("num_samples"), "num_samples not in ds_info, check your dataloader"
		assert self.ds_info.get("batch_size"), "batch_size not in ds_info, check your dataloader"

		self.batch_size = self.ds_info.get("batch_size")
		self.num_samples = self.ds_info.get("num_samples")
		self.ignore_value = self.ds_info.get("ingore_value")

		# ------------------ GIN CONFIGURABLES --------------- #

		self.num_epochs = num_epochs
		self.steps_per_epoch = self.num_samples // self.batch_size

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
		# Check if API key is valid
		self._configure_wandb()

		# Log configuration
		log_dict(self.config_dict, "GIN Configuration")
		# Log Dataset info
		log_dict(self.ds_info, "Dataset Information")

		# Log
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
		"""Trains a TensorFlow model with step progress bar and epoch-end metric logging."""
		logging.info("Start TensorFlow training...")

		for epoch in range(1, self.num_epochs + 1):
			with Progress() as progress:  # Progress bar only for steps within the epoch
				step_task = progress.add_task(f"[blue]Epoch {epoch}", total=self.steps_per_epoch)

				for step, batch in enumerate(self.train_ds, start=1):
					features, labels = batch["features"], batch["labels"]
					self._train_step_tensorflow(features, labels, self.ignore_value)

					# Update step progress
					progress.update(step_task, advance=1)

					if step >= self.steps_per_epoch:
						break

				for val_batch in self.val_ds:
					self._val_step_tensorflow(val_batch["features"], val_batch["labels"], self.ignore_value)

			# Log training and validation metrics after each epoch (outside the progress bar context)
			template = (
				"Epoch {} | Loss: {:.4f}, Accuracy: {:.2f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%"
			)
			logging.info(
				template.format(
					epoch,  # Include the epoch number
					self.train_loss.result(),
					self.train_accuracy.result() * 100,
					self.val_loss.result(),
					self.val_accuracy.result() * 100,
				)
			)

			# Log to wandb
			if self.wandb_api_key:
				wandb.log(
					{
						"train_loss": self.train_loss.result(),
						"train_accuracy": self.train_accuracy.result(),
						"val_loss": self.val_loss.result(),
						"val_accuracy": self.val_accuracy.result(),
						"epoch": epoch,
					}
				)

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

	def _configure_wandb(self):
		# Attempt to log in to wandb if API key is provided
		if self.wandb_api_key:
			try:
				wandb.login(key=self.wandb_api_key)
			except wandb.errors.error.UsageError:  # Invalid API key
				logging.warning("Invalid wandb API key. Disabling wandb logging.")
				self.wandb_api_key = None  # Reset to None to disable logging

		# Log to wandb if logged in successfully
		if self.wandb_api_key:
			wandb.init(project="Research-thesis", config=self.config_dict)
			wandb.config.update(self.ds_info)
