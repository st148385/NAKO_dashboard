from typing import Union

import gin
from data.dataloaders import ScikitLearnDataloader, TensorflowDataloader


@gin.configurable
class Runner:
	def __init__(self, model, dataloader):
		self.model = model
		self.dataloader = dataloader
		self.train_ds, self.val_ds = dataloader.get_datasets()

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
