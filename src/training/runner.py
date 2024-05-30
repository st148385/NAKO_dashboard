from typing import Union

import gin
from data.dataloaders.dataloaders import ScikitLearnDataloader, TensorflowDataloader
from data.workflows import Diabetes13kWorkflow, Metadata30kWorkflow


@gin.configurable
class Runner:
	def __init__(
		self,
		dataloader: Union[ScikitLearnDataloader, TensorflowDataloader],
		workflow: Union[Diabetes13kWorkflow, Metadata30kWorkflow],
	):
		# Init workflow with gin specified config
		self.workflow = workflow
		data = self.workflow.run()
		# Workaround using "load" since init and providing data is nested in current setup
		self.dataloader = dataloader
		self.dataloader.load(data)

	def run(self):
		# Preprocess data
		data = self.workflow.run()

		# Create a dataloader
		self.dataloader(data)
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
		# TensorFlow-specific training logic using tensors from self.dataloader
		for batch in self.dataloader.get_dataset():
			features, labels = batch["features"], batch["labels"]

	def _train_scikitlearn(self):
		# Scikit-learn training logic using NumPy arrays from self.dataloader
		for batch in self.dataloader.get_dataset():
			features, labels = batch["features"], batch["labels"]
