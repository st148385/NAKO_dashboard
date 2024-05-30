import gin
from data.dataloaders.dataloaders import ScikitLearnDataloader, TensorflowDataloader
from data.workflows import Diabetes13kWorkflow, Metadata30kWorkflow


@gin.configurable
class Runner:
	def __init__(self, dataloader, workflow):
		# Init workflow with gin specified config
		self.workflow = workflow
		data = self.workflow.run()
		# Workaround since init and providing data is nested in current setup
		dataloader.load(data)
		self.data = dataloader.get_dataset()
		for batch in self.data:
			print(batch["features"], batch["labels"])

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
		for batch in self.dataloader:
			# ... your TensorFlow training code ...
			pass

	def _train_scikitlearn(self):
		# Scikit-learn training logic using NumPy arrays from self.dataloader
		for batch in self.dataloader:
			# ... your Scikit-learn training code ...
			pass
