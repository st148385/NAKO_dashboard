import gin
from data.dataloaders.dataloaders import ScikitLearnDataloader, TensorflowDataloader
from data.workflows import Diabetes13kWorkflow, Metadata30kWorkflow


@gin.configurable
class Runner:
	def __init__(self, dataloader, workflow):
		self.dataloader = dataloader
		self.workflow = workflow
		self.workflow.run()
