import gin

from data.workflows.diabetes_13k_workflow import Diabetes13kWorkflow
from data.workflows.metadata_30k_workflow import Metadata30kWorkflow


@gin.configurable
class WorkflowFactory:
	"""Factory class for creating workflow instances based on configuration."""

	def __init__(self, workflow_name: str, **kwargs):
		"""
		Args:
		    workflow_name: The name of the workflow class to instantiate.
		    **kwargs: Additional keyword arguments to pass to the workflow constructor.
		"""

		self._workflow_classes = {
			"Diabetes13kWorkflow": Diabetes13kWorkflow,
			"Metadata30kWorkflow": Metadata30kWorkflow,
		}

		workflow_class = self._workflow_classes.get(workflow_name)
		assert workflow_class, f"Invalid workflow name: {workflow_name}"

		self.workflow = workflow_class(**kwargs)

	def run(self):
		"""Executes the workflow."""
		return self.workflow.run()
