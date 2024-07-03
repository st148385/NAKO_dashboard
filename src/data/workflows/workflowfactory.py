import gin

from data.workflows.diabetes_13k_workflow import Diabetes13kWorkflow
from data.workflows.metadata_30k_workflow import Metadata30kWorkflow


@gin.configurable
class WorkflowFactory:
	"""
	Factory for creating and executing data workflows.

	This class provides a flexible way to instantiate and run different data workflows
	based on configuration. It currently supports the following workflows:

	* :class:`Diabetes13kWorkflow`
	* :class:`Metadata30kWorkflow`

	The specific workflow to be executed is determined by the `workflow_name` parameter,
	and additional keyword arguments can be passed to the workflow constructor for customization.

	:param workflow_name: The name of the workflow class to instantiate ("Diabetes13kWorkflow" or "Metadata30kWorkflow")
	:type workflow_name: str
	:param \**kwargs: Additional keyword arguments to pass to the workflow constructor.
	:raises AssertionError: If an invalid ``workflow_name`` is provided.
	"""

	def __init__(self, workflow_name: str, **kwargs):
		"""
		Initializes the `WorkflowFactory` and instantiates the specified workflow.

		:param workflow_name: The name of the workflow class to instantiate.
		:type workflow_name: str
		:param \**kwargs: Additional keyword arguments to pass to the workflow constructor.
		:raises AssertionError: If an invalid `workflow_name` is provided.

		"""
		self._workflow_classes = {
			"Diabetes13kWorkflow": Diabetes13kWorkflow,
			"Metadata30kWorkflow": Metadata30kWorkflow,
		}

		workflow_class = self._workflow_classes.get(workflow_name)
		assert workflow_class, f"Invalid workflow name: {workflow_name}"

		self.workflow = workflow_class(**kwargs)

	def run(self):
		"""
		Executes the instantiated workflow.

		This method calls the `run` method of the specific workflow object that was
		created during initialization.

		:return: The result of the workflow execution. The type of the result depends on the specific workflow.

		"""
		return self.workflow.run()
