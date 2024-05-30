import gin

from .scikit_learn_model_zoo import DummyModelScikitLearn
from .tensorflow_model_zoo import DummyModelTensorflow


@gin.configurable
class ModelFactory:
	"""Factory class for creating workflow instances based on configuration."""

	def __init__(self, model_name: str, **kwargs):
		"""
		Args:
			workflow_name: The name of the workflow class to instantiate.
			**kwargs: Additional keyword arguments to pass to the workflow constructor.
		"""

		self._model_classes = {
			"DummyModelScikitLearn": DummyModelScikitLearn,
			"DummyModelTensorflow": DummyModelTensorflow,
		}

		model_class = self._model_classes.get(model_name)
		assert model_class, f"Invalid workflow name: {model_name}"

		self.model = model_class(**kwargs)
