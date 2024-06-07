from typing import Any, Dict

import gin

from .scikitlearn.scikitlearn_dummy_model import DummyModelScikitLearn
from .tensorflow.tensorflow_dummy_model import DummyModelTensorflow


@gin.configurable
class ModelFactory:
	"""Factory class for creating model instances based on configuration."""

	_model_classes = {
		"DummyModelScikitLearn": DummyModelScikitLearn,
		"DummyModelTensorflow": DummyModelTensorflow,
	}

	def __init__(self, model_name: str, ds_info: Dict[str, Any], **kwargs):
		"""
		Args:
		    model_name: The name of the model class to instantiate.
		    ds_info: Dataset information necessary for model.
		    **kwargs: Additional keyword arguments to pass to the model constructor.
		"""
		assert ds_info, f"ds_info must be provided. Not {ds_info}. Check your dataloader."
		model_class = self._model_classes.get(model_name)
		assert model_class, f"Invalid model name: {model_name}"
		self.model = model_class(ds_info=ds_info, **kwargs)
