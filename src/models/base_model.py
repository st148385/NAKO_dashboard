from abc import ABC, abstractmethod

import tensorflow as tf
from sklearn.base import BaseEstimator


class BaseModel(ABC):
	@abstractmethod
	def __init__(self, ds_info, **kwargs):
		self.ds_info = ds_info

	@abstractmethod
	def explain(self):
		pass


class BaseModelScitkitLearn(BaseModel, BaseEstimator):
	@abstractmethod
	def __init__(self, ds_info, **kwargs):
		super().__init__(ds_info, **kwargs)
		BaseEstimator.__init__(self)

	@abstractmethod
	def fit(self, X, y, **kwargs):
		pass

	@abstractmethod
	def predict(self, X):
		pass

	@abstractmethod
	def explain(self):
		pass


class BaseModelTensorflow(BaseModel, tf.keras.Model):
	@abstractmethod
	def __init__(self, ds_info, **kwargs):
		super().__init__(ds_info, **kwargs)
		tf.keras.Model.__init__(self, **kwargs)

	@abstractmethod
	def call(self, inputs):
		pass

	@abstractmethod
	def explain(self):
		pass
