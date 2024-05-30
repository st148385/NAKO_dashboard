import logging
from abc import ABC, abstractmethod
from typing import Dict, Generator

import gin
import numpy as np  # For NumPy arrays (scikit-learn)
import polars as pl
import tensorflow as tf  # For TensorFlow-specific data types


@gin.configurable
class BaseDataLoader(ABC):
	"""Base class for data loading and preprocessing."""

	def __init__(self, target_feature: str, batch_size: int):
		self.target_feature = target_feature
		self.batch_size = batch_size
		self.data = None

	def _validate_data(self):
		"""Basic validation of data."""
		if self.target_feature not in self.data.columns:
			raise KeyError(f"Target feature '{self.target_feature}' not in data.")

		# Add any other validation checks you need here

	def load(
		self,
		data: pl.DataFrame,
	):
		"""Helper function resulting from the GIN setup i use."""
		assert isinstance(data, pl.DataFrame), "Data must be polars DataFrame"
		self.data = data

	@abstractmethod
	def get_dataset(self) -> Generator[Dict[str, any], None, None]:
		"""Abstract method to be implemented by subclasses.
		Yields batches of data in the appropriate format for the ML framework.
		"""
		assert self.data is not None, "Data has not been loaded. First use dataloader.load(data)."


@gin.configurable
class TensorflowDataloader(BaseDataLoader):
	"""DataLoader for TensorFlow models."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def get_dataset(self) -> tf.data.Dataset:
		super().get_dataset()

		# Identify and remove string (object) columns
		numeric_data = self.data.select(pl.exclude(pl.datatypes.Utf8))

		tf_dataset = tf.data.Dataset.from_tensor_slices(
			{
				"features": numeric_data.drop(self.target_feature).to_numpy().astype("float32"),
				"labels": numeric_data[self.target_feature].to_numpy().astype("float32"),  # Assuming regression
			}
		)

		tf_dataset = tf_dataset.batch(self.batch_size)

		# Optionally add more dataset transformations (shuffle, prefetch, etc.)
		return tf_dataset


@gin.configurable
class ScikitLearnDataloader(BaseDataLoader):
	"""DataLoader for scikit-learn models."""

	def __init__(self, scaler, **kwargs):
		super().__init__(**kwargs)

	def get_dataset(self) -> Generator[Dict[str, np.ndarray], None, None]:
		super().get_dataset()
		X = self.data.drop(self.target_feature).to_numpy()
		y = self.data[self.target_feature].to_numpy()

		batches = []
		for i in range(0, len(X), self.batch_size):
			batch = {"features": X[i : i + self.batch_size], "labels": y[i : i + self.batch_size]}
			batches.append(batch)

		return batches
