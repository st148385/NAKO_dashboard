from typing import Tuple

import gin
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .base_dataloader import BaseDataLoader


@gin.configurable
class TensorflowDataloader(BaseDataLoader):
	"""
	Data loader designed for TensorFlow models.

	This class extends :class:`BaseDataLoader` and is tailored for preparing data
	specifically for TensorFlow models. It handles the removal of non-numeric
	columns, splitting of the data into training and validation sets, and
	batching for efficient TensorFlow training.

	:param data: The input DataFrame containing features and labels.
	:type data: pl.DataFrame
	:param target_feature: The name of the column in the DataFrame that represents the target variable.
	:type target_feature: str
	:param val_split: The proportion of data to be used for validation (default: 0.2).
	:type val_split: float
	:param batch_size: The size of each batch of data (default: 32).
	:type batch_size: int
	:param shuffle: Whether to shuffle the data before splitting (default: True).
	:type shuffle: bool
	:param seed: The random seed for reproducibility (default: None).
	:type seed: int

	"""

	def __init__(self, **kwargs):
		"""
		Initializes the ``TensorflowDataloader``.

		:param \**kwargs: Additional keyword arguments that can be passed to the parent class (``BaseDataLoader``)
						or used to configure specific parameters of this class.

		"""
		super().__init__(**kwargs)

	def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
		"""
		Prepares and returns batched TensorFlow datasets for training and validation.

		This method performs the following steps:

		1. Filters out non-numeric columns from the input DataFrame.
		2. Splits the data into training and validation sets using ``train_test_split``.
		3. Creates TensorFlow ``Dataset`` objects for both the training and validation sets.
		4. Batches the datasets according to the specified ``batch_size``.

		:return: A tuple containing the training and validation ``tf.data.Dataset`` objects.
		:rtype: Tuple[tf.data.Dataset, tf.data.Dataset]

		"""
		# Filter out non-numeric columns
		numeric_data = self.data.select(pl.exclude(pl.datatypes.Utf8))

		features_full = numeric_data.drop(self.target_feature).to_numpy().astype("float32")
		labels_full = numeric_data[self.target_feature].to_numpy().astype("float32")

		ds_info = self._get_dataset_info(self.data, labels_full, self.scope)

		# Split the data
		train_indices, val_indices = train_test_split(
			range(len(features_full)), test_size=self.val_split, shuffle=self.shuffle, random_state=self.seed
		)

		# Create Train dataset
		features = features_full[train_indices]
		labels = labels_full[train_indices]
		train_ds = tf.data.Dataset.from_tensor_slices({"features": features, "labels": labels})
		# TODO check behavior
		train_ds = train_ds.map(lambda x: self._remap_nan(x, self.ignore_value))
		train_ds = train_ds.batch(self.batch_size)
		train_ds = train_ds.repeat(-1)
		train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

		# Create Validation dataset
		features = features_full[val_indices]
		labels = labels_full[val_indices]
		val_ds = tf.data.Dataset.from_tensor_slices({"features": features, "labels": labels})
		# TODO check behavior
		val_ds = val_ds.map(lambda x: self._remap_nan(x, self.ignore_value))
		val_ds = val_ds.batch(self.batch_size)

		return train_ds, val_ds, ds_info

	@tf.function
	def _remap_nan(self, data_dict, ignore_value):
		new_data_dict = {}
		ignore_value_float = tf.cast(ignore_value, tf.float32)
		for key in data_dict:
			new_data_dict[key] = tf.where(tf.math.is_nan(data_dict[key]), ignore_value_float, data_dict[key])
		return new_data_dict
