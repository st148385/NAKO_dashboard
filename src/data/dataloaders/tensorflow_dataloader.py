import gin
import polars as pl
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .base_dataloader import BaseDataLoader


@gin.configurable
class TensorflowDataloader(BaseDataLoader):
	"""DataLoader for TensorFlow models."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def get_datasets(self) -> tf.data.Dataset:
		# TODO might handle differently
		# Identify and remove string (object) columns
		numeric_data = self.data.select(pl.exclude(pl.datatypes.Utf8))

		features_full = numeric_data.drop(self.target_feature).to_numpy().astype("float32")
		labels_full = numeric_data[self.target_feature].to_numpy().astype("float32")

		# Split the data
		train_indices, val_indices = train_test_split(
			range(len(features_full)), test_size=self.val_split, shuffle=self.shuffle, random_state=self.seed
		)

		# Create Train dataset
		features = features_full[train_indices]
		labels = labels_full[train_indices]
		train_ds = tf.data.Dataset.from_tensor_slices({"features": features, "labels": labels})
		train_ds = train_ds.batch(self.batch_size)

		# Create Validation dataset
		features = features_full[val_indices]
		labels = labels_full[val_indices]
		val_ds = tf.data.Dataset.from_tensor_slices({"features": features, "labels": labels})
		val_ds = val_ds.batch(self.batch_size)

		return train_ds, val_ds
