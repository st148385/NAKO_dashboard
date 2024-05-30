import gin
import polars as pl

from .scikitlearn_dataloader import ScikitLearnDataloader
from .tensorflow_dataloader import TensorflowDataloader


@gin.configurable
class DataLoaderFactory:
	"""Factory class for creating dataloader instances based on configuration."""

	def __init__(self, dataloader_name: str, **kwargs):
		"""
		Args:
		    dataloader_name: The name of the dataloader class to instantiate.
		    **kwargs: Additional keyword arguments to pass to the dataloader constructor.
		"""

		self._dataloader_classes = {
			"TensorflowDataloader": TensorflowDataloader,
			"ScikitLearnDataloader": ScikitLearnDataloader,
		}

		dataloader_class = self._dataloader_classes.get(dataloader_name)
		assert dataloader_class, f"Invalid dataloader name: {dataloader_name}"

		self.dataloader = dataloader_class(**kwargs)

	def load(self, data: pl.DataFrame):
		"""Loads data into the dataloader."""
		self.dataloader.load(data)

	def get_datasets(self):
		"""Retrieves the dataset from the dataloader."""
		return self.dataloader.get_dataset()
