from abc import ABC, abstractmethod
from typing import Dict, Generator

import gin
import polars as pl


@gin.configurable
class BaseDataLoader(ABC):
	"""Base class for data loading and preprocessing."""

	def __init__(
		self,
		data: pl.DataFrame,
		target_feature: str,
		batch_size: int,
		val_split: float = 0.2,
		shuffle: bool = True,
		seed: int = 42,
	):
		self.target_feature = target_feature
		self.batch_size = batch_size
		self.data = data
		self.val_split = val_split
		self.shuffle = shuffle
		self.seed = seed

	def _validate_data(self):
		"""Basic validation of data."""
		if self.target_feature not in self.data.columns:
			raise KeyError(f"Target feature '{self.target_feature}' not in data.")

		# Add any other validation checks you need here

	@abstractmethod
	def get_dataset(self) -> Generator[Dict[str, any], None, None]:
		"""Abstract method to be implemented by subclasses.
		Yields batches of data in the appropriate format for the ML framework.
		"""
		pass
