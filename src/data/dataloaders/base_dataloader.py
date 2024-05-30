from abc import ABC, abstractmethod
from typing import Any, Tuple

import gin
import polars as pl


@gin.configurable
class BaseDataLoader(ABC):
	"""
	Base class for data loading and preprocessing.

	This abstract base class establishes a common interface for data loaders, providing
	methods for loading data, optionally splitting it into training and validation sets,
	and performing preprocessing steps as needed. It serves as a foundation for concrete
	data loader implementations tailored to specific machine learning frameworks.

	:param data: The input DataFrame containing features and labels.
	:type data: pl.DataFrame
	:param target_feature: The name of the column in the DataFrame that represents the target variable.
	:type target_feature: str
	:param batch_size: The size of each batch of data.
	:type batch_size: int
	:param val_split: The proportion of data to be used for validation (default: 0.2).
	:type val_split: float
	:param shuffle: Whether to shuffle the data before splitting (default: True).
	:type shuffle: bool
	:param seed: The random seed for reproducibility (default: 42).
	:type seed: int

	"""

	def __init__(
		self,
		data: pl.DataFrame,
		target_feature: str,
		batch_size: int,
		val_split: float = 0.2,
		shuffle: bool = True,
		seed: int = 42,
	):
		"""
		Initializes the ``BaseDataLoader``.

		:param data: The input DataFrame containing features and labels.
		:type data: pl.DataFrame
		:param target_feature: The name of the column in the DataFrame that represents the target variable.
		:type target_feature: str
		:param batch_size: The size of each batch of data.
		:type batch_size: int
		:param val_split: The proportion of data to be used for validation (default: 0.2).
		:type val_split: float
		:param shuffle: Whether to shuffle the data before splitting (default: True).
		:type shuffle: bool
		:param seed: The random seed for reproducibility (default: 42).
		:type seed: int


		.. note::
		Subclasses must implement the ``get_datasets`` method.

		.. warning::
			This class cannot be instantiated directly.
		"""

		self.target_feature = target_feature
		self.batch_size = batch_size
		self.data = data
		self.val_split = val_split
		self.shuffle = shuffle
		self.seed = seed
		self._validate_data()

	def _validate_data(self):
		"""
		Performs basic validation checks on the input data.

		This method currently checks if the specified ``target_feature`` exists in the DataFrame.
		Additional validation checks can be added here as needed.

		:raises KeyError: If the ``target_feature`` is not found in the DataFrame.

		"""

		if self.target_feature not in self.data.columns:
			raise KeyError(f"Target feature '{self.target_feature}' not in data.")

	@abstractmethod
	def get_datasets(self) -> Tuple[Any, Any]:
		"""
		Abstract method for generating datasets.

		Subclasses must implement this method to return training and validation datasets in a format
		suitable for the specific machine learning framework they are designed for.

		:return: A tuple containing the training and validation datasets. The actual types will vary
				depending on the subclass implementation.
		:rtype: Tuple[Any, Any]

		"""

		pass
