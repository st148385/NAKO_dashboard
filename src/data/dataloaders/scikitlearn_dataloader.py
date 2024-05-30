from typing import Dict, List, Tuple

import gin
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from .base_dataloader import BaseDataLoader


@gin.configurable
class ScikitLearnDataloader(BaseDataLoader):
	"""
	Data loader tailored for scikit-learn models, providing train/validation splits.

	This class extends the :class:`BaseDataLoader` and is designed to prepare data for
	scikit-learn models. It handles the splitting of data into training and validation sets,
	as well as batching the data for efficient training.

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
		Initializes the ``ScikitLearnDataloader`` with configurable parameters.

		:param \**kwargs: Additional keyword arguments that can be passed to the parent class (``BaseDataLoader``)
						or used to configure specific parameters of this class.

		"""
		super().__init__(**kwargs)

	def get_datasets(self) -> Tuple[List[Dict[str, npt.NDArray]], List[Dict[str, npt.NDArray]]]:
		"""
		Splits the data, creates batches, and returns training and validation datasets.

		This method performs the following steps:

		1. Separates the features (X) and labels (y) from the input DataFrame.
		2. Splits the data into training and validation sets using ``train_test_split``.
		3. Creates batches of data for both training and validation sets. Each batch is represented as a dictionary
			with keys "features" and "labels", where the values are NumPy arrays.
		4. Returns a tuple containing two lists: the list of training batches and the list of validation batches.

		:return: A tuple where the first element is a list of training data batches, and the second element is a list of
			validation data batches.
		:rtype: Tuple[List[Dict[str, npt.NDArray]], List[Dict[str, npt.NDArray]]]
		:raises ValueError: If ``self.batch_size`` is not greater than 0.

		"""

		if self.batch_size <= 0:
			raise ValueError(f"Batch size must be greater than 0. Got {self.batch_size}")

		X = self.data.drop(self.target_feature).to_numpy()
		y = self.data[self.target_feature].to_numpy()

		# Split the data
		X_train, X_val, y_train, y_val = train_test_split(
			X, y, test_size=self.val_split, shuffle=self.shuffle, random_state=self.seed
		)

		# Create training data batches
		train_ds = []
		for i in range(0, len(X_train), self.batch_size):
			batch = {"features": X_train[i : i + self.batch_size], "labels": y_train[i : i + self.batch_size]}
			train_ds.append(batch)

		# Create validation data batches
		val_ds = []
		for i in range(0, len(X_val), self.batch_size):
			batch = {"features": X_val[i : i + self.batch_size], "labels": y_val[i : i + self.batch_size]}
			val_ds.append(batch)

		return train_ds, val_ds
