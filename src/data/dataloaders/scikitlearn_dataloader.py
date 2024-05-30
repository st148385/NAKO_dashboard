from typing import Dict, Generator, Tuple

import gin
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from .base_dataloader import BaseDataLoader


@gin.configurable
class ScikitLearnDataloader(BaseDataLoader):
	"""DataLoader for scikit-learn models with train/validation split functionality."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def get_dataset(self) -> Generator[Dict[str, npt.NDArray], None, None]:
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
