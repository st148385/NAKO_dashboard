from pathlib import Path

import polars as pl
import yaml


def read_data_with_polars(file_path: Path) -> pl.DataFrame:
	"""Reads data from a file using Polars, inferring the format from the extension.

	Supported formats:

	* CSV
	* JSON
	* Parquet
	* YAML

	:param file_path: The path to the data file.
	:type file_path: Path
	:return: The data loaded as a Polars DataFrame.
	:rtype: pl.DataFrame
	:raises ValueError: If the file format is not supported.
	"""

	file_extension = file_path.suffix.lstrip(".")

	if file_extension == "csv":
		return pl.read_csv(file_path)
	elif file_extension == "json":
		return pl.read_json(file_path)
	elif file_extension == "parquet":
		return pl.read_parquet(file_path)
	elif file_extension == "yaml":
		with file_path.open("r") as file:
			data = yaml.safe_load(file)
		return pl.DataFrame(data)
	else:
		raise ValueError(f"Unsupported file format: {file_extension}")
