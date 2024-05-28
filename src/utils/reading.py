from pathlib import Path

import polars as pl
import yaml


def read_data_with_polars(file_path: Path, **kwargs) -> pl.DataFrame:
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

	file_extension = Path(file_path).suffix.lstrip(".")

	if file_extension == "csv":
		separator = kwargs["separator"]
		encoding = kwargs["encoding"]
		infer_schema_length = kwargs.get("infer_schema_length", None)  # TODO this might resolve for new version of data
		truncate_ragged_lines = kwargs.get("truncate_ragged_lines", False)
		return pl.read_csv(
			file_path,
			separator=separator,
			encoding=encoding,
			infer_schema_length=infer_schema_length,
			truncate_ragged_lines=truncate_ragged_lines,
		)
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
