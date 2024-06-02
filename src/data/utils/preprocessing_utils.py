from typing import Any, Dict

import polars as pl
from data.utils.constants import IGNORE_VALUE


def filter_data_by_mapping_dict(data: pl.DataFrame, mapping_dict: Dict[int, Any]) -> pl.DataFrame:
	"""Filter the original data by using the mapping dict.

	The mapping dict contains information about which data is "missing"
	and can therefore be used to filter out this data.

	:param data: original dataframe
	:type data: pl.DataFrame
	:param mapping_dict: mapping dict with label value <-> label name mapping
	:type mapping_dict: Dict[int, Any]
	"""

	# Create a list to store column transformations
	column_transformations = []

	# Loop through the columns and apply the mapping if available
	for feat in data.columns:
		feature_mapping = mapping_dict.get(feat, False)
		if feature_mapping:
			# Construct the mapping dictionary for values to ignore
			mapping_to_ignore = {
				int(key): IGNORE_VALUE
				for key, value in feature_mapping.items()
				if "weiß nicht" in value.lower() or "(missing)" in value.lower() or "keine angabe" in value.lower()
			}
			if mapping_to_ignore:
				# Create an expression to map values using when-then-otherwise
				col_expr = pl.col(feat).map_dict(mapping_to_ignore, default=pl.col(feat))
				column_transformations.append(col_expr.alias(feat))
		else:
			# No mapping, just keep the column as is
			column_transformations.append(pl.col(feat))

	# Apply all transformations and return the new DataFrame
	return data.select(column_transformations)


def get_mapping_from_metadata(metadata: pl.DataFrame) -> Dict[int, Dict[int, Any]]:
	"""Get the value-label mapping from metadata (Polars DataFrame).

	:param metadata: Polars DataFrame with "Ausprägung-Wert" and "Ausprägung-Label" columns
	:return: Dict in the format {feature1: {value1: label1, ...}, feature2: {...}, ...}
	"""
	mapping_dict = {}
	feature_names = metadata["Variablenname"].unique()

	for feature in feature_names:
		temp_df = metadata.filter(pl.col("Variablenname") == feature)

		empty_mapping_flag = temp_df.select(pl.all().is_null().all()).select("Ausprägung-Wert").item()
		if empty_mapping_flag:
			mapping_dict[feature] = {}
			continue

		mapping_dict[feature] = dict(zip(temp_df["Ausprägung-Wert"].to_list(), temp_df["Ausprägung-Label"].to_list()))

	return mapping_dict
