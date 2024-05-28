import polars as pl


def min_max_norm(df: pl.DataFrame, column: str) -> pl.DataFrame:
	"""Applies min-max normalization to a column."""
	return df.with_columns((pl.col(column) - pl.col(column).min()) / (pl.col(column).max() - pl.col(column).min()))


def drop(df: pl.DataFrame, column: str) -> pl.DataFrame:
	"""Drops a column from the DataFrame."""
	return df.drop(column)
