"""Utility functions for streamlit webpage."""

import csv
import re
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup

sns.set_theme()


@st.cache_data
def extract_infos_given_datapath(
	data_path: Union[str, Path], description_file: Union[str, Path], metadata_path: Union[str, Path]
) -> Dict[str, Any]:
	"""This function is used to search for the description of a specific feature given some
	data.

	The @st.cache_data decorator ensures that the the processing given the parameter setup is
	done only once. Hashable objects like lists can't be used in this process.

	:param data_path: Path of the CSV file containing the features.
	:type data_path: Union[str, Path]
	:param description_file: HTML file containing the data information
	:type description_file: Union[str, Path]
	:param metadata_path: path to the metadata file
	:type metadata_path: Union[str, Path]
	:return: Dictionary with {feature: description} where description describes what the feature tells
	:rtype: Dict[str, Any]
	"""

	feature_description: Dict[str, Any] = {}

	data = pd.read_csv(data_path, sep=";", encoding="latin1", quoting=csv.QUOTE_NONE)
	features = data.columns[1:]

	with open(str(description_file)) as file:
		html_content = file.read()

	soup = BeautifulSoup(html_content, "html.parser")
	metadata = pd.read_csv(metadata_path, sep=";", on_bad_lines="skip")
	metadata.dropna(inplace=True)
	mapping_dict = get_mapping_from_metadata(metadata=metadata)
	for feat in features:
		# Get info text
		feature_information_text = get_information_from_html_file(feature=feat, soup=soup)
		# Get stats
		feature_mapping = mapping_dict.get(feat, {})
		feature_stats = get_data_stats(data[feat], feature_mapping=feature_mapping)
		# Get plots

		temp_dict = {
			"feature_information_text": feature_information_text,
			"feature_stats": feature_stats,
		}
		feature_description[feat] = temp_dict

	return feature_description


def get_information_from_html_file(feature: str, soup: BeautifulSoup) -> str:
	"""Extract feature description text from HTML file

	:param feature: Feature you want the information for
	:type feature: str
	:param soup: soup object containing the HTML info
	:type soup: BeautifulSoup
	:return: description of the corresponding feature
	:rtype: str
	"""
	feature_html_content = soup.find("tr", id=feature)
	if feature_html_content is None:
		return "No description found..."
	feature_text = feature_html_content.find_all(string=True, limit=10)[0:]

	# Use regex to filter unwanted characters (non numerical or alphabetical)
	pattern = re.compile(r"[^\w\s\xa0:°]+")
	merged_data = ["".join(filter(lambda x: not pattern.search(x), sublist)) for sublist in feature_text[2:]]
	feature_text = "".join(merged_data).strip()
	return feature_text


# TODO some calculations are still somehow wrong and don't match the HTML file
def get_data_stats(feature_series: pd.Series, feature_mapping: Dict[int, str]) -> Dict[str, Any]:
	"""Calculate data stats and store in dict

	:param feature_series: Data for a specific feature
	:type feature_series: pd.Series
	:param metadata_path: path to the metadata file
	:type metadata_path: Union[str, Path]
	:return: Dict holding information of data
	:rtype: Dict[str, Any]
	"""
	feature_series.dropna(inplace=True)
	feature_stats = {}
	feature_stats["dtype"] = feature_series.dtype

	# TODO add mapping to be able to also compute stats...
	if np.issubdtype(feature_series.dtype, str):
		feature_stats["values"] = feature_series.unique()
		return feature_stats

	# Somehow there are string values in some columns...
	# TODO check this behavior
	feature_series = feature_series.apply(pd.to_numeric, errors="coerce").dropna()

	# Compute numerical stats
	series_arr = np.array(feature_series)
	feature_stats["max"] = feature_series.max()
	feature_stats["min"] = feature_series.min()
	if np.issubdtype(feature_series.dtype, int):
		values, counts = np.unique(series_arr, return_counts=True)
		feature_stats["data_distribution"] = {
			int(k): {"count": int(v), "name": feature_mapping.get(int(k), k)} for k, v in zip(values, counts)
		}

	feature_stats["mean"] = feature_series.mean()
	feature_stats["std"] = feature_series.std()
	feature_stats["0.25-quantile"] = feature_series.quantile(0.25)
	feature_stats["0.75-quantile"] = feature_series.quantile(0.75)
	feature_stats["median"] = feature_series.median()

	return feature_stats


def get_mapping_from_metadata(metadata: pd.DataFrame) -> Dict[int, Any]:
	"""_summary_

	:param metadata: dataframe from the metadata where column 0 is the attribute name, column -2 is the attribute value
		and column -1 is the attribute name
	:type metadata: pd.DataFrame
	:return: dict holding all the mappings label_value <--> label_name
	:rtype: Dict[int, Any]
	"""
	mapping_dict = {}
	feature_names = metadata.iloc[:, 0].unique()
	for feature in feature_names:
		temp_df = metadata[metadata.iloc[:, 0] == feature]
		mapping_dict[feature] = {
			int(label_value): label_name for (label_value, label_name) in zip(temp_df.iloc[:, -2], temp_df.iloc[:, -1])
		}
	return mapping_dict


def create_distribution_plot(data_distribution: Dict[int, Dict[str, Any]]) -> plt.Figure:
	"""_summary_

	:param values: _description_
	:type values: NDArray
	:param counts: _description_
	:type counts: NDArray
	:return: _description_
	:rtype: plt.Figure
	"""

	counts = [data_point["count"] for data_point in data_distribution.values()]
	# label_names = [data_point["name"] for data_point in data_distribution.values()]
	x_ticks = range(len(counts))

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	ax.bar(x_ticks, counts)
	ax.set_ylabel("Data count")
	ax.set_xlabel("Label value")
	ax.set_title("Class distribution")
	return fig
