"""Utility functions for streamlit webpage."""

import re
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup

sns.set_theme()


@st.cache_data
def filter_data(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""_summary_

	:param dataframe: _description_
	:type dataframe: pd.DataFrame
	:return: _description_
	:rtype: pd.DataFrame
	"""

	df_new = dataframe.copy()

	for _, series in df_new.items():
		# If more than one datatype
		unique_types = series.apply(type).unique()
		# more than 1 data type
		if len(unique_types) > 1:
			continue

		# Get all potentially classification features...
		series_arr = series.copy(deep=True)
		series_arr.dropna(inplace=True)
		# if np.array_equal(series_arr, series_arr.astype(int)):
		# TODO might be the wrong threshold
		series[series > 5000] = np.nan

		# TODO not only classification data seems to have this mixture
		# anthro hueftumfang for example also is float but contains 9999 as not known data
	return df_new


@st.cache_data
def extract_infos_given_datapath(
	original_data: pd.DataFrame,
	filtered_data: pd.DataFrame,
	description_file: Union[str, Path],
	metadata_path: Union[str, Path],
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
	features = filtered_data.columns[1:]

	with open(str(description_file)) as file:
		html_content = file.read()

	soup = BeautifulSoup(html_content, "html.parser")
	metadata = pd.read_csv(metadata_path, sep=";", on_bad_lines="skip")  # , encoding="latin1")
	mapping_dict = get_mapping_from_metadata(metadata=metadata)
	for feat in features:
		# Get info text
		feature_information_text = get_information_from_metadata_or_html_file(
			feature=feat, soup=soup, metadata=metadata
		)
		# Get stats
		feature_mapping = mapping_dict.get(feat, {})
		feature_stats = get_data_stats(original_data[feat], filtered_data[feat], feature_mapping=feature_mapping)

		# Merge info and stats
		temp_dict = {
			"feature_information_text": feature_information_text,
			"feature_stats": feature_stats,
		}
		feature_description[feat] = temp_dict

	return feature_description


def get_information_from_metadata_or_html_file(feature: str, soup: BeautifulSoup, metadata: pd.DataFrame) -> str:
	"""Extract feature description text from HTML file

	:param feature: Feature you want the information for
	:type feature: str
	:param soup: soup object containing the HTML info
	:type soup: BeautifulSoup
	:return: description of the corresponding feature
	:rtype: str
	"""

	# First check if the metadata got the description
	feature_text = ""
	if feature in metadata["Variablenname"].unique():
		feature_metadata = metadata[metadata["Variablenname"] == feature]
		feature_text = feature_metadata["Label"].iloc[0]
		return feature_text

	# IF not scrap the html file...
	if feature_text in [None, ""]:
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
def get_data_stats(
	feature_series: pd.Series, feature_series_filtered: pd.Series, feature_mapping: Dict[int, str]
) -> Dict[str, Any]:
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
	if str in feature_series.apply(type).unique():
		feature_stats["values"] = feature_series.unique()
		return feature_stats

	# Compute numerical stats
	series_arr = np.array(feature_series)

	# Sometimes discrete values are something saved as float (e.g. d_an_msart1b)
	# check if equally holds if converted to int
	if np.array_equal(series_arr, series_arr.astype(int)):
		values, counts = np.unique(series_arr, return_counts=True)
		feature_stats["data_distribution"] = {
			int(k): {"count": int(v), "name": feature_mapping.get(int(k), int(k))} for k, v in zip(values, counts)
		}

	# Compute stats on filtered data
	feature_stats["max"] = feature_series_filtered[feature_series_filtered.notnull()].max()
	feature_stats["min"] = feature_series_filtered[feature_series_filtered.notnull()].min()
	feature_stats["mean"] = feature_series_filtered[feature_series_filtered.notnull()].mean()
	feature_stats["std"] = feature_series_filtered[feature_series_filtered.notnull()].std()
	feature_stats["0.25-quantile"] = feature_series_filtered[feature_series_filtered.notnull()].quantile(0.25)
	feature_stats["0.75-quantile"] = feature_series_filtered[feature_series_filtered.notnull()].quantile(0.75)
	feature_stats["median"] = feature_series_filtered[feature_series_filtered.notnull()].median()

	return feature_stats


def get_mapping_from_metadata(metadata: pd.DataFrame) -> Dict[int, Any]:
	"""_summary_

	:param metadata: dataframe from the metadata where column 0 is the attribute name, column -2 is the attribute value
		and column -1 is the attribute name
	:type metadata: pd.DataFrame
	:return: dict holding all the mappings label_value <--> label_name
	:rtype: Dict[int, Any]
	"""
	metadata_for_mapping = metadata.copy()
	metadata_for_mapping.dropna(inplace=True)
	mapping_dict = {}
	feature_names = metadata_for_mapping.iloc[:, 0].unique()
	for feature in feature_names:
		temp_df = metadata_for_mapping[metadata_for_mapping.iloc[:, 0] == feature]
		mapping_dict[feature] = {
			int(label_value): label_name for (label_value, label_name) in zip(temp_df.iloc[:, -2], temp_df.iloc[:, -1])
		}
	return mapping_dict


def create_distribution_plot(
	data_distribution: Dict[int, Dict[str, Any]], title: str = "Data Distribution"
) -> plt.Figure:
	"""_summary_

	:param values: _description_
	:type values: NDArray
	:param counts: _description_
	:type counts: NDArray
	:return: _description_
	:rtype: plt.Figure
	"""

	counts = [data_point["count"] for data_point in data_distribution.values()]

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	x_ticks = range(len(counts))
	ax.bar(x_ticks, counts)
	ax.set_ylabel("Data count")
	ax.set_title(title)

	tick_control_parameter = 1 if len(counts) <= 15 else 10
	label_names = [data_point["name"] for data_point in data_distribution.values()]
	label_values = [label for label in data_distribution]
	ax.set_xticks(x_ticks[::tick_control_parameter])
	ax.set_xticklabels(label_names[::tick_control_parameter])

	for tick, label_name, label_value in zip(ax.get_xticklabels(), label_names, label_values):
		if np.issubdtype(type(label_name), str):
			tick.set_text(f"{tick.get_text()} ({label_value})")
			tick.set_rotation(90)

	return fig


# TODO there is still bugs e.g. in 13k diabetes, also handling string data etc must take place beforehand....
# TODO OVERALL a data cleaning must take place, e.g. some values are 9999 and bias the whole stats.
def compute_correlation_and_plot_data(
	feature1: str, feature2: str, data: pd.DataFrame, attr_info: Dict[str, Any]
) -> Tuple[float, plt.Figure]:
	"""_summary_

	:param feature1: _description_
	:type feature1: str
	:param feature2: _description_
	:type feature2: str
	:param data: _description_
	:type data: pd.DataFrame
	:return: _description_
	:rtype: Tuple[float, plt.Figure]
	"""
	sub_df = data[data[[feature1, feature2]].ne("").all(axis=1)][[feature1, feature2]]
	sub_df.dropna(inplace=True)
	correlation = sub_df.corr()[feature1].iloc[-1]

	# extract feature specific info
	feature_dict = {}
	feature_dict["feature1"] = [
		data_point for data_point in attr_info[feature1]["feature_stats"].get("data_distribution", {}).values()
	]
	feature_dict["feature2"] = [
		data_point for data_point in attr_info[feature2]["feature_stats"].get("data_distribution", {})
	]

	fig = create_feature_plot(data[feature1], data[feature2], feature_dict)
	data_count = len(sub_df)
	return correlation, fig, data_count


def create_feature_plot(feature1_arr, feature2_arr, feature_dict: Dict[str, Any]) -> plt.Figure:
	"""_summary_

	:param feature1_arr: _description_
	:type feature1_arr: _type_
	:param feature2_arr: _description_
	:type feature2_arr: _type_
	:param feature_dict: _description_
	:type feature_dict: Dict[str, Any]
	:return: _description_
	:rtype: plt.Figure
	"""

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
	sns.regplot(x=feature1_arr, y=feature2_arr)

	"""
	# Regulate x ticks
	tick_control_parameter = 1 if len(feature1_arr) <= 10 else 10
	ticks = range(len(feature1_arr))
	label_names = [data_point["name"] for data_point in feature_dict["feature1"].values()]
	label_values = [label for label in feature_dict[feature1_arr]]
	ax.set_xticks(ticks[::tick_control_parameter])
	ax.set_xticklabels(label_names[::tick_control_parameter])

	for tick, label_name, label_value in zip(ax.get_xticklabels(), label_names, label_values):
		if np.issubdtype(type(label_name), str):
			tick.set_text(f"{tick.get_text()} ({label_value})")
			tick.set_rotation(90)

	# feature 1 is x_axis
	"""
	return fig
