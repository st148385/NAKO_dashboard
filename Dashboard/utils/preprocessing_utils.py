import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from utils.constants import IGNORE_VALUE
from utils.reading_utils import read_csv_file_cached, read_html_from_path


# THIS IS ALL ASSUMED TO BE STATIC, not really in general form
def read_mri_data_from_folder(path: Union[str, Path]) -> pd.DataFrame:
	"""Read the mri data and directly merge the features.

	:param path: folder path where ff, vol and CSA data is stored.
	:type path: Union[str, Path]
	:return: dataframe from ff, vol and CSA files
	:rtype: pd.DataFrame
	"""
	path = Path(path)
	csv_files = list(path.glob("*.csv"))
	# Seperate CSV files to have fat fraction seperated.
	# I want to easily drop all columns without mm2 or mm3 from the other files
	fat_fraction_csv = [
		csv_files.pop(i) for i, file_path in enumerate(csv_files) if file_path.stem.lower() == "fat_fractions"
	][0]

	for i, file_path in enumerate(csv_files):
		if i == 0:
			dataframe = read_csv_file_cached(file_path, sep=";")
			continue
		data = read_csv_file_cached(file_path, sep=";")
		# Remove replica in form of pixel/voxel
		columns_to_keep = [col for col in data.columns if "mm^2" in col or "mm^3" in col or "ID" in col]
		data = data[columns_to_keep]
		dataframe = pd.merge(data, dataframe, on="ID")
	# Include Fat fraction file
	dataframe = pd.merge(dataframe, read_csv_file_cached(fat_fraction_csv, sep=";"), on="ID")
	return dataframe


def extract_data_type_from_html_soup(
	feature_list: List[str],
	html_soup: BeautifulSoup,
) -> Dict[str, str]:
	"""Extract datatype from HTML file

	:param html_soup: html description file
	:type html_soup: BeautifulSoup
	:return: dict holding the datatype from the corresponding feature feature --> datatype
	:rtype: Dict[str, str]
	"""

	datatype_dict: Dict[str, str] = {}

	for feature in feature_list:
		feature_html_content = html_soup.find("tr", id=feature)
		if feature_html_content is None:
			continue
		feature_text = feature_html_content.find_all(string=True)[:30]
		pattern = re.compile(r"Datentyp:\s*(.*?)\n")
		merged_data = [
			pattern.search(sublist).group(1) if pattern.search(sublist) else None for sublist in feature_text
		]
		# Get first non None element
		datatype = next((item for item in merged_data if item is not None), None)
		datatype_dict[feature] = datatype

	return datatype_dict


def get_information_text_from_metadata_or_html_soup(
	feature: str,
	metadata: pd.DataFrame,
	html_soup: BeautifulSoup,
) -> str:
	"""Get information text.

	First check metadata for available information.
	If not there scrap the html file to extract info.

	:param feature: the feature we want to get info from
	:type feature: str
	:param metadata: metadata Dataframe
	:type metadata: pd.DataFrame
	:param html_soup: html content as soup
	:type html_soup: BeautifulSoup
	:return: information text of the feature
	:rtype: str
	"""

	# First check if the metadata got the description
	feature_text = "No description found..."
	if feature in metadata.get("Variablenname").unique():
		feature_metadata = metadata[metadata["Variablenname"] == feature]
		feature_text = feature_metadata["Label"].iloc[0]
		return feature_text

	# IF not scrap the html file...
	feature_html_content = html_soup.find("tr", id=feature)
	if feature_html_content is None:
		return feature_text
	feature_text = feature_html_content.find_all(string=True, limit=10)[0:]

	# Use regex to filter unwanted characters (non numerical or alphabetical)
	pattern = re.compile(r"[^\w\s\xa0:°]+")
	merged_data = ["".join(filter(lambda x: not pattern.search(x), sublist)) for sublist in feature_text[2:]]
	feature_text = "".join(merged_data).strip()
	return feature_text


def get_mapping_from_metadata(metadata: pd.DataFrame) -> Dict[int, Any]:
	"""Get the value-label mapping from metadata

	:param metadata: dataframe from the metadata with attributes "Ausprägung-Wert" and "Ausprägung-Label"
	:type metadata: pd.DataFrame
	:return: dict holding all the mappings label_value <--> label_name
	:rtype: Dict[int, Any]
	"""
	mapping_dict: Dict[int, str] = {}
	feature_names = metadata.iloc[:, 0].unique()
	for feature in feature_names:
		temp_df = metadata[metadata.iloc[:, 0] == feature]
		mapping_dict[feature] = {
			int(label_value): label_name
			for (label_value, label_name) in zip(temp_df["Ausprägung-Wert"], temp_df["Ausprägung-Label"])
			if not np.isnan(label_value)
		}
	return mapping_dict


def get_mapping_and_datatype(metadata: pd.DataFrame, html_soup: BeautifulSoup) -> Dict[str, Any]:
	"""_summary_

	:param metadata: _description_
	:type metadata: pd.DataFrame
	:param html_soup: _description_
	:type html_soup: BeautifulSoup
	:return: _description_
	:rtype: Dict[str, Any]
	"""

	final_mapping_dict: Dict[str, Any] = {}

	mapping_dict = get_mapping_from_metadata(metadata=metadata)
	feature_list = list(mapping_dict.keys())
	datatype_dict = extract_data_type_from_html_soup(feature_list=feature_list, html_soup=html_soup)

	for feature in feature_list:
		temp_dict = {"mapping": mapping_dict.get(feature, {}), "datatype": datatype_dict.get(feature, {})}
		final_mapping_dict[feature] = temp_dict

	return final_mapping_dict


def get_iqr_filtered_data(data: pd.DataFrame, feature_dict: Dict[str, str]) -> pd.DataFrame:
	"""Filter the data using IQR method.

	First numerical columns are identified using the dtype dict
	then the IQR filtering takes place to remove outlier.

	:param data: dataframe we want to filter
	:type data: pd.DataFrame
	:param dtype_dict: dict to identify the dtype of the corresponding features
	:type dtype_dict: Dict[str, str]
	:return: filtered dataframe
	:rtype: pd.DataFrame
	"""

	# Filter Data using IQR method. Since i don't know the unterlying distributions i#m not using Zscore
	# ONLY the data which is not tagged as nominal will be filtered
	# it does not make sense to filter basis_uort in such a way.
	selected_features = [feat for feat, feat_info in feature_dict.items() if not feat_info["nominal/ordinal"]]
	sub_df = data[selected_features]
	Q1 = sub_df.quantile(0.25)
	Q3 = sub_df.quantile(0.75)
	IQR = Q3 - Q1
	outliers = (sub_df < (Q1 - 1.5 * IQR)) | (sub_df > (Q3 + 1.5 * IQR))
	sub_df[outliers] = np.nan
	data[selected_features] = sub_df
	return data


def filter_data_by_mapping_dict(data: pd.DataFrame, mapping_dict: Dict[int, Any]):
	"""Filter the original data by using the mapping dict.

	The mapping dict contains information about which data is "missing"
	an can therefore be used to filter out the this data.

	:param data: original dataframe
	:type data: pd.DataFrame
	:param mapping_dict: mapping dict with label value <-> label name mapping
	:type mapping_dict: Dict[int, Any]
	"""

	filtered_data = data.copy()

	# Remap all values which are decalred as "missing" or something similar to IGNORE_VALUE (e.g. NaN)
	# TODO IF there are no mapping dict given by the metadata, there still sometimes appear high values
	# which with high certainty correspond to MISSING data
	for feat in data:
		feature_mapping = mapping_dict.get(feat, False)
		# empty mapping or feature not in mapping_dict
		if not feature_mapping:
			continue
		mapping_to_ignore = {
			int(key): IGNORE_VALUE
			for key, value in feature_mapping.items()
			if "weiß nicht" in value.lower() or "(missing)" in value.lower() or "keine angabe" in value.lower()
		}

		if mapping_to_ignore:
			filtered_data[feat] = filtered_data[feat].map(lambda x: mapping_to_ignore.get(x, x))

	return filtered_data


def manually_filter_and_merge_data(
	df: pd.DataFrame, mapping_dict: Dict[int, str], feature_dict: Dict[str, Any], dataset_name: str
) -> pd.DataFrame:
	"""Filter the data in a manual way.

	E.g. some stuff can be merged and others can be removed.
	(merge hgl_kraft by getting the mean over all 3 features)

	:param data: dataframe
	:type data: pd.DataFrame
	:param dataset_name: dataset name, because some features might do appear only in one dataset
	:type dataset_name: str
	:return: cleaned dataframe
	:rtype: pd.DataFrame
	"""

	# Handle KRAFT left and right hand
	# --------------------------------------------------------------------------
	force_features = df.columns[df.columns.str.contains("lh.*kraft", case=False)]
	df["hgr_lh_kraft_mean"] = df[force_features].mean(axis=1)
	# df.drop(force_features, axis="columns", inplace=True)
	mapping_dict["hgr_lh_kraft_mean"] = mapping_dict[force_features[0]]
	feature_dict["hgr_lh_kraft_mean"] = feature_dict[force_features[0]]
	feature_dict["hgr_lh_kraft_mean"].update(
		{
			"info_text": "Mittelwert Kraft der linken Hand über lh1 - lh3",
		}
	)

	force_features = df.columns[df.columns.str.contains("rh.*kraft", case=False)]
	df["hgr_rh_kraft_mean"] = df[force_features].mean(axis=1)
	# df.drop(force_features, axis="columns", inplace=True)
	mapping_dict["hgr_rh_kraft_mean"] = mapping_dict[force_features[0]]
	feature_dict["hgr_rh_kraft_mean"] = feature_dict[force_features[0]]
	feature_dict["hgr_rh_kraft_mean"].update(
		{
			"info_text": "Mittelwert Kraft der rechten Hand über rh1 - rh3",
		}
	)

	# --------------------------------------------------------------------------

	return df, mapping_dict, feature_dict


# @st.cache_data
def extract_dataset_information(
	data: pd.DataFrame,
	metadata: pd.DataFrame,
	html_path: Union[str, Path],
	dataset_name: str,
) -> Dict[str, Any]:
	"""Extract feature specific information and store them in some dictionary

	:param data: original dataframe
	:type data: pd.DataFrame
	:param metadata: original metadataframe
	:type metadata: pd.DataFrame
	:param html_soup: original html soup
	:type html_soup: BeautifulSoup
	:return: dict with feature specific information
	:rtype: Dict[str, Any]
	"""
	# Init feature dict
	feature_dict: Dict[str, Any] = {}

	html_soup = read_html_from_path(html_path)

	# Get the mapping of values to labels from metadata
	mapping_dict = get_mapping_from_metadata(metadata=metadata)
	filtered_data = filter_data_by_mapping_dict(data, mapping_dict)

	# Access features in data
	features = data.columns[1:]

	for feat in features:
		feature_information_text = get_information_text_from_metadata_or_html_soup(feat, metadata, html_soup)
		feature_dict[feat] = {"info_text": feature_information_text}

		# Check for the datatype
		if str in filtered_data[feat].apply(type).unique():
			feature_dict[feat].update({"dtype": str, "nominal/ordinal": True})
			continue
		# The data is sometimes not saved as INT even though it is categorical
		if np.array_equal(filtered_data[feat].copy().dropna(), filtered_data[feat].copy().dropna().astype(int)):
			feature_dict[feat].update({"dtype": int, "nominal/ordinal": False})
		else:
			feature_dict[feat].update({"dtype": float, "nominal/ordinal": False})

		# Mapping dict shall also hold identity mappings which are not mentioned in metadata
		if feature_dict[feat]["dtype"] == int:
			temp_mapping_dict = {identity: identity for identity in filtered_data[feat].dropna()}
			mapping_dict[feat] = temp_mapping_dict | mapping_dict.get(feat, {})
			# Sort mapping by keys
			mapping_dict[feat] = {key: mapping_dict.get(feat)[key] for key in sorted(mapping_dict.get(feat, {}))}

			# IF the majority of the INTEGER VALUES are identity mappings
			# THEN the data type is really numerical
			# ELSE the data type is NOMINAL/ORDINAL
			identity_count = len([counter for counter, val in mapping_dict[feat].items() if counter == val])
			mapping_count = len([counter for counter, val in mapping_dict[feat].items() if counter != val])

			if mapping_count >= identity_count:
				feature_dict[feat]["nominal/ordinal"] = True

	# TODO i have to identify nominal data which shall not be filtered this way.
	filtered_data = get_iqr_filtered_data(filtered_data, feature_dict)
	filtered_data, mapping_dict, feature_dict = manually_filter_and_merge_data(
		filtered_data, mapping_dict, feature_dict, dataset_name
	)

	return feature_dict, filtered_data, mapping_dict


# @st.cache_data
def calculate_correlation_groupby(data: pd.DataFrame, groupby_options: List[str], correlation_method: str):
	"""Calculate the correlation based on the grouped dataFrame.

	:param data: Dataframe
	:type data: pd.DataFrame
	:param groupby: List containing the features to filter after
	:type groupby: List[str]
	"""

	# TODO dependent on the data type i want to use different correlation methods...
	if groupby_options:
		grouped_data = data.drop(["ID"], axis=1).groupby(groupby_options, sort=False)
		return grouped_data.corr(numeric_only=True, method=correlation_method), grouped_data

	return data.drop(["ID"], axis=1).corr(numeric_only=True, method=correlation_method), data
