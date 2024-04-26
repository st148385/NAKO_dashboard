import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from utils.constants import IGNORE_VALUE
from utils.reading_utils import read_html_from_path


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
	mapping_to_ignore: Dict[int, float] = {}
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


@st.cache_data
def extract_dataset_information(
	data: pd.DataFrame, metadata: pd.DataFrame, html_path: Union[str, Path]
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
	correlation = filtered_data.corr(numeric_only=True)

	# TODO STRING data is not properly handled atm.

	# Access features in data
	features = data.columns

	for feat in features:
		feature_information_text = get_information_text_from_metadata_or_html_soup(feat, metadata, html_soup)
		break
