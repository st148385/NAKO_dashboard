import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


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


def extract_dataset_information(data: pd.DataFrame, metadata: pd.DataFrame, html_soup: BeautifulSoup) -> Dict[str, Any]:
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

	# Get the mapping of values to labels from metadata
	mapping_dict = get_mapping_and_datatype(metadata=metadata, html_soup=html_soup)
	st.write(mapping_dict)

	# Filter the data.
	# FLOAT DATA
	# 1. check if float data.
	# 2. check if there is a value in the mapping_dict
	# 3. If there is a value, this probably means something like "missing data"

	# INT DATA
	# 1. Probably i need to scrap the DATATYPE from the HTML file.
	# Usually it seems that the interesting classes are in the lower range
	# And the "unknown" classes are in the end of the speicifc DATATYPE
	# e.g. TINYINT got plausible values 1, 2 but unknown/missing is 8/9

	"""It seems that
	SMALLIMT(4) always includes 8888, 9999 as missing/unknown,
	TINYINT(2) can be 8/9 or 88/99
	"""

	# STRING
	# Dont know yet...

	# Access features in data
	features = data.columns

	for feat in features:
		feature_information_text = get_information_text_from_metadata_or_html_soup(feat, metadata, html_soup)
		break
