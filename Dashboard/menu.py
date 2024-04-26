from pathlib import Path

import streamlit as st
from utils.constants import DATASETS_CSV
from utils.preprocessing_utils import extract_dataset_information
from utils.reading_utils import read_csv_file_cached, read_html_from_path


def no_dataset():
	return


def csv_dataset(root_dir, dataset):
	# Combine root and dataset name
	data_root = Path(root_dir, dataset)

	# Visualize the constants.
	st.markdown("####  Dataset specific configuration")
	st.info("Modify the values such that they fit and confirm afterwards.")
	DATASET_INFO = st.data_editor(DATASETS_CSV[dataset], disabled=[0])
	# Extract the necessary information
	data_path = data_root / DATASET_INFO["data_filename"]
	metadata_path = data_root / DATASET_INFO["metadata_filename"]
	html_path = data_root / DATASET_INFO["HTML_filename"]
	seperator = DATASET_INFO["data_seperator"]
	metadata_seperator = DATASET_INFO["metadata_seperator"]
	encoding = DATASET_INFO["encoding"]

	st.button("Configuration is correct", key="dataset_configuration")

	# If configuration is confirmed we start to get specific for the dataset
	if st.session_state.dataset_configuration:
		# Load the data
		data = read_csv_file_cached(data_path, sep=seperator, encoding=encoding)
		metadata = read_csv_file_cached(metadata_path, sep=metadata_seperator)
		# html_soup = read_html_from_path(html_path)

		# Process data.
		# 1. Extract general for specific features using metadata and html
		feature_information = extract_dataset_information(data, metadata, html_path)
		# 2.
	return


def menu():
	# Determine if a user is logged in or not, then show the correct
	# navigation menu
	if "dataset" not in st.session_state or st.session_state.dataset is None:
		no_dataset()
		return

	if st.session_state.dataset in set(DATASETS_CSV.keys()) and st.session_state.root_dir:
		csv_dataset(st.session_state.root_dir, st.session_state.dataset)
	return
