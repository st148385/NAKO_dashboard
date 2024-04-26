from pathlib import Path

import plotly.express as px
import streamlit as st
from utils.constants import DATASETS_CSV
from utils.preprocessing_utils import calculate_correlation_groupby, extract_dataset_information
from utils.reading_utils import read_csv_file_cached

if "_dataset_configuration_button" not in st.session_state:
	st.session_state.dataset_configuration_button = False

if "dataset_configuration_button" not in st.session_state:
	st.session_state.dataset_configuration_button = False


def set_dataset_configuration():
	st.session_state.dataset_configuration_button = st.session_state._dataset_configuration_button


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

	st.button("Configuration is correct", key="_dataset_configuration_button", on_click=set_dataset_configuration)

	# If configuration is confirmed we start to get specific for the dataset
	if st.session_state.dataset_configuration_button:
		# Load the data
		data = read_csv_file_cached(data_path, sep=seperator, encoding=encoding)
		metadata = read_csv_file_cached(metadata_path, sep=metadata_seperator)
		# html_soup = read_html_from_path(html_path)

		# Process data.
		# 1. Extract general for specific features using metadata and html
		feature_information, filtered_data, mapping_dict = extract_dataset_information(data, metadata, html_path)

		col1, col2 = st.columns(2)
		with col1:
			option = st.selectbox("Choose the attribute you wish to get more info about.", data.columns[1:])

		with col2:
			st.markdown("""
				**Data Description:**
						""")
			st.markdown(f"{feature_information[option]}")

			if mapping_dict.get(option):
				st.write("Mapping:")
				st.write(mapping_dict.get(option))

		# Create plotly figure
		fig = px.histogram(
			filtered_data,
			x=option,
			color="basis_sex",
			hover_data=filtered_data.columns,
		)

		fig.update_layout(
			title={
				"text": f"{option}-distribution after filtering",
				"y": 0.9,
				"x": 0.5,
				"xanchor": "center",
				"yanchor": "top",
			}
		)

		# Rename legend
		for label, sex in mapping_dict.get("basis_sex").items():
			print(label, sex)
			fig.update_traces(
				{"name": sex.replace("'", "")},
				selector={"name": str(label)},
			)

		st.plotly_chart(fig)

		# Correlation.
		st.markdown("#### Correlation")
		groupby_options = st.multiselect("How to you want to group the data", data.columns[1:], ["basis_sex"])

		correlation = calculate_correlation_groupby(filtered_data, groupby_options)

		col3, col4 = st.columns(2)

		feature_list = [feat for feat in data.columns[1:] if feat not in groupby_options]

		# TODO this does not work as wished atm.
		with col3:
			feature1_corr = st.selectbox("Choose first attribute", feature_list, key="feature1Corr")

		with col4:
			feature2_corr = st.selectbox("Choose second attribute", feature_list, key="feature2Corr")

		fig_corr = px.scatter(
			filtered_data,
			x=feature1_corr,
			y=feature2_corr,
			color="basis_sex",
			opacity=0.4,
			trendline="ols",
			color_discrete_map={
				"1": "red",
				"2": "green",
			},
		)

		# Rename legend
		for label, sex in mapping_dict.get("basis_sex").items():
			print(label, sex)
			fig_corr.update_traces(
				{"name": sex.replace("'", "")},
				selector={"name": str(label)},
			)

		st.plotly_chart(fig_corr)
		sub_df = filtered_data[filtered_data[[feature1_corr, feature2_corr]].ne("").all(axis=1)][
			[feature1_corr, feature2_corr]
		]
		st.markdown(f"Used samples: {len(sub_df.dropna())}")

		col5, col6 = st.columns(2)

		with col5:
			for groupby in groupby_options:
				for label, name in mapping_dict.get(groupby, {}).items():
					st.markdown(name)
					top_k_corr = correlation[feature1_corr][label].sort_values(ascending=False)[:10]
					st.write(top_k_corr)

		with col6:
			for groupby in groupby_options:
				for label, name in mapping_dict.get(groupby, {}).items():
					st.markdown(name)
					top_k_corr = correlation[feature2_corr][label].sort_values(ascending=False)[:10]
					st.write(top_k_corr)
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
