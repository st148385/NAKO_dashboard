from pathlib import Path
from typing import Union

import pandas as pd
import streamlit as st
from utils.constants import DATASETS_CSV, MAX_GROUPBY_NUMBER
from utils.preprocessing_utils import (
	calculate_correlation_groupby,
	extract_dataset_information,
	read_mri_data_from_folder,
)
from utils.reading_utils import read_csv_file_cached
from utils.visu_utils import create_plotly_histogram, create_plotly_scatterplot

if "_dataset_configuration_button" not in st.session_state:
	st.session_state.dataset_configuration_button = False

if "dataset_configuration_button" not in st.session_state:
	st.session_state.dataset_configuration_button = False


def set_dataset_configuration():
	st.session_state.dataset_configuration_button = st.session_state._dataset_configuration_button


def no_dataset():
	return


def csv_dataset(root_dir: Union[str, Path], dataset: str):
	"""CSV Dateset generation

	:param root_dir: Root directory
	:type root_dir: Union[str, Path]
	:param dataset: dataset name
	:type dataset: str
	"""
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
	# MRI DATA
	mri_folder = DATASET_INFO.get("mri_folder", False)

	st.button("Configuration is correct", key="_dataset_configuration_button", on_click=set_dataset_configuration)

	# If configuration is confirmed we start to get specific for the dataset
	if st.session_state.dataset_configuration_button:
		# Load the data
		data = read_csv_file_cached(data_path, sep=seperator, encoding=encoding)
		metadata = read_csv_file_cached(metadata_path, sep=metadata_seperator)
		# html_soup = read_html_from_path(html_path)
		if mri_folder:
			mri_folder = Path(root_dir) / mri_folder
			mri_data = read_mri_data_from_folder(mri_folder)
			# TODO add feature description
			data = pd.merge(data, mri_data, on="ID")

		# Process data.
		# 1. Extract general for specific features using metadata and html
		feature_dict, filtered_data, mapping_dict = extract_dataset_information(
			data, metadata, html_path, dataset_name=dataset
		)

		col1, col2 = st.columns(2)
		feature_list = filtered_data.columns[1:]
		with col1:
			option = st.selectbox(
				"Choose the attribute you wish to get more info about.",
				feature_list,
				format_func=lambda option: f"[{option}] --- {feature_dict[option]['info_text']}",
			)

		with col2:
			st.markdown("""
				**Data Description:**
						""")

			if feature_dict.get(option):
				st.write("Feature info:")
				st.json(feature_dict.get(option), expanded=False)

			if mapping_dict.get(option):
				st.write("Mapping:")
				st.json(mapping_dict.get(option), expanded=False)

		feature_histogram = create_plotly_histogram(
			data=filtered_data, x_axis=option, groupby="basis_sex", mapping_dict=mapping_dict
		)
		_, mid, _ = st.columns(3)
		with mid:
			st.plotly_chart(feature_histogram)

		# ---------------------------------------------------------------------------------------- #

		# Correlation.
		st.markdown("#### Correlation")
		st.warning(
			f"""Data can only be grouped by NOMINAL/ORDINAL data currently. 
			Also the amount of groups are restricted to {MAX_GROUPBY_NUMBER}"""
		)
		# TODO maybe consider only as set of different features to group by
		# e.g.
		# [Sex, Age (needs to be set for ranges), features which are limited by there number of values, binary features]
		groupby_feature_list = [feat for feat, feat_info in feature_dict.items() if feat_info["nominal/ordinal"]]
		groupby_options = st.multiselect(
			"How do you want to group the data",
			groupby_feature_list,
			["basis_sex"],
			format_func=lambda option: f"{feature_dict[option]['info_text']} [{option}]",
			max_selections=MAX_GROUPBY_NUMBER,
		)
		correlation_method = st.selectbox(
			"Choose correlation method",
			["pearson", "spearman", "kendall"],
			help="""
			Different correlation methods. \n
			Pearson: is the most used one. The computation is quick but 
			the correlation describes the linear behaviour.\n
			Spearman: Calculation takes longer due to rank statstics. The correlation describes the monotonicity. \n
			Kendall: Also rank based correlation. Describes monotocity. \n
			TODO: Add Xicorr. The calculation takes longer but it describes the if Y is dependent on X.
			This correlation might be the most useful one, if the features are non-linearly dependent.
			""",
		)

		correlation, grouped_data = calculate_correlation_groupby(filtered_data, groupby_options, correlation_method)

		col3, col4 = st.columns(2)

		feature_list = [feat for feat in filtered_data.columns[1:] if feat not in groupby_options]

		# TODO this does not work as wished atm.
		with col3:
			feature1_corr = st.selectbox(
				"Choose first attribute",
				feature_list,
				key="feature1Corr",
				format_func=lambda option: f"[{option}] --- {feature_dict[option]['info_text']}",
			)

		with col4:
			feature2_corr = st.selectbox(
				"Choose second attribute",
				feature_list,
				key="feature2Corr",
				format_func=lambda option: f"[{option}] --- {feature_dict[option]['info_text']}",
			)

		fig_corr = create_plotly_scatterplot(
			data=filtered_data,
			feature1=feature1_corr,
			feature2=feature2_corr,
			groupby="basis_sex",
			mapping_dict=mapping_dict,
		)

		_, mid2, _ = st.columns(3)
		with mid2:
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

					# Get the top 10 correlations and drop the feature itself
					top_k_corr = pd.DataFrame(
						correlation.unstack()[feature1_corr]
						.sort_values(by=label, ascending=False, axis=1, key=abs)
						.loc[label]
					).drop(feature1_corr)[:10]
					top_k_corr.insert(
						1,
						"Samples used",
						[
							len(
								filtered_data[filtered_data[[feature1_corr, label]].ne("").all(axis=1)][
									[feature1_corr, label]
								].dropna()
							)
							for label in top_k_corr.index
						],
					)
					st.write(top_k_corr)

		with col6:
			for groupby in groupby_options:
				for label, name in mapping_dict.get(groupby, {}).items():
					st.markdown(name)

					# Get the top 10 correlations and drop the feature itself
					top_k_corr = pd.DataFrame(
						correlation.unstack()[feature2_corr]
						.sort_values(by=label, ascending=False, axis=1, key=abs)
						.loc[label]
					).drop(feature2_corr)[:10]

					top_k_corr.insert(
						1,
						"Samples used",
						[
							len(
								filtered_data[filtered_data[[feature2_corr, label]].ne("").all(axis=1)][
									[feature2_corr, label]
								].dropna()
							)
							for label in top_k_corr.index
						],
					)

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
