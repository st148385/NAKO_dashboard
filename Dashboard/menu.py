from pathlib import Path
from typing import Union
import uuid

import pandas as pd
import streamlit as st

import numpy as np

from utils.constants import DATASETS_CSV, MAX_GROUPBY_NUMBER
from utils.preprocessing_utils import (
    calculate_correlation_groupby,
    extract_dataset_information,
    read_mri_data_from_folder,
)
from utils.reading_utils import read_csv_file_cached
from utils.visu_utils import (create_plotly_heatmap, create_plotly_histogram, create_plotly_scatterplot,
                              plot_f_of_xy, create_plotly_f_of_xy, create_count_matrix, height_label_naming)

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
            data=filtered_data, x_axis=option, feature_dict=feature_dict, groupby="basis_sex", mapping_dict=mapping_dict
        )
        _, mid, _ = st.columns(3)
        with mid:
            st.plotly_chart(feature_histogram)

        # ---------------------------------------------------------------------------------------- #

        # Visualize features against each other.
        st.markdown("#### Correlation")
        st.markdown("PLACEHOLDER TEXT")
        col3, col4 = st.columns(2)

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

        fig_relation = create_plotly_scatterplot(
            data=filtered_data,
            feature1=feature1_corr,
            feature2=feature2_corr,
            groupby="basis_sex",
            feature_dict=feature_dict,
            mapping_dict=mapping_dict,
        )

        _, mid2, _ = st.columns((4, 10, 4))
        with mid2:
            st.plotly_chart(fig_relation)
        sub_df = filtered_data[filtered_data[[feature1_corr, feature2_corr]].ne("").all(axis=1)][
            [feature1_corr, feature2_corr]
        ]
        st.markdown(f"Used samples: {len(sub_df.dropna())}")

        # ------------- Correlation calculations -------------------#
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

        # ---------------- GROUPBY CORRELATION ---------------------------- #

        groupby_feature_list = [feat for feat, feat_info in feature_dict.items() if feat_info["nominal/ordinal"]]
        groupby_options = st.multiselect(
            "How do you want to group the data",
            groupby_feature_list,
            ["basis_sex"],
            format_func=lambda option: f"{feature_dict[option]['info_text']} [{option}]",
            max_selections=MAX_GROUPBY_NUMBER,
        )

        # Get the unique values as selecting option dropping nan
        unique_values_per_groupby_option = {
            feat: filtered_data[feat].dropna(inplace=False).unique() for feat in groupby_options
        }

        if groupby_options:
            groupby_columns = st.columns(len(groupby_options))
            # ---
            for i, (take_col, col_feat) in enumerate(zip(groupby_columns, groupby_options)):
                with take_col:
                    st.selectbox(
                        f"Choose unique values from: {col_feat} --- {feature_dict[col_feat]['info_text']}",
                        unique_values_per_groupby_option[col_feat],
                        format_func=lambda option: f"[{option}] --- {mapping_dict[col_feat].get(option)}",
                        key=f"groupby_feature_{str(i)}",
                    )

        correlation, grouped_data = calculate_correlation_groupby(filtered_data, groupby_options, correlation_method)

        # Initialize the filter condition
        filter_condition = []
        # Iterate over the groupby_options
        for i, feature in enumerate(groupby_options):
            # Get the feature value from session state
            feature_value = st.session_state[f"groupby_feature_{i}"]
            # Add condition for the current feature-value pair
            if not any(filter_condition):
                filter_condition = correlation[feature] == feature_value
            else:
                filter_condition &= correlation[feature] == feature_value

        # Apply the filter condition to get the sub dataframe

        filtered_correlation = correlation
        if any(filter_condition):
            filtered_correlation = correlation[filter_condition]
            # Remove the groupby columns and set index to be features
            filtered_correlation = filtered_correlation.drop(groupby_options, axis="columns")
            filtered_correlation.set_index(filtered_correlation.columns[0], inplace=True)

        # Create plot
        correlation_heatmap = create_plotly_heatmap(filtered_correlation, cmap="RdBu_r", zmin=-1, zmax=1)
        _, mid3, _ = st.columns((4, 10, 4))
        with mid3:
            st.plotly_chart(correlation_heatmap)

        st.write(filtered_correlation)

        # col5, col6 = st.columns(2)

        # Produce a list of column names that the user can select from in the 3D plot and counts matrix UI containers
        excluded_cols_for_visualization = ["hgr_rh1_kraft", "hgr_rh2_kraft", "hgr_rh3_kraft",
                                           "hgr_lh1_kraft", "hgr_lh2_kraft", "hgr_lh3_kraft",
                                           "further_col_name_string", "further_col_name_string", ]
        selectable_cols_list = []
        for i in filtered_data.columns[1:]:
            if i not in excluded_cols_for_visualization:
                selectable_cols_list.append(i)

        # Title and selectbox for the occurrence matrix matplotlib plots in col7 and col8
        st.markdown("Before plotting a 3D plot "
                    "$ f \\left( \\mathrm{BMI}_\\mathrm{group}, \\mathrm{Age}_\\mathrm{group} \\right) $, "
                    "the following matrix can be used to visualize the number of occurrences for the chosen variable "
                    "$f$.  "
                    "I.e., check for and ignore outliers when there aren't representative or use other groups to "
                    "remove any non-representative entries of the subsequent 3D plot.")

        # # old version
        # height_variable_option_col78 = st.selectbox(
        #     'Select gender to filter data (chosen gender will be visualized):',
        #     ("hgr_rh_kraft_mean", 'ff_Glu_right', 'ff_Glu_left', 'anthro_fettmasse')  # TODO Add some options
        # )

        height_variable_option_col78 = st.selectbox(
            "Choose the attribute you wish to get more info about.",
            selectable_cols_list,
            format_func=lambda
                height_variable_option_col78: f"[{height_variable_option_col78}] — {feature_dict[height_variable_option_col78]['info_text']}",
            index=selectable_cols_list.index("hgr_rh_kraft_mean"),  # default selection on startup
            key="Select_var_for_both_counts_matrices"
        )

        height_label_col78 = height_label_naming(height_variable_option_col78)

        col7_male, col8_female = st.columns(2)

        with col7_male:

            # st.markdown("Num Measurements -- Males")
            st.markdown("<h1 style='text-align: center; color: grey;'>Num measurements — Males</h1>",
                        unsafe_allow_html=True)

            visualized_data_male = filtered_data[filtered_data["basis_sex"] == 1]

            default_age_bins_male = [18, 30, 40, 50, 60, 70, round(visualized_data_male["basis_age"].max())]
            default_BMI_bins_male = [0, 18.5, 25, 30, 35,
                                     round(max(visualized_data_male["anthro_gew"] / (
                                             visualized_data_male["anthro_groe"] / 100) ** 2), 1)]

            age_bins_input_male = st.text_input("Enter age bins (comma-separated):",
                                                ", ".join(map(str, default_age_bins_male)),
                                                key="age_bins_male_counts_matrix")
            BMI_bins_input_male = st.text_input("Enter BMI bins (comma-separated):",
                                                ", ".join(map(str, default_BMI_bins_male)),
                                                key="BMI_bins_male_counts_matrix")

            age_bins_male = np.array([float(x.strip()) for x in age_bins_input_male.split(",")])
            BMI_bins_male = np.array([float(x.strip()) for x in BMI_bins_input_male.split(",")])

            fig_male = create_count_matrix(df=visualized_data_male, height_var=height_variable_option_col78,
                                           streamlit_choose_bins=True, BMI_bins=BMI_bins_male, age_bins=age_bins_male)
            st.pyplot(fig_male)

        with col8_female:

            # st.markdown("Num measurements -- Females")
            st.markdown("<h1 style='text-align: center; color: grey;'>Num measurements — Females</h1>",
                        unsafe_allow_html=True)

            visualized_data_female = filtered_data[filtered_data["basis_sex"] == 2]

            default_age_bins_female = [18, 30, 40, 50, 60, 70, round(visualized_data_female["basis_age"].max())]
            default_BMI_bins_female = [0, 18.5, 25, 30, 35,
                                       round(max(visualized_data_female["anthro_gew"] / (
                                               visualized_data_female["anthro_groe"] / 100) ** 2), 1)]

            age_bins_input_female = st.text_input("Enter age bins (comma-separated):",
                                                  ", ".join(map(str, default_age_bins_female)),
                                                  key="age_bins_female_counts_matrix")
            BMI_bins_input_female = st.text_input("Enter BMI bins (comma-separated):",
                                                  ", ".join(map(str, default_BMI_bins_female)),
                                                  key="BMI_bins_female_counts_matrix")

            age_bins_female = np.array([float(x.strip()) for x in age_bins_input_female.split(",")])
            BMI_bins_female = np.array([float(x.strip()) for x in BMI_bins_input_female.split(",")])

            fig_female = create_count_matrix(df=visualized_data_female, height_var=height_variable_option_col78,
                                             streamlit_choose_bins=True, age_bins=age_bins_female,
                                             BMI_bins=BMI_bins_female)
            st.pyplot(fig_female)

        col9, col10 = st.columns(2)

        with col9:
            st.markdown("3D plot to visualize some variable against age groups and BMI groups")

            height_variable_option = st.selectbox(
                "Select metadata variable to visualize on the height axis:",
                selectable_cols_list,
                format_func=lambda
                    height_variable_option: f"[{height_variable_option}] — {feature_dict[height_variable_option]['info_text']}",
                index=selectable_cols_list.index("hgr_rh_kraft_mean"),  # default selection on startup
                key="Select_height_var_for_left_3D_plot"
            )

            # Selection box (st.selectbox) to choose the height variable f in the f(age, bmi) 3D plot
            if height_variable_option == "hgr_rh_kraft_mean":
                height_label = "right hand strength (kg)"
            elif "fett" in height_variable_option:
                height_label = f"{height_variable_option} (%)"
            elif "ff" in height_variable_option:
                height_label = f"{height_variable_option}"
            else:
                height_label = f"{height_variable_option} (a.u.)"

            gender_option = st.selectbox(
                'Select gender to filter data (chosen gender will be visualized):',
                ('Male', 'Female', 'All'),
                key="Select_gender_for_left_3D_plot"
            )

            use_above_groups_left = st.checkbox("Use above $$\\uparrow$$ custom BMI and age groups",
                                                value=False, key="checkbox_customgrps_left", label_visibility="visible")

            dont_start_3d_plot_from_zero_left = st.checkbox("Start height values from actual measurement results "
                                                            "instead of zero",
                                                            value=True,  # default checkbox choice on first startup
                                                            key="checkbox_3D_plot_height_axis_left",
                                                            label_visibility="visible")

            if gender_option == "Male":
                visualized_data = filtered_data[filtered_data["basis_sex"] == 1]
            elif gender_option == "Female":
                visualized_data = filtered_data[filtered_data["basis_sex"] == 2]
            elif gender_option == "All":
                visualized_data = filtered_data
            else:
                visualized_data = filtered_data
                print(f"Chosen gender {gender_option} is not allowed!")

            height_squared = (visualized_data["anthro_groe"] / 100) ** 2
            BMI = visualized_data["anthro_gew"] / height_squared

            age = visualized_data["basis_age"]

            height_values = visualized_data[height_variable_option]  # chosen via selectbox() above

            # Plot the 3D graph with matplotlib:
            # fig = plot_f_of_xy(BMI=BMI, age=age, hand_strength=hand_strength)
            # st.pyplot(fig)

            # Plot the data with plotly (for interactivity)
            if use_above_groups_left:
                fig = create_plotly_f_of_xy(BMI=BMI, age=age, height_var=height_values, height_label=height_label,
                                            age_bins=age_bins_male, BMI_bins=BMI_bins_male,
                                            dont_start_from_height_zero=dont_start_3d_plot_from_zero_left)
            else:
                fig = create_plotly_f_of_xy(BMI=BMI, age=age, height_var=height_values, height_label=height_label,
                                            dont_start_from_height_zero=dont_start_3d_plot_from_zero_left)
            st.plotly_chart(fig)

    with col10:  # The following is just a copy&paste of the code for col9 and changed var names to abc10

        st.markdown("3D plot to visualize some variable against age groups and BMI groups")

        height_variable_option_10 = st.selectbox(
            "Select metadata variable to visualize on the height axis:",
            selectable_cols_list,
            format_func=lambda
                height_variable_option_10: f"[{height_variable_option_10}] — {feature_dict[height_variable_option_10]['info_text']}",
            index=selectable_cols_list.index("hgr_rh_kraft_mean"),  # default selection on startup
            key="Select_height_var_for_right_3D_plot"
        )

        # Selection box (st.selectbox) to choose the height variable f in the f(age, bmi) 3D plot
        if height_variable_option_10 == "hgr_rh_kraft_mean":
            height_label_10 = "right hand strength (kg)"
        elif "fett" in height_variable_option_10:
            height_label_10 = f"{height_variable_option_10} (%)"
        elif "ff" in height_variable_option_10:
            height_label_10 = f"{height_variable_option_10}"
        else:
            height_label_10 = f"{height_variable_option_10} (a.u.)"

        gender_option_10 = st.selectbox(
            'Select gender to filter data (chosen gender will be visualized):',
            ('Male', 'Female', 'All'),
            key="Select_gender_for_right_3D_plot"
        )

        use_above_groups_right = st.checkbox("Use above $$\\uparrow$$ custom BMI and age groups",
                                             value=False, key="checkbox_customgrps_right", label_visibility="visible")

        dont_start_3d_plot_from_zero_right = st.checkbox("Start height values from actual measurement results "
                                                         "instead of zero",
                                                         value=True,  # default checkbox choice on first startup
                                                         key="checkbox_3D_plot_height_axis_right",
                                                         label_visibility="visible", )

        if gender_option_10 == "Male":
            visualized_data = filtered_data[filtered_data["basis_sex"] == 1]
        elif gender_option_10 == "Female":
            visualized_data = filtered_data[filtered_data["basis_sex"] == 2]
        elif gender_option_10 == "All":
            visualized_data = filtered_data
        else:
            visualized_data = filtered_data
            print(f"Chosen gender {gender_option_10} is not allowed!")

        height_squared_10 = (visualized_data["anthro_groe"] / 100) ** 2
        BMI_10 = visualized_data["anthro_gew"] / height_squared_10

        age_10 = visualized_data["basis_age"]

        height_values_10 = visualized_data[height_variable_option_10]  # chosen via selectbox() above

        # Plot the data with plotly (for interactivity)
        if use_above_groups_right:
            fig_10 = create_plotly_f_of_xy(BMI=BMI_10, age=age_10, height_var=height_values, height_label=height_label,
                                           age_bins=age_bins_female, BMI_bins=BMI_bins_female,
                                           dont_start_from_height_zero=dont_start_3d_plot_from_zero_right)
        else:
            fig_10 = create_plotly_f_of_xy(BMI=BMI_10, age=age_10, height_var=height_values_10,
                                           height_label=height_label_10,
                                           dont_start_from_height_zero=dont_start_3d_plot_from_zero_right)
        st.plotly_chart(fig_10)

    return  # return of "csv_dataset()" -> NO return value!


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if "dataset" not in st.session_state or st.session_state.dataset is None:
        no_dataset()
        return

    if st.session_state.dataset in set(DATASETS_CSV.keys()) and st.session_state.root_dir:
        csv_dataset(st.session_state.root_dir, st.session_state.dataset)
    return
