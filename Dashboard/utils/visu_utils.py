import copy
from typing import Any, Dict, List, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

# matplotlib.use('TkAgg')  # To fix seaborn (sns) UserWarning on plt.show()

# from utils.constants import TICK_FREQUENCY
TICK_FREQUENCY = 5


def create_plotly_histogram(
    data: pd.DataFrame,
    x_axis: str,
    feature_dict: Dict[str, Any],
    groupby: str = None,
    mapping_dict: Dict[int, str] = False,
):
    """Create histogram figure

    :param data: dataframe
    :type data: pd.DataFrame
    :param x_axis: over which feature we want to compute the histogram
    :type x_axis: str
    :param groupby: Over which feature we want to group (e.g. colorize)
    :type groupby: str
    """

    if not mapping_dict:
        mapping_dict = {}

    # Create plotly figure
    fig = px.histogram(
        data,
        x=x_axis,
        color=groupby,
        hover_data=data.columns,
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.8,
        barmode="overlay",
    )

	fig.update_layout(
		title={
			"text": f"{x_axis}-distribution after filtering",
			"y": 0.9,
			"x": 0.5,
			"xanchor": "center",
			"yanchor": "top",
		}
	)
	if feature_dict.get(x_axis)["type"] in {"binary", "ordinal", "nominal"}:
		tick_frequency = TICK_FREQUENCY if len(mapping_dict[x_axis].keys()) > TICK_FREQUENCY else 1
		fig.update_xaxes(
			tickvals=[key for i, (key, _) in enumerate(mapping_dict[x_axis].items()) if i % tick_frequency == 0],
			ticktext=[val for i, (_, val) in enumerate(mapping_dict[x_axis].items()) if i % tick_frequency == 0],
		)

	# Rename legend
	if mapping_dict.get(groupby):
		for label, label_name in mapping_dict.get(groupby).items():
			fig.update_traces(
				{"name": str(label_name).replace("'", "")},
				selector={"name": str(copy.copy(label))},
			)

	return fig


def create_plotly_scatterplot(
    data: pd.DataFrame,
    feature1: str,
    feature2: str,
    feature_dict: Dict[str, Any],
    groupby: Union[List[str], str],
    mapping_dict: Dict[int, str] = False,
):
    """_summary_

    :param data: _description_
    :type data: pd.DataFrame
    :param feature1: _description_
    :type feature1: str
    :param feature2: _description_
    :type feature2: str
    :param groupby: _description_
    :type groupby: Union[List[str], str]
    :param mapping_dict: _description_, defaults to {}
    :type mapping_dict: Dict[int, str], optional
    """

    if not mapping_dict:
        mapping_dict = {}

    # There's no beautiful solution to group the plot by multiple stuff.
    # Maybe only for binary features.
    copied_data = data.copy()
    if isinstance(groupby, str):
        groupby = [groupby]

    # To ensure discrete color in legend
    for group in groupby:
        copied_data[group] = copied_data[group].astype(str)

    fig = px.scatter(
        copied_data,
        x=feature1,
        y=feature2,
        color=groupby[0],
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.1,
        marginal_x="box",
        marginal_y="box",
        trendline="ols",
        width=1000,
        height=800,
    )

	if feature_dict.get(feature1)["type"] in {"binary", "ordinal", "nominal"}:
		tick_frequency = TICK_FREQUENCY if len(mapping_dict[feature1].keys()) > TICK_FREQUENCY else 1
		fig.update_xaxes(
			tickvals=[key for i, (key, _) in enumerate(mapping_dict[feature1].items()) if i % tick_frequency == 0],
			ticktext=[val for i, (_, val) in enumerate(mapping_dict[feature1].items()) if i % tick_frequency == 0],
		)

	if feature_dict.get(feature2)["type"] in {"binary", "ordinal", "nominal"}:
		tick_frequency = TICK_FREQUENCY if len(mapping_dict[feature2].keys()) > TICK_FREQUENCY else 1
		fig.update_yaxes(
			tickvals=[key for i, (key, _) in enumerate(mapping_dict[feature2].items()) if i % tick_frequency == 0],
			ticktext=[val for i, (_, val) in enumerate(mapping_dict[feature2].items()) if i % tick_frequency == 0],
		)

	if mapping_dict.get(groupby[0]):
		# Rename legend
		for label, label_name in mapping_dict.get(groupby[0]).items():
			fig.update_traces(
				{"name": str(label_name).replace("'", "")},
				selector={"name": str(label)},
			)
	return fig


def create_plotly_heatmap(data: pd.DataFrame, cmap: str = "RdBu_r", zmin: float = -1, zmax: float = 1):
    """_summary_

    :param data: _description_
    :type data: pd.DataFrame
    :param cmap: _description_, defaults to "RdBu_r"
    :type cmap: str, optional
    :param zmin: _description_, defaults to -1
    :type zmin: float, optional
    :param zmax: _description_, defaults to 1
    :type zmax: float, optional
    """
    return px.imshow(data, color_continuous_scale=cmap, origin="lower", zmax=zmax, zmin=zmin, width=800, height=800)


def plot_f_of_xy(BMI, age, hand_strength, z_name="default (unit)", elev=20., azim=30, streamlit=True):
    """
    <<<ATTENTION: CURRENTLY USING THE PLOTLY FUNCTION BELOW INSTEAD OF THIS ONE!!>>>

    Plot f(x,y) in a 3D plot. Uncommenting "ax.view_init()" allows to change
    the view angle of the resulting plot (e.g. like panning it).

    Takes np.arrays of x,y and z values, returns a plotly fig to put into st.plotly_chart(fig)
    TODO: Add naming for height variable 'z', e.g. z_name = 'hand strength (kg)' changes it in title and z labels
    """
    # Define age and BMI groups
    # age_bins = np.arange(20, 71, 10)
    age_bins = np.array([18, 30, 40, 50, 60, 70])
    # BMI_bins = np.arange(18, 36, 10)
    BMI_bins = np.array([0, 18.5, 25, 30, 35, 60])

    # Compute the average hand strength for each age and BMI group
    age_groups = np.digitize(age, age_bins)
    BMI_groups = np.digitize(BMI, BMI_bins)
    average_hand_strength = np.zeros((len(BMI_bins), len(age_bins)))

    for i in range(1, len(BMI_bins)):
        for j in range(1, len(age_bins)):
            mask = (BMI_groups == i) & (age_groups == j)
            if np.any(mask):
                average_hand_strength[i, j] = np.mean(hand_strength[mask])
            else:
                average_hand_strength[i, j] = np.nan  # Handle empty groups

    # Handle NaNs for visualization (optional) - currently unused, i.e., NO effect because NaN's are changed to np.nan
    avg_hand_strength = np.nan_to_num(average_hand_strength, nan=np.nan)    # Change np.nan to any number to use this

    # Create a meshgrid for the plot
    BMI_grid, age_grid = np.meshgrid(BMI_bins, age_bins)

    # Creating a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(BMI_grid, age_grid, avg_hand_strength.T, cmap='viridis', edgecolor='none')

    # Labels and title
    ax.set_xlabel('BMI (kg/$m^2$)')
    ax.set_ylabel('Age (years)')
    ax.set_zlabel('Hand Strength (kg)')

    plt.title('3D Plot: Average Hand Strength vs BMI and Age Groups')

    # Custom view angle (also optional)
    ax.view_init(elev=elev, azim=azim)

    # Color bar
    clb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    clb.ax.set_ylabel("Hand Strength (kg)")

    plt.tight_layout()
    if streamlit is not True:
        plt.show()

    return fig


def create_plotly_f_of_xy(BMI, age, height_var, height_label="r-hand strength (kg)", elev=20., azim=30, streamlit=True,
                          age_bins=None, BMI_bins=None, dont_start_from_height_zero=True):
    """
    Plot f(x,y) in an interactive 3D plot using plotly.
    Takes np.arrays of x,y and z values, returns a plotly fig to put into st.plotly_chart(fig)
    """
    # Define age and BMI groups
    if age_bins is None:
        # age_bins = np.arange(20, 71, 10)
        age_bins = np.array([18, 30, 40, 50, 60, max(age)])
    if BMI_bins is None:
        # BMI_bins = np.arange(18, 36, 10)
        BMI_bins = np.array([0, 18.5, 25, 30, 35, max(BMI)])

    # Compute the average hand strength (height_var) for each age and BMI group
    age_groups = np.digitize(age, age_bins)
    BMI_groups = np.digitize(BMI, BMI_bins)
    average_hand_strength = np.zeros((len(BMI_bins), len(age_bins)))

    for i in range(1, len(BMI_bins)):
        for j in range(1, len(age_bins)):
            mask = (BMI_groups == i) & (age_groups == j)
            if np.any(mask):
                average_hand_strength[i, j] = np.mean(height_var[mask])
            else:
                average_hand_strength[i, j] = np.nan  # Handle empty groups; can optionally change nans in line below

    # Handle NaNs for visualization (optional) - currently unused, i.e., NO effect because np.nans are changed to np.nan
    average_hand_strength = np.nan_to_num(average_hand_strength, nan=np.nan)

    # Create a meshgrid for the plot
    BMI_grid, age_grid = np.meshgrid(BMI_bins, age_bins)

    # The first column and first row are probably 0 for any height variable. They at least are for hand str and systole
    # Let's remove the first row and first column of x, y and z to have the 3D plot start from heights != 0
    if dont_start_from_height_zero is True:
        age_grid = age_grid[1:, :]   # all columns but not all rows, skip first row (row 0)
        BMI_grid = BMI_grid[:, 1:]   # all rows but not all columns, skip first column (col 0)
        average_hand_strength = average_hand_strength[1:, 1:]   # Skip first col AND first row (there only zero entries in those)

    # Create a 3D surface plot using Plotly
    fig = go.Figure(data=[go.Surface(z=average_hand_strength.T, x=BMI_grid, y=age_grid, colorscale='Viridis')])

    # Update layout for the plot
    fig.update_layout(
        title='3D Plot: Average Hand Strength vs BMI and Age Groups',
        scene=dict(
            xaxis_title='BMI (kg/m²)',
            yaxis_title='Age (years)',
            zaxis_title=height_label,
            # zaxis=dict(nticks=4, range=[np.nanmin(height_var), np.nanmax(height_var)])
            zaxis=dict(nticks=4, range=[np.nanmin(average_hand_strength), np.nanmax(average_hand_strength)])
        )
    )

    # If you are running this code outside of Streamlit, uncomment the next line
    # fig.show()

    return fig


def create_count_matrix(df, height_var="PWC130", type="count", age_bins=None, BMI_bins=None, streamlit_choose_bins=False):
    """
    Creates a pd.DataFrame (matrix). Elements of the matrix show the number of occurrences (only if type="count") of
    "height_var" entries (default: PWC130 entries) in the inputted "df", depending on age groups and BMI groups.

    Args:
        df (pandas.DataFrame): The input DataFrame containing patient data.
        height_var (str, optional): The name of the column containing the height measurement variable.
            Defaults to "PWC130".
        type (str, optional): The type of aggregation to perform. type="count" counts the number of non-null entries
        in the specified variable within each group. Another possibility is type="sum", summing up all respective
        height_var value entries instead of just counting non-null entries. See pd.pivot_table doc for further
        possibilities.
            Defaults to "count".

    Returns:
        pd.DataFrame: A DataFrame (matrix) with age groups as rows and BMI groups as columns.
            Each element represents the number of occurrences (if type="count") specified by "height_var"
            within that particular age and BMI group combination.

    Example: Suppose there are 20 patients with a BMI of 0 to 18.5 kg/m^2 and an age between 18 and 30 years.
    But only 7 of them participated in the PWC130 fitness test, whereas the other 13 patients have a NaN as entry in
    their PWC130 column.
    This function will return a matrix with BMI groups as columns and age groups as rows, where
    the entry for BMI group "0 - 18.5" AND age group "18 - 30" will be 7.
    Why? The occurrence of non-NaN entries are counted. For this particular age and BMI group combination, out of
    the 20 patients who fall into this combination, only 7 actually had some value for PWC130.
    """

    # This is actually just for the std. NAKO table, where BMI is not included, but height and weight are.
    # Should probably put this inside an if and make it work with input DataFrames more generally (no key, ...)
    height_squared = (df["anthro_groe"] / 100) ** 2
    BMI = df["anthro_gew"] / height_squared

    df_with_fixed_ages = df.loc[(df["basis_age"] > 17) & (df["basis_age"] < df["basis_age"].max())]
    age = df_with_fixed_ages["basis_age"]

    # Choose default bins when the bin arguments were omitted (None) or when streamlit_choose_bins was chosen as False
    if age_bins is None or streamlit_choose_bins is False:
        # age_bins = np.array([18, 30, 40, 50, 60, 70, max(age) + .01])
        age_bins = np.array([18, 30, 40, 50, 60, 70, max(age)])  # Edit: nearly noone is in age group 70-80
    if BMI_bins is None or streamlit_choose_bins is False:
        BMI_bins = np.array([0, 18.5, 25, 30, 35, round(max(BMI)) + 1])  # round(max(BMI)) + 1 for rounding up

    # Don't necessarily take the maximum if the user chooses not to do so
    if streamlit_choose_bins is False:
        # Adjust the maximum value in the bins to ensure correct binning
        age_bins[-1] = max(age) + 0.01
        BMI_bins[-1] = round(max(BMI)) + 0.01

    # Digitize the age and BMI columns into bins -> "age_groups" contains the grp indices, each patient is put in
    age_groups = np.digitize(df["basis_age"], age_bins, right=False)
    BMI_groups = np.digitize(BMI, BMI_bins, right=False)

    # Ensure the digitized groups do not exceed the number of bins
    age_groups = np.clip(age_groups, 1, len(age_bins) - 1)
    BMI_groups = np.clip(BMI_groups, 1, len(BMI_bins) - 1)


    # Create a new DataFrame with these groups
    df['AgeGroup'] = age_groups  # E.g. age = 25,35,45 -> age_groups = 1,2,3  // for groups (18-29, 30-40, 40-50)
    df['BMIGroup'] = BMI_groups

    # Create a pivot table counting the number of entries for each age and BMI group
    # // aggfunc='count' only counts "values" (here PWC130 entries) that are not null
    num_occurrences_matrix = df.pivot_table(index='AgeGroup', columns='BMIGroup', values=height_var, aggfunc=type,
                                            fill_value=0)
    # TODO: The final line of the pivot table "num_occurrences_matrix" is the issue! I don't know where it is coming from, though!

    # Filter out rows where the target (e.g. PWC130) is not available (no entry/NaN/null)
    # num_occurrences_matrix = num_occurrences_matrix.dropna(subset=['PWC130'])

    # Set the age and BMI bin labels
    # age_labels = [f'{round(age_bins[i - 1])} - {round(age_bins[i]) - 1}' for i in range(1, len(age_bins))]
    age_labels = [
        f'{round(age_bins[i - 1])} - {round(age_bins[i]) - 1}' if i < len(age_bins) - 1  # Last age excluded
        else f'{round(age_bins[i - 1])} - {round(age_bins[i])}'  # Final age (here 80) included
        for i in range(1, len(age_bins))]
    BMI_labels = [f'{BMI_bins[i - 1]} - {round(BMI_bins[i], 1) - .1}' if i < len(BMI_bins) - 1
                  else f'{BMI_bins[i - 1]} - {round(BMI_bins[i], 1)}'
                  for i in range(1, len(BMI_bins))]

    exception_occurred = False
    try:
        # num_occurrences_matrix.index = age_labels
        # num_occurrences_matrix.columns = BMI_labels
        num_occurrences_matrix.index = pd.Index(range(1, len(age_bins)), name='AgeGroup')
        num_occurrences_matrix.columns = pd.Index(range(1, len(BMI_bins)), name='BMIGroup')
        # num_occurrences_matrix = num_occurrences_matrix.reindex(index=range(1, len(age_bins)),
        #                                                         columns=range(1, len(BMI_bins)), fill_value=0)
        num_occurrences_matrix.index = age_labels
        num_occurrences_matrix.columns = BMI_labels

    except Exception as e:
        exception_occurred = True

        # Cheap fix for age and BMI labels if there was one label less than pivot_table rows&columns.
        # Won't work if <only the age labels> or <only the BMI labels> are erroneous! (And is generally bad, TODO)
        age_labels.append("Error")
        num_occurrences_matrix.index = age_labels
        BMI_labels.append("Error")
        num_occurrences_matrix.columns = BMI_labels

        warnings.warn(f"Attention! Caught an Exception:"
                      f"\nThere was at least one example of age groups or BMI groups where the used data doesn't include "
                      f"even a single example. This is not fixed currently and results in the matrix labels just being the "
                      f"original numbers. "
                      f"\nFor reference, the exception was: \n{e}")
    # finally:
        # if exception_occurred:
        #     print("TODO: The 'fix' below doesn't work. It correctly changes the column names and row names of the "
        #           "matrix -- but in num_occurrences_matrix.reindex, all actual matrix entries, i.e., all non-zero "
        #           "elements that show the counts/number of occurrences, are changed to 0!")
            # full_index = pd.Index(range(1, len(age_bins)), name='AgeGroup')
            # full_columns = pd.Index(range(1, len(BMI_bins)), name='BMIGroup')
            # num_occurrences_matrix = num_occurrences_matrix.reindex(index=full_index, columns=full_columns,
            #                                                         fill_value=0)
            # # Assign the correct labels to the index and columns
            # num_occurrences_matrix.index = age_labels
            # num_occurrences_matrix.columns = BMI_labels

    # Visualize the matrix using a heatmap
    fig = plt.figure(figsize=(10, 6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("", ["red", "green"])
    sns.heatmap(num_occurrences_matrix, annot=True, fmt="d", cmap=cmap,  # cmap="YlGnBu",
                cbar_kws={"label": f"Amount of successful {height_var} measurements per subgroup"})
    plt.xlabel('BMI Group')
    plt.ylabel('Age Group')
    plt.title(f'Number of Patients with {height_var} measurement by Age and BMI Groups')

    # If you are running this code outside of Streamlit, uncomment the next line
    # plt.show()

    return fig


def height_label_naming(height_variable_option):
    """
    Obtain a more meaningful label, depending on the input string.
    Args:
         height_variable_option (str):
         Existing column name in currently used pd.DataFrame
    Returns (str):
        More described height label string to use in plots

    Example:
        Changes input column string "hgr_rh_kraft_mean" to "right hand strength (kg)".
        The latter includes the unit and overall improved clarity.
    """

    if height_variable_option == "hgr_rh_kraft_mean":
        height_label = "right hand strength (kg)"
    elif "fett" in height_variable_option:
        height_label = f"{height_variable_option} (%)"
    elif "ff" in height_variable_option:
        height_label = f"{height_variable_option}"
    else:
        height_label = f"{height_variable_option} (a.u.)"

    return height_label

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Example data
    np.random.seed(42)
    BMI = np.random.uniform(18, 35, 100)
    hand_strength = np.random.uniform(10, 60, 100)
    age = np.random.uniform(20, 70, 100)

    # Define age and BMI groups
    age_bins = np.arange(20, 71, 10)
    BMI_bins = np.arange(18, 36, 5)

    # Compute the average hand strength for each age and BMI group
    age_groups = np.digitize(age, age_bins)
    BMI_groups = np.digitize(BMI, BMI_bins)
    average_hand_strength = np.zeros((len(BMI_bins), len(age_bins)))

    for i in range(1, len(BMI_bins)):
        for j in range(1, len(age_bins)):
            mask = (BMI_groups == i) & (age_groups == j)
            if np.any(mask):
                average_hand_strength[i, j] = np.mean(hand_strength[mask])
            else:
                average_hand_strength[i, j] = np.nan  # Handle empty groups

    # Handle NaNs for visualization (optional)
    average_hand_strength = np.nan_to_num(average_hand_strength, nan=np.nan)

    # Plot the data
    plot_f_of_xy(BMI_bins, age_bins, average_hand_strength)
