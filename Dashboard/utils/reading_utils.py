from pathlib import Path
from typing import Union

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup


def read_html_from_path(path: Union[str, Path]) -> BeautifulSoup:
    """Create soup object directly from path.

    :param path: path to html file
    :type path: Union[str, Path]
    :return: soup object which can be scraped
    :rtype: BeautifulSoup
    """
    with open(str(path)) as file:
        html_content = file.read()
    return BeautifulSoup(html_content, "html.parser")


@st.cache_data
def read_csv_file_cached(path: Union[str, Path], sep: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
    """Read CSV file using pandas.
    The backend remains exatcly the same, the only difference is
    that the function shall be cached due to usage in streamlit.

    :param path: path to the file to be read.
    :type path: Union[str, Path]
    :param sep: seperator used in the CSV file.
    :type sep: str
    :param encoding: encoding used.
    :type encoding: str
    :return: Dataframe similar to read_csv
    :rtype: pd.DataFrame
    """
    return pd.read_csv(path, sep=sep, encoding=encoding, on_bad_lines="warn")


if __name__ == "__main__":
    import warnings
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    import matplotlib
    matplotlib.use('TkAgg')  # To fix seaborn (sns) UserWarning on plt.show()


    def create_count_matrix(df, height_var="PWC130", type="count"):
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

        # Define the age and BMI bins
        age_bins = np.array([18, 30, 40, 50, 60, 70, df['basis_age'].max() + .01])
        BMI_bins = np.array([0, 18.5, 25, 30, 35, df['BMI'].max() + .01])

        # Digitize the age and BMI columns into bins -> "age_groups" contains the grp indices, each patient is put in
        age_groups = np.digitize(df["basis_age"], age_bins, right=False)
        BMI_groups = np.digitize(df["BMI"], BMI_bins, right=False)

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

        # Filter out rows where the target (e.g. PWC130) is not available (no entry/NaN/null)
        # num_occurrences_matrix = num_occurrences_matrix.dropna(subset=['PWC130'])

        # Set the age and BMI bin labels
        # age_labels = [f'{round(age_bins[i - 1])} - {round(age_bins[i]) - 1}' for i in range(1, len(age_bins))]
        age_labels = [
            f'{round(age_bins[i - 1])} - {round(age_bins[i]) - 1}' if i < len(age_bins) - 1  # Last age excluded
            else f'{round(age_bins[i - 1])} - {round(age_bins[i])}'  # Final age (here 80) included
            for i in range(1, len(age_bins))]   # i = 1, 2, 3, 4, 5, 6 if len(age_bins) = 7
        BMI_labels = [f'{BMI_bins[i - 1]} - {round(BMI_bins[i], 1) - .1}' if i < len(BMI_bins) - 1
                      else f'{BMI_bins[i - 1]} - {round(BMI_bins[i], 1)}'
                      for i in range(1, len(BMI_bins))]

        exception_occurred = False
        try:
            num_occurrences_matrix.index = age_labels
            num_occurrences_matrix.columns = BMI_labels

        except Exception as e:
            exception_occurred = True
            warnings.warn(f"Attention! Caught an Exception:"
                          f"\nThere was at least one example of age groups or BMI groups where the used data doesn't include "
                          f"even a single example. This is not fixed currently and results in the matrix labels just being the "
                          f"original numbers. "
                          f"\nFor reference, the exception was: \n{e}")
        finally:
            if exception_occurred:
                print("TODO: The 'fix' below doesn't work. It correctly changes the column names and row names of the "
                      "matrix -- but in num_occurrences_matrix.reindex, all actual matrix entries, i.e., all non-zero "
                      "elements that show the counts/number of occurrences, are changed to 0!")
                # full_index = pd.Index(range(1, len(age_bins)), name='AgeGroup')
                # full_columns = pd.Index(range(1, len(BMI_bins)), name='BMIGroup')
                # num_occurrences_matrix = num_occurrences_matrix.reindex(index=full_index, columns=full_columns,
                #                                                         fill_value=0)
                # # Assign the correct labels to the index and columns
                # num_occurrences_matrix.index = age_labels
                # num_occurrences_matrix.columns = BMI_labels

        # Visualize the matrix using a heatmap
        plt.figure(figsize=(10, 6))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("", ["red", "green"])
        sns.heatmap(num_occurrences_matrix, annot=True, fmt="d", cmap=cmap,  # cmap="YlGnBu",
                    cbar_kws={"label": f"Amount of successful {height_var} measurements per subgroup"})
        plt.xlabel('BMI Group')
        plt.ylabel('Age Group')
        plt.title(f'Number of Patients with {height_var} measurement by Age and BMI Groups')
        plt.show()

        return num_occurrences_matrix

    # Example usage
    data = {
        'basis_age': [25, 35, 45, 55, 65, 30, 40, 50, 60, 70, 25, 45, 55, 65, 72, 0],
        'BMI': [17, 27, 32, 28, 24, 26, 31, 33, 23, 25, 22, 27, 32, 28, 40, 33],
        'PWC130': [100, 110, None, 130, 140, None, 150, None, 140, 170, 180, 190, 200, 210, None, 199]
    }

    df = pd.DataFrame(data)
    count_matrix = create_count_matrix(df)
    print(count_matrix)
