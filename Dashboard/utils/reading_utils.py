from pathlib import Path
from typing import Union

import numpy as np
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


def get_diabetes_3classes_from_hba1c(diabetes_df, hba1c_col_name="sa_hba1c"):
    """
    This function handles the classification of patients into three categories based on their HbA1c levels (see WHO).

    Classes:
    - Diabetes Mellitus (DM): HbA1c >= 48 mmol/mol      // class 0
    - Prediabetes: 42 <= HbA1c < 48 mmol/mol            // class 1
    - Healthy: HbA1c < 42 mmol/mol                      // class 2
    - (Missing HbA1c measurement)                       // class -1

    :param diabetes_df: pd.DataFrame which must have the column <hba1c_col_name> (currently, this is "sa_hba1c")
    :param hba1c_col_name: Column name of the HbA1c column in the pd.DataFrame <diabetes_df>
    :return: Array of diabetes class labels per patients for the 3-class problem (prediabetes NOT split into IFG/IGT!)
    """
    # labels "y" using HbA1c for just 3 classes (0: DM, 1: Prediabetes, 2: Healthy)
    conditions = [
        (diabetes_df[hba1c_col_name] >= 48),  # HbA1c >= 48
        (diabetes_df[hba1c_col_name] >= 42) & (diabetes_df[hba1c_col_name] < 48),  # 42 <= HbA1c < 48
        (diabetes_df[hba1c_col_name] < 42),  # HbA1c < 42
        (np.isnan(diabetes_df[hba1c_col_name]))
    ]

    # labels = [0, 1, 2]  # 0 for >= 48 (DM) || 1 for 42 <= HbA1c < 48 (Prediabetes) || 2 for < 42 (Healthy)
    labels = [0, 1, 2, -1]  # 0 for >= 48 (DM) || 1 for 42 <= HbA1c < 48 (Prediabetes) || 2 for < 42 (Healthy)

    # Initialize the y array with NaNs
    y = np.full(diabetes_df.shape[0], np.nan)
    # Apply the conditions and assign labels
    for label, condition in zip(labels, conditions):
        y[condition] = label
    # Convert y to integer type
    return y.astype(int)


def get_diabetes_4classes_from_ogtt(ogtt0h, ogtt2h, which_definition="WHO"):
    """
    Warning: The table's oGTT values are in mmol/l

    Lena diabetes groups:
    1.    manifester DM: Nüchternplasmaglukose (ogTT 0h) ≥ 125 mg/dL und/oder 2h-Plasmaglukose nach 75 g oGTT (oGTT 2h) ≥ 200 mg/dL

    2.    Prädiabetes:
    a.    erhöhter Nüchternblutzucker (IFG: impaired fasting gycaemia):  Nüchternplasmaglukose 110 bis 125 mg/dL
    b.    gestörte Glukosetoleranz (IGT: impaired glucose tolerance): 2h-Plasmaglukose nach 75 g oGTT 140 bis 200 mg/dL

    3.    Normoglykämie: Nüchternplasmaglukose < 110 mg/dL und 2h-Plasmaglukose nach 75 g oGTT < 140 mg/dL

    Umrechnungsfaktor von mmol/L in mg/dL ist:
        1 mmol/L = 180.16 g/mol * 1 mmol * 1/10 * 1/dL = 0.018 g/dL = 18 mg/dL
    Rückrichtung für mg/dL in mmol/L:
        1 mg/dL = 1/18 mmol/L = 0.0555 mmol/L

    Diabetes classes (X := ogtt0h; Y := ogtt2h):
    [0]: oGTT 0h ≥ 125 mg/dl  OR  oGTT 2h ≥ 200 mg/dl               // manifester DM
    ->   X ≥ 6.94 mmol/l      OR  Y ≥ 11.11 mmol/l
    [1]: 110 mg/dl ≤ oGTT 0h < 125 mg/dl  ->  6.11 ≤ X < 6.94       // Prädiabetes: erhöhter Nüchternblutzucker (IFG)
    [2]: 140 mg/dl ≤ oGTT 2h < 200 mg/dl  ->  7.78 ≤ Y < 11.11      // Prädiabetes: gestörte Glukosetoleranz (IGT)
    [3]: oGTT 0h < 110 mg/dl  AND  oGTT 2h < 140 mg/dl              // Normoglykämie
    ->   X < 6.11             AND  Y < 7.78

    Nachtrag, nach obiger "exakter" Umrechnung:
    Changed it to the values given by WHO diabetes diagnostic criteria:
    [0]: X ≥ 7 mmol/l  OR  Y ≥ 11.1 mmol/l          // Diabetes mellitus (DM)
    [1]: 6.1 ≤ X < 7                                // IFG
    [2]  7.8 ≤ Y < 11.1                             // IGT
    [3]: X < 6.1  AND  Y < 7.8                      // Normal

    Sources for the latter definition:
    1.) Definition and diagnosis of diabetes mellitus and intermediate hyperglycemia: Report of a WHO/IDF
    consultation (PDF). Geneva: World Health Organization. 2006. p. 21. ISBN 978-92-4-159493-6.
    2.)  Vijan S (March 2010). "In the clinic. Type 2 diabetes". Annals of Internal Medicine. 152 (5): ITC31-15, quiz
    ITC316. doi:10.7326/0003-4819-152-5-201003020-01003. PMID 20194231.

    :return: Diabetes classes instead of oGTT measurements (array of {0,1,2,3} instead of array of positive real numbers)
    """

    if not np.max(ogtt0h) < 100 or not np.max(ogtt2h) < 100:  # By now, all "missings" indicator values should be gone
        assert False, (f"The oGTT values don't make sense. Did the filtering fail? Are there still any indicator "
                       f"values for 'missings' included in the data? \n "
                       f"The maximums are {np.max(ogtt0h) = } and {np.max(ogtt2h) = } \n"
                       f"Reminder: The 'missings' indicators are 111111, 222222, 333333, 4444444, 555555, 666666 and"
                       f"999999.")

    # if ogtt0h.isnull().any() or ogtt2h.isnull().any():
    #     assert False, "There is a nan in the ogtt0h or ogtt2h data"

    diabetes_classes_array = np.array([])  # empty np.array

    if which_definition not in "WHO":
        # The following were my calculation results
        print(f"Applying own conversion results of Lena's mg/dl oGTT0h/oGTT2h class limits")
        for k in range(len(ogtt0h)):
            # Diabetes classes as four classes 0, 1, 2 and 3
            if ogtt0h.values[k] >= 6.94 or ogtt2h.values[k] >= 11.11:
                diabetes_classes_array = np.append(diabetes_classes_array, values=0)  # Class 0
            elif 6.11 <= ogtt0h.values[k] < 6.94:
                diabetes_classes_array = np.append(diabetes_classes_array, values=1)  # Class 1
            elif 7.78 <= ogtt2h.values[k] < 11.11:
                diabetes_classes_array = np.append(diabetes_classes_array, values=2)  # Class 2
            elif ogtt0h.values[k] < 6.11 and ogtt2h.values[k] < 7.78:
                diabetes_classes_array = np.append(diabetes_classes_array, values=3)  # Class 3
            else:
                assert False, (f"All cases of oGTT values should be caught and a class (0, 1, 2 or 3) determined for "
                               f"them, with the current class definitions. We can't get here! Look for mistakes in the "
                               f"if statement. \n"
                               f"The problem occurred for ogtt0h = {ogtt0h.values[k]} and ogtt2h = {ogtt2h.values[k]}, "
                               f"while {k = }")

    else:
        # The following are the WHO diabetes diagnostic criteria values when using mmol/L as the unit
        print("Applying the WHO diabetes diagnostic criteria on oGTT 0h and oGTT 2h")
        for k in range(len(ogtt0h)):
            # Diabetes classes as four classes 0, 1, 2 and 3
            if ogtt0h.values[k] >= 7 or ogtt2h.values[k] >= 11.1:
                diabetes_classes_array = np.append(diabetes_classes_array, values=0)  # Class 0
            elif 6.1 <= ogtt0h.values[k] < 7:
                diabetes_classes_array = np.append(diabetes_classes_array, values=1)  # Class 1
            elif 7.8 <= ogtt2h.values[k] < 11.1:
                diabetes_classes_array = np.append(diabetes_classes_array, values=2)  # Class 2
            elif ogtt0h.values[k] < 6.1 and ogtt2h.values[k] < 7.8:
                diabetes_classes_array = np.append(diabetes_classes_array, values=3)  # Class 3
            else:
                assert False, (f"All cases of oGTT values should be caught and a class (0, 1, 2 or 3) determined for "
                               f"them, with the current class definitions. We can't get here! Look for mistakes in the "
                               f"if statement. \n"
                               f"The problem occurred for ogtt0h = {ogtt0h.values[k]} and ogtt2h = {ogtt2h.values[k]}, "
                               f"while {k = }")

    return diabetes_classes_array


if __name__ == "__main__":
    import warnings
    import pandas as pd
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
