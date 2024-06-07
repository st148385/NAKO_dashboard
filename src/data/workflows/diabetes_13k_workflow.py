import logging
import math
from pathlib import Path

import gin
import polars as pl

from data.utils.preprocessing_utils import filter_data_by_mapping_dict_nako, get_mapping_from_metadata_nako
from utils.constants import DATA_PATH, IGNORE_VALUE, INPUT_ROOT, INPUTS, METADATA_PATH, OPTIONAL_INPUTS, OUTPUT_ROOT
from utils.reading import read_data_with_polars
from .abstractworkflow import AbstractWorkflow


# Assuming df is your Polars DataFrame containing sa_ogtt0 and sa_ogtt2 columns
def classify_diabtes_severity(row):
    sa_ogtt0, sa_ogtt2 = row["sa_ogtt0"], row["sa_ogtt2"]

    # Due to remapping some values are "nan"
    if (
            sa_ogtt0 is None
            or (isinstance(sa_ogtt0, float) and math.isnan(sa_ogtt0))
            or sa_ogtt2 is None
            or (isinstance(sa_ogtt2, float) and math.isnan(sa_ogtt2))
    ):
        return IGNORE_VALUE
    if sa_ogtt0 >= 7 or sa_ogtt2 >= 11.1:
        return 0  # Class 0
    elif 6.1 <= sa_ogtt0 < 7:
        return 1  # Class 1
    elif 7.8 <= sa_ogtt2 < 11.1:
        return 2  # Class 2
    elif sa_ogtt0 < 6.1 and sa_ogtt2 < 7.8:
        return 3  # Class 3
    else:
        raise ValueError(
            """
            All cases of oGTT values should be caught and a class (0, 1, 2 or 3) determined for them, 
            with the current class definitions. We can't get here! Look for mistakes in the if statement. \n
            The problem occurred for sa_ogtt0 = {sa_ogtt0} and sa_ogtt2 = {sa_ogtt2}"""
        )


@gin.configurable
class Diabetes13kWorkflow(AbstractWorkflow):
    """
    Workflow for processing the Diabetes13k dataset.

    This workflow inherits from :class:`AbstractWorkflow` and provides specific implementations
    for preprocessing, processing, and postprocessing steps tailored to the Diabetes13k dataset.

    :param \**kwargs: Additional keyword arguments to pass to the parent class (``AbstractWorkflow``).
    """

    def __init__(self, **kwargs):
        """
        Initializes the ``Diabetes13kWorkflow``.

        :param \**kwargs: Keyword arguments passed to the base class constructor.
        """
        super().__init__(**kwargs)

        # Use pathlib for cleaner path handling
        metadata_path = Path(self.path_collection[INPUT_ROOT]) / self.path_collection[INPUTS][METADATA_PATH]
        data_path = Path(self.path_collection[INPUT_ROOT]) / self.path_collection[INPUTS][DATA_PATH]
        self.output_dir = Path(self.path_collection.get(OUTPUT_ROOT, "."))  # Default to current dir if not provided

        self.metadata = read_data_with_polars(
            metadata_path, separator=";", encoding="utf-8", infer_schema_length=0, truncate_ragged_lines=True
        )
        self.data = read_data_with_polars(data_path, separator=";", encoding="latin1")

        # Merge additional data (pathlib handling)
        for data_name, rel_path in self.path_collection.get(OPTIONAL_INPUTS, {}).items():
            logging.info(f"Merging '{data_name}' data...")
            data_path = Path(self.path_collection[INPUT_ROOT]) / rel_path
            additional_data = read_data_with_polars(data_path, separator=";", encoding="latin1")
            self.data = self.data.join(additional_data, on="ID", how="left")

    def preprocess(self, data):
        """
        Preprocesses the Diabetes13k dataset.

        This method first calls the general preprocessing of the parent class and then
        performs additional preprocessing specific to the Diabetes13k dataset.

        :param data: The input data to be preprocessed.
        :type data: pl.DataFrame
        :return: The preprocessed data.
        :rtype: pl.DataFrame

        """

        # General preprocessing
        data = super().preprocess(data)
        # TODO: Add dataset-specific preprocessing steps here.

        # Get mapping dict using the metadata
        self.mapping_dict = get_mapping_from_metadata_nako(self.metadata)
        # Replace Missing, unknown not valid data with nans
        self.data = filter_data_by_mapping_dict_nako(self.data, self.mapping_dict)

        ########################################################
        ####### use WHO definition regarding sa_ogtt to ########
        ####### 	create diabetes_severity column		########
        data = data.with_columns(
            pl.struct(["sa_ogtt0", "sa_ogtt2"])
            .apply(lambda row: classify_diabtes_severity(row))
            .alias("diabetes_class")
        )
        # Artifically add clas and update the preprocess_basis
        # otherwise the class will be removed by _filter_by_config
        # Since it is not in there...
        self._feature_info.update({"diabetes_class": {"transforms": [None], "dtype": "int", "type": "ordinal"}})
        ########################################################

        # The manually choosen stuff will be done here then.

        # TODO might need to rethink this, if artifcially adding stuff here
        # this won't appear in the original config.

        data = self._filter_by_config(data)

        return data

    def process(self, data):
        """
        Processes the preprocessed Diabetes13k data.

        This method currently delegates processing to the parent class.

        :param data: The preprocessed data to be processed.
        :type data: Any
        :return: The processed data.
        :rtype: Any

        """
        return data

    def postprocess(self, data):
        """
        Postprocesses the processed Diabetes13k data.

        This method currently delegates postprocessing to the parent class.

        :param data: The processed data to be postprocessed.
        :type data: Any
        :return: The postprocessed data.
        :rtype: Any

        """
        return data

    def run(self):
        super().run()

        # Save the data optionally if output dir is given
        if OUTPUT_ROOT in self.path_collection:
            output_root = Path(self.path_collection[OUTPUT_ROOT])
            input_root = Path(self.path_collection[INPUT_ROOT])

            for path_key, key in zip([DATA_PATH, METADATA_PATH], ["data", "metadata"]):
                rel_path = input_root / self.path_collection[INPUTS][path_key]
                output_path = output_root / rel_path.relative_to(input_root)
                self.save_to_csv(output_path, getattr(self, key.lower()), overwrite=True)

        return self.data
