# DATASET specific configuration
METADATA_CONSTANTS = {
	"metadata_filename": "NAKO-536_2023-07-13_Erstuntersuchung_MRT_PSN_wie_194_Metadaten.csv",
	"data_filename": "NAKO-536_2023-07-13_Erstuntersuchung_MRT_PSN_wie_194.csv",
	"HTML_filename": "NAKO-536 - 2023-07-13 - Beschreibung des Übergabedatensatzes.html",
	"metadata_seperator": ";",
	"data_seperator": ";",
	"encoding": "latin1",
}

DIABETES_CONSTANTS = {
	"metadata_filename": "NAKO_536_61223_export_baseln_MRT_PSN_wie_194_Metadaten.csv",
	"data_filename": "NAKO_536_61223_export_baseln_MRT_PSN_wie_194.csv",
	"HTML_filename": "NAKO-536 - 2023-12-06 - Beschreibung des Übergabedatensatzes.html",
	"metadata_seperator": ";",
	"data_seperator": ";",
	"encoding": "latin1",
}

DATASETS_CSV = {"30k_metadata": METADATA_CONSTANTS, "13k_diabetes": DIABETES_CONSTANTS}

# List all datasets avaiable...
DATASETS = list(DATASETS_CSV.keys())


IGNORE_VALUE = float("nan")


# DATA representation
MAX_GROUPBY_NUMBER = 3
