import gin

###### DATAPATH SPECIFIC CONSTANTS
## ROOT paths
INPUT_ROOT = "input_root"
OUTPUT_ROOT = "output_root"
# INPUT KEYS
INPUTS = "inputs"
OPTIONAL_INPUTS = "optional_inputs"
METADATA_PATH = "metadata_path"
DATA_PATH = "data_path"


# IGNORE VAL
IGNORE_VALUE = float("nan")


def register_constants_with_gin():
	"""Registers all variables from constants.py with Gin."""
	for name, value in globals().items():
		if name.isupper() and not name.startswith("__"):  # Filter constants
			gin.constant(name, value)
