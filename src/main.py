import logging

import gin
from absl import app, flags
from data.preprocess import DataPreprocessor
from utils import utils_misc, utils_params

# Define different arguments for the main
# e.g. WANDB API KEY...
FLAGS = flags.FLAGS

# TODO might remove to get train, eval and test included inside of single pipeline...
flags.DEFINE_boolean("train", True, "Specify if train mode or eval mode.")
flags.DEFINE_string(
	"experiment_dir",
	None,
	"Specify folder to resume training, otherwise train from scratch",
)


# TODO remove, only for debugging
FLAGS.experiment_dir = "debug"


def main(argv) -> None:
	# Parse gin config
	run_paths = utils_params.gen_run_folder(FLAGS.experiment_dir)
	utils_misc.set_loggers(run_paths["path_logs_train"], logging.INFO)

	gin.parse_config_files_and_bindings(["configs/config.gin"], [])
	dpc = DataPreprocessor()
	return


if __name__ == "__main__":
	app.run(main)
