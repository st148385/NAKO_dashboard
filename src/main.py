import logging
from pathlib import Path

import gin
from absl import app, flags
from data.dataloders import ScikitLearnDataloader, TensorflowDataloader
from data.workflows import Diabetes13kWorkflow, Metadata30kWorkflow
from utils import utils_misc, utils_params

# Define different arguments for the main
# e.g. WANDB API KEY...
FLAGS = flags.FLAGS

# TODO might remove to get train, eval and test included inside of single pipeline...
flags.DEFINE_boolean("train", True, "Specify if train mode or eval mode.")
flags.DEFINE_string(
	"experiment_dir", None, "Specify folder to resume training, otherwise train from scratch", short_name="e"
)

flags.DEFINE_string("config_file", None, "Specify the configuration file to use, e.g. train_config.gin", short_name="c")


def main(argv) -> None:
	# Parse gin config
	run_paths = utils_params.gen_run_folder(FLAGS.experiment_dir)
	utils_misc.set_loggers(run_paths["path_logs_train"], logging.INFO)

	# Parse provided config file
	config_file = [Path("configs") / FLAGS.config_file]
	gin.parse_config_files_and_bindings(config_file, [])

	dpc = TensorflowDataloader()

	print(dpc.batch_size, dpc.data)
	return


if __name__ == "__main__":
	app.run(main)
