import logging
from pathlib import Path

import gin
from absl import app, flags
from data.dataloaders import DataLoaderFactory
from data.workflows import WorkflowFactory
from models import ModelFactory
from training import Runner
from utils import utils_misc, utils_params, utils_register_gin

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

	# Create Workflow. Preprocess data
	workflow_factory = WorkflowFactory()
	data = workflow_factory.run()

	# Create DataLoader, Process data to correct format
	dataloader_factory = DataLoaderFactory(data=data)
	dataloader = dataloader_factory.dataloader
	train_ds, _ = dataloader.get_datasets()

	# Load model with correct shapes
	for batch in train_ds:
		input_shape = batch["features"].shape[1:]
		output_shape = batch["labels"].shape[1:]
		break
	wrapper_model = ModelFactory(input_shape=input_shape, output_shape=output_shape)

	# Init runner with dataloader and model.
	runner = Runner(model=wrapper_model.model, dataloader=dataloader, run_paths=run_paths)
	runner.train()

	return


if __name__ == "__main__":
	app.run(main)
