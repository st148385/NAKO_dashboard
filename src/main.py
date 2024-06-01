import argparse
import logging
from pathlib import Path

import gin
import gin_registry
from data.dataloaders import DataLoaderFactory
from data.workflows import WorkflowFactory
from models import ModelFactory
from training import Runner
from utils import utils_misc, utils_params


def parse_args():
	parser = argparse.ArgumentParser(description="Train or evaluate a model")

	# Clear and concise argument definitions
	parser.add_argument("--train", action="store_true", default=True, help="Enable training mode (default)")
	parser.add_argument("-e", "--experiment_dir", help="Experiment dir to log stuff and save checkpoints")
	parser.add_argument(
		"-c", "--config_file", required=True, help="Path to the configuration file (e.g., train_config.gin)"
	)
	parser.add_argument("-wb", "--wandb_api_key", help="API key for Weights & Biases tracking")
	parser.add_argument(
		"-s",
		"--scope",
		choices=["classification", "regression"],
		required=True,
		help="scope type either regression or classification, case sensitive",
	)

	args = parser.parse_args()
	return args


def main():
	args = parse_args()

	# Generate run folders and set up logging
	run_paths = utils_params.gen_run_folder(args.experiment_dir)
	utils_misc.set_loggers_with_rich(run_paths["path_logs_train"], logging.INFO)

	# Parse configuration file
	gin.bind_parameter("DataLoaderFactory.scope", args.scope)
	config_file = [Path("configs").resolve() / args.scope / args.config_file]
	gin.parse_config_files_and_bindings(config_file, [])

	# Create workflow and preprocess data
	workflow_factory = WorkflowFactory()
	data = workflow_factory.run()

	# Create data loader and process data
	dataloader_factory = DataLoaderFactory(data=data, scope=args.scope)
	dataloader = dataloader_factory.dataloader
	_, _, ds_info = dataloader.get_datasets()

	# Create model
	model_factory = ModelFactory(ds_info=ds_info)

	# Initialize and run the training/evaluation process
	runner = Runner(
		model=model_factory.model, dataloader=dataloader, run_paths=run_paths, wandb_api_key=args.wandb_api_key
	)
	runner.run()


if __name__ == "__main__":
	main()
