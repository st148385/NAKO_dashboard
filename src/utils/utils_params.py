import datetime
from pathlib import Path


def gen_run_folder(experiment_dir=None):
	"""
	Generates a run folder for an experiment.

	Behavior:
	- If `experiment_dir` is a valid path, use it as is.
	- If `experiment_dir` is a string (not a path), create a new folder within
	  the "experiments" directory with that name.
	- If `experiment_dir` is None, generate a new folder using a timestamp.

	Args:
	    experiment_dir: Optional string or Path object representing the experiment directory.

	Returns:
	    A dictionary containing paths for the run folder, logs, checkpoints, and configuration.
	"""

	root_dir = Path(__file__).resolve().parents[2] / "experiments"

	if isinstance(experiment_dir, str):
		model_path = root_dir / experiment_dir
	elif isinstance(experiment_dir, Path):
		# If Path object, check if it's a directory and within the root_dir
		if experiment_dir.is_dir() and root_dir in experiment_dir.parents:
			model_path = experiment_dir
		else:
			raise ValueError(f"Invalid experiment_dir path: {experiment_dir}")
	else:
		# If None or not a valid path, generate a timestamped folder
		model_id = f"run_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')}"
		model_path = root_dir / model_id

	# Create model directory if it doesn't exist
	model_path.mkdir(parents=True, exist_ok=True)

	run_paths = {
		"path_model_id": model_path,
		"model_id": model_path.name,
		"path_logs_train": model_path / "logs" / "run.log",
		"path_ckpts_train": model_path / "ckpts",
		"path_gin": model_path / "config_operative.gin",
	}

	# Create log directory and ensure the log file exists
	(model_path / "logs").mkdir(exist_ok=True)
	(run_paths["path_logs_train"]).touch(exist_ok=True)

	return run_paths


def save_config(path_gin, config):
	with open(path_gin, "w") as f_config:
		f_config.write(config)
