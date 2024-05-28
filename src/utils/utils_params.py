import datetime
from pathlib import Path


def gen_run_folder(experiment_dir=None):
	"""Generates a run folder for an experiment.

	This function determines the appropriate path for storing experiment data based
	on the input and creates the necessary directories and files.

	**Behavior:**

	- **Valid Path:** If `experiment_dir` is a valid path, it will be used as the
	  experiment directory.
	- **String (Not Path):** If `experiment_dir` is a string but not a valid path,
	  a new folder with that name will be created within the default "experiments"
	  directory.
	- **None:** If `experiment_dir` is None, a new timestamped folder will be
	  created within the default "experiments" directory.

	:param experiment_dir: A string or Path object representing the experiment
		directory. Can be a full path, a relative path, or just the folder name.
	:type experiment_dir: str or pathlib.Path, optional

	:raises ValueError: If `experiment_dir` is a Path object that is invalid or
		does not reside within the project's "experiments" directory.

	:return: A dictionary containing the following paths:

		* **path_model_id:** The full path to the experiment directory.
		* **model_id:** The name of the experiment directory.
		* **path_logs_train:** The full path to the training log file.
		* **path_ckpts_train:** The full path to the training checkpoints directory.
		* **path_gin:** The full path to the operative configuration file.
	:rtype: dict
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
