import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def set_loggers_with_rich(path_log=None, logging_level=logging.INFO, b_debug=False):
	custom_theme = Theme(
		{
			"logging.level.debug": "dim",
			"logging.level.info": "dim cyan",
			"logging.level.warning": "magenta",
			"logging.level.error": "red",
			"logging.level.critical": "bold red",
		}
	)

	rich_handler = RichHandler(
		level=logging_level,
		rich_tracebacks=True,
		console=Console(theme=custom_theme),
		show_time=True,
		show_path=True,
		markup=True,
	)

	# Update the log message format for RichHandler (add %(filename)s)
	rich_handler.setFormatter(
		logging.Formatter("[%(asctime)s] %(levelname)s - %(filename)s:%(funcName)s - %(message)s", datefmt="%X")
	)

	logging.basicConfig(
		level=logging_level if not b_debug else logging.DEBUG,
		format="%(message)s",
		datefmt="[%X]",
		handlers=[rich_handler],
	)

	if path_log:
		# Update the log message format for file_handler (add %(filename)s)
		file_handler = logging.FileHandler(path_log)
		file_handler.setFormatter(
			logging.Formatter("[%(asctime)s] %(levelname)s - %(filename)s:%(funcName)s - %(message)s", datefmt="%X")
		)
		logging.getLogger().addHandler(file_handler)


def log_dict(input_dict, headline="", indent=0):
	"""Logs nested dictionaries with indentation, handling various data types."""

	# Handle None and non-printable types
	if input_dict is None:
		input_dict = "None"
	elif not isinstance(input_dict, (str, dict)):
		input_dict = repr(input_dict)  # Use repr() for non-printable types

	# Base case: if not a dict, just log the key-value pair
	if not isinstance(input_dict, dict):
		logging.info(" " * indent + f"| {headline:<20} | {input_dict:>20} |")
		return

	# Print headline if provided
	if headline:
		logging.info(" " * indent + "=" * 40)
		logging.info(" " * indent + f"{headline}:")
		logging.info(" " * indent + "=" * 40)

	# Recursively process nested dictionaries
	for key, value in input_dict.items():
		new_headline = f"{key}:" if isinstance(value, dict) else str(key)
		log_dict(value, new_headline, indent + 2)

	# Print bottom border if headline was provided
	if headline:
		logging.info(" " * indent + "=" * 40)


def gin_config_to_readable_dictionary(gin_config: dict) -> dict:
	parsed_config = {}
	for scope_name, bindings in gin_config.items():
		# Ensure scope_name is a string
		if isinstance(scope_name, tuple):
			scope_name = ".".join(scope_name)

		module_name = scope_name.rsplit(".", 1)[-1]
		for binding_name, value in bindings.items():
			new_key = f"{module_name}/{binding_name}"
			parsed_config[new_key] = value
	return parsed_config
