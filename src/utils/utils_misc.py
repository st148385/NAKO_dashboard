import logging


class CustomFormatter(logging.Formatter):
	"""Logging formatter with colorized output for different log levels.

	This formatter applies ANSI escape codes to colorize the log output based on the
	severity level of the message. It uses a clear and concise format to display
	the timestamp, log level, source file, line number, and the log message itself.

	**Attributes:**

	- grey: ANSI escape code for grey text.
	- blue: ANSI escape code for blue text.
	- yellow: ANSI escape code for yellow text.
	- red: ANSI escape code for red text.
	- bold_red: ANSI escape code for bold red text.
	- reset: ANSI escape code to reset text formatting.
	- FORMAT: The base format string for log messages.
	- FORMATS: A dictionary mapping logging levels to their corresponding
	  colorized format strings.

	**Methods:**

	- format(record): Formats the log record according to the specified
	  format and color for its log level.
	"""

	grey = "\x1b[38;21m"
	blue = "\x1b[38;5;39m"
	yellow = "\x1b[38;5;226m"
	red = "\x1b[38;5;196m"
	bold_red = "\x1b[31;1m"
	reset = "\x1b[0m"

	FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d  >>>  %(message)s"

	FORMATS = {
		logging.DEBUG: f"{grey}{FORMAT}{reset}",
		logging.INFO: f"{blue}{FORMAT}{reset}",
		logging.WARNING: f"{yellow}{FORMAT}{reset}",
		logging.ERROR: f"{red}{FORMAT}{reset}",
		logging.CRITICAL: f"{bold_red}{FORMAT}{reset}",
	}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)


def set_loggers(path_log=None, logging_level=logging.INFO, b_stream=True, b_debug=False):
	"""Configures loggers for file and/or console output.

	This function sets up logging to a file (if a path is provided) and/or the
	console. It allows you to customize the logging level and enable/disable debug mode.

	:param path_log: The path to the log file. If None, logging to a file is disabled.
	:type path_log: str, optional
	:param logging_level: The minimum logging level to capture. Defaults to logging.INFO.
	:type logging_level: int, optional
	:param b_stream: Whether to log to the console. Defaults to True.
	:type b_stream: bool, optional
	:param b_debug: Whether to enable debug mode, setting the logging level to DEBUG. Defaults to False.
	:type b_debug: bool, optional
	"""
	logger = logging.getLogger()
	logger.setLevel(logging_level)

	# Remove existing handlers to avoid duplicate logs (but only if streaming)
	if b_stream and logger.hasHandlers():
		logger.handlers.clear()

	# File Handler (if path_log is provided)
	if path_log:
		file_handler = logging.FileHandler(path_log)
		file_handler.setFormatter(logging.Formatter(CustomFormatter.FORMAT))  # Use uncolored format for file
		logger.addHandler(file_handler)

	# Stream Handler (if enabled)
	if b_stream:
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(CustomFormatter(CustomFormatter.FORMAT))
		logger.addHandler(stream_handler)

	# Debug mode (if enabled)
	if b_debug:
		logger.setLevel(logging.DEBUG)  # Set to DEBUG level
