import logging


class CustomFormatter(logging.Formatter):
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
