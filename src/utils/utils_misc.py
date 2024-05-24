import logging


class CustomFormatter(logging.Formatter):
	"""Custom Formatter for logging.
	This formatter is only used to colorize the log messages.
	It does not define the log message itself.
	"""

	grey = "\x1b[38;21m"
	blue = "\x1b[38;5;39m"
	yellow = "\x1b[38;5;226m"
	red = "\x1b[38;5;196m"
	bold_red = "\x1b[31;1m"
	reset = "\x1b[0m"

	def __init__(self, fmt):
		super().__init__()
		self.fmt = fmt
		self.FORMATS = {
			logging.DEBUG: self.grey + self.fmt + self.reset,
			logging.INFO: self.blue + self.fmt + self.reset,
			logging.WARNING: self.yellow + self.fmt + self.reset,
			logging.ERROR: self.red + self.fmt + self.reset,
			logging.CRITICAL: self.bold_red + self.fmt + self.reset,
		}

	def format(self, record):
		log_fmt = self.FORMATS.get(record.levelno)
		formatter = logging.Formatter(log_fmt)
		return formatter.format(record)


def set_loggers(path_log=None, logging_level=0, b_stream=True, b_debug=False):
	"""Define logging settings."""

	# std. logger
	logger = logging.getLogger()
	logger.setLevel(logging_level)

	# Formatter
	format_msg = "==== %(asctime)s | %(levelname)9s | %(filename)s:%(lineno)d | >> %(message)8s"
	formatter_stream = CustomFormatter(format_msg)
	formatter_file = logging.Formatter(format_msg)
	if path_log:
		file_handler = logging.FileHandler(path_log)
		file_handler.setFormatter(formatter_file)
		logger.addHandler(file_handler)

	# plot to console
	if b_stream:
		# remove existing handlers
		if logger.hasHandlers():
			for handler in logger.handlers:
				logger.removeHandler(handler)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter_stream)
		logger.addHandler(stream_handler)
