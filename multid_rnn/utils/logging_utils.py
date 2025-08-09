import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_multiD_RNN_LOGGER_NAME = "multiD_RNN"


__logger_ininitialized = False


def get_logger() -> logging.Logger:
    """Get the multiD_RNN logger.

    Returns:
        logging.Logger: The multiD_RNN logger.
    """
    logger = logging.getLogger(name=_multiD_RNN_LOGGER_NAME)
    assert logger is not None, "Failed to get the logger."

    # Initialize logger only once
    global __logger_ininitialized
    if not __logger_ininitialized:
        init_logger(logger)
        __logger_ininitialized = True

    return logger

def get_debug_logger() -> logging.Logger:
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    [h.setLevel(logging.DEBUG) for h in logger.handlers]
    return logger


def init_logger(logger: logging.Logger) -> None:
    """Initialize the logger.

    Ininitialize the logger with the following settings:

    - Log format: (ISO-8601 DataTime)\t(LogLevel)\t(Module).(Function):(FileName):(LineNo)\t(Message)
    - Log stream: stdout
    - Log level
        - Logger: INFO
        - Handler: INFO

    Note:
        This function should be called only once for each logger.
        Must not be called inside the multiD_RNN package.

    Args:
        logger: The logger to initialize

    Example:
        >>> import logging
        >>> from multid_rnn.logging import get_logger, init_logger
        >>> logger = get_logger()
        >>> init_logger(logger)
        >>> logger.info("Hello, multiD_RNN!")
    """
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = create_tsv_formatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def init_execution_trace_logging(
    logger: logging.Logger, execution_log_path: Path
) -> None:
    """Initialize the logger for execution trace logging.

    Ininitialize the logger with the following settings:

    - Log format: JSON lines
        - "timestamp": ISO-8601 DataTime
        - "level": LogLevel
        - "module": Module
        - "function": Function
        - "message": Message
    - Log stream: file (execution_log_path)
    - Log level
        - Logger: DEBUG
        - Handler: DEBUG

    Note:
        This function should be called only once for each logger.
        Must not be called inside the multiD_RNN package.

    Args:
        logger: The logger to initialize

    Example:
        >>> import logging
        >>> from multid_rnn.logging import get_logger, init_execution_trace_logging
        >>> logger = get_logger()
        >>> execution_log_path = Path("execution.log")
        >>> init_execution_trace_logging(logger, execution_log_path)
        >>> logger.debug("Hello, multiD_RNN!", extra={"answer": 42})
    """
    logger.setLevel(logging.DEBUG)
    if not execution_log_path.exists():
        execution_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(execution_log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = create_json_lines_formatter()
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def create_tsv_formatter() -> logging.Formatter:
    """Create a TSV formatter for logging.

    - Log format: (ISO-8601 DataTime)\t[(LogLevel)]\t(Module).(Function):(FileName):(LineNo)\t(Message)

    Returns:
        logging.Formatter: The TSV formatter.
    """
    formatter = TabToSpacesFormatter(
        fmt="%(asctime)s\t[%(levelname)s]\t%(module)s.%(funcName)s:%(filename)s:%(lineno)d\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    return formatter


class TabToSpacesFormatter(logging.Formatter):
    def formatMessage(self, record):
        # replace tab with 4 spaces
        record.message = record.getMessage().replace("\t", " " * 4)
        return super().formatMessage(record)


def create_json_lines_formatter() -> logging.Formatter:
    """Create a JSON formatter for logging.

    - Log format
        - "timestamp": ISO-8601 DataTime
        - "level": LogLevel
        - "module": Module
        - "function": Function
        - "filename": FileName
        - "lineno": LineNo
        - "message": Message

    Returns:
        logging.Formatter: The JSON formatter.
    """
    formatter = JSONLinesFormatter()
    return formatter


class JSONLinesFormatter(logging.Formatter):
    STANDARD_KEYS = {
        "timestamp",
        "level",
        "module",
        "function",
        "filename",
        "lineno",
        "message",
        "asctime",
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "exc_info",
        "exc_text",
        "stack_info",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record):
        """format log record to json lines

        Args:
            self: log formatter
            record: log records

        Returns:
            log text (json lines format)
        """
        log_record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "filename": record.filename,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }

        # update extra key-values
        extra_record = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.STANDARD_KEYS
        }
        if extra_record:
            log_record["extra"] = extra_record

        return json.dumps(log_record, ensure_ascii=False, default=str)
