#!/usr/bin/env python3
import os
import sys
import logging
from pythonjsonlogger import jsonlogger


SUCCESS_LEVEL_NUM = int(os.getenv("LOG_SUCCESS_LEVEL_NUM", "25"))
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def success(self, message=None, **kwargs):
    """
    Custom log level for SUCCESS events. If everything goes well this should be the last messaged logged from the job.
    """
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, (), **kwargs)


logging.Logger.success = success


class JobIDFilter(logging.Filter):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def filter(self, record):
        record.job_id = self.job_id
        return True


def setup_logger(job_id: str) -> logging.Logger:
    """
    Initialize a JSON-format logger that conforms to the log conventions specified in job_conventions.md.
    
    Args:
        job_id: The job identifier (e.g., "fim_mosaicker", "hand_inundator", etc.)
    
    Returns:
        Configured logger instance
    """
    log = logging.getLogger(job_id)
    if log.handlers:
        return log

    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(JobIDFilter(job_id))

    fmt = "%(asctime)s %(levelname)s %(job_id)s %(message)s"
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt=fmt,
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                # job_id stays as-is
                # message stays as-is
            },
            json_ensure_ascii=False,
        )
    )

    log.addHandler(handler)
    log.propagate = False
    return log