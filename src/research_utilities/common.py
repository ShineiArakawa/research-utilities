import logging
import os
import pathlib
import sys
import typing


class GlobalSettings:
    """Global settings for the project.
    This class contains global constants and settings that can be used throughout the project.
    It is recommended to use this class to store global settings instead of using environment variables
    """

    # autopep8: off
    DEBUG_MODE       : typing.Final[bool]                        = False
    GLOBAL_LOG_LEVEL : typing.Final[str]                         = "trace"
    CACHE_DIR        : typing.Final[pathlib.Path]                = pathlib.Path("~/.cache/research_utilities").expanduser().resolve()
    # autopep8: on


def get_logger(
    log_level: str | None = None,
    name: str | None = None,
    logger_type: typing.Literal["native", "loguru"] = "loguru"
) -> logging.Logger | typing.Any:
    """Get logger object. If not found `loguru`, `logger_type` will be set to `native` automatically.
    The log level is set by `log_level` or `GLOBAL_LOG_LEVEL` environment variable.

    Parameters
    ----------
    log_level : str, optional
        Log level, by default None
    name : str, optional
        Logger name. If you use loguru, this option will be ignored, by default None
    logger_type : str, optional
        Logger type. You can select from ['native', 'loguru'], by default "loguru"

    Returns
    -------
    logging.Logger | loguru.Logger
        logger object
    """

    log_level = log_level if log_level else os.environ.get("GLOBAL_LOG_LEVEL", GlobalSettings.GLOBAL_LOG_LEVEL)
    logger = None

    if logger_type == "loguru":
        try:
            import loguru
            from loguru import logger as _logger
        except ModuleNotFoundError:
            logger_type = "native"
            pass
        pass

    if logger_type == "native":
        logging.basicConfig(
            level=log_level.upper(),
            format="logging:@:%(filename)s(%(lineno)s):fn:%(funcName)s:\nlevel:%(levelname)s:%(message)s"
        )
        logger = logging.getLogger(name)
    elif logger_type == "loguru":
        _logger: loguru.Logger
        logger = _logger
        logger.remove()
        logger.add(sys.stdout, level=log_level.upper())
    else:
        raise RuntimeError(f"Unknown logger type: {logger_type}")

    return logger
