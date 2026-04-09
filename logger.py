import logging
import sys
from logging.handlers import RotatingFileHandler

LOG_FILE = "pipeline.log"

def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # # 写入文件，最大 10MB，保留 5 个备份
    # fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    # fh.setFormatter(fmt)
    # logger.addHandler(fh)

    # # 同时输出到终端
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setFormatter(fmt)
    # logger.addHandler(sh)

    return logger


logger = get_logger()
