import logging
import os
import random
import sys
import numpy as np
import torch
from pathlib import Path
import traceback


def set_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(save_dir, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):

    if os.path.exists(save_dir):
        raise FileExistsError(f"{save_dir} already exists!")
    os.makedirs(save_dir, exist_ok=True)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    if info_filename is not None:
        info_file_handler = logging.FileHandler(os.path.join(save_dir, info_filename))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(os.path.join(save_dir, debug_filename))
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = exception_handler


def init_path(work_dir):
    Path(os.path.join(work_dir, 'val')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(work_dir, 'logs')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(work_dir, 'test')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(work_dir, 'param')).mkdir(parents=True, exist_ok=True)