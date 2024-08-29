import os, sys
import traceback

import os
import logging
from datetime import datetime

def get_new_logger(log_dir=None, log_name=None):
    if log_dir is None:
        log_dir = './log_dir'
    if log_name is None:
        log_name = 'log_{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

    if not os.path.exists(log_dir):
        print("[LOG_INFO_Dynamic] Buid new log dir:%s"%log_dir)
        os.makedirs(log_dir)
    log_file_path = '%s/%s'%(log_dir, log_name)

    print("[LOG_INFO_Dynamic] log_file_path:%s"%log_file_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter( '[LOG_INFO_Dynamic] %(asctime)s %(filename)s %(lineno)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

if __name__ == "__main__":
    abs_path = os.path.abspath(__file__)

    # log
    log_dir = './log_dir'
    log_name = 'log_{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = get_new_logger(log_dir, log_name)
    logger.info("[Dynamic] == log_dir={} ===".format(log_dir))
    logger.info("abs_path={}".format(abs_path))

