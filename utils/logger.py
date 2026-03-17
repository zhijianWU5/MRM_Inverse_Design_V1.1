import logging
import os

def setup_logger(log_dir='data/logs'):
    """配置全局日志系统，同时输出到控制台和文件"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger('MRM_Optimizer')
    logger.setLevel(logging.INFO)

    # 【核心修复 1】阻止日志向上层(Root Logger)传递，防止被其他第三方库重复打印
    logger.propagate = False 

    # 【核心修复 2】暴力清空当前可能残留的旧 Handlers，确保不管环境怎么重载都只有一对
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. 输出到文件的配置
    file_handler = logging.FileHandler(os.path.join(log_dir, 'optimization.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 2. 输出到控制台的配置
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('\033[1;32m[%(levelname)s]\033[0m %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger