import logging
import os
import sys
from datetime import datetime

def setup_logger(log_dir: str, name: str = "DeepEnergy", log_level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回一个标准的 Logger 对象。
    
    科研用途:
    1. 同时输出到 控制台(Console) 和 文件(File)。
    2. 文件名带时间戳，防止覆盖之前的实验记录。
    3. 解决 Jupyter/多次运行时日志重复打印的问题。
    
    Args:
        log_dir (str): 日志保存的文件夹路径 (例如 ./logs/)
        name (str): Logger 的名称 (通常对应实验名或模块名)
        log_level (int): 日志级别，默认为 INFO
        
    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    # 1. 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # 2. 获取 Logger 实例
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # [关键步骤] 防止重复打印
    # Python 的 logger 是全局单例的。如果再次调用 setup_logger (比如在不同 py 文件中引用)，
    # 之前的 handler 还在，会导致一条日志打两遍。这里必须先清空。
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 3. 定义日志格式
    # 格式: [时间] [模块名] [日志级别] 消息内容
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 4. Handler 1: 输出到文件 (FileHandler)
    # 文件名格式: name_20260115_203000.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    
    # 5. Handler 2: 输出到控制台 (StreamHandler)
    # 输出到 stdout，方便在终端实时查看进度
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # 可选：打印一条初始化信息（只在文件里留底，不在控制台刷屏）
    # file_handler.stream.write(f"Log initialized at {log_filepath}\n")
    
    return logger