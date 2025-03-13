import logging
import os
import inspect
import traceback
from datetime import datetime

class ContextFilter(logging.Filter):
    def filter(self, record):
        # 获取调用栈信息
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == '_log' and frame.f_back:
                frame = frame.f_back
                break
            frame = frame.f_back
        
        if frame:
            record.filename = os.path.basename(frame.f_code.co_filename)
            record.lineno = frame.f_lineno
        else:
            record.filename = 'unknown'
            record.lineno = 0
        return True

class CustomFormatter(logging.Formatter):
    def format(self, record):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return (
            f"[{current_time}] {record.filename}:{record.lineno} - "
            f"{record.getMessage()}"
        )

# 配置全局logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 添加上下文过滤器
context_filter = ContextFilter()
logger.addFilter(context_filter)

# 配置控制台handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

def log_exception(e: Exception, message: str = None):
    """ 自定义异常记录快捷方式 """
    logger.error(
        f"{message}: {str(e)}" if message else f"Exception: {str(e)}",
        exc_info=True
    )
