import logging
import os
import inspect
import traceback
from datetime import datetime
from io import StringIO

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
    
    def formatException(self, ei):
        """
        Format and return the specified exception information as a string.
        Includes filename and line number for each frame in the stack trace.
        """
        sio = StringIO()
        tb = ei[2]
        while tb:
            frame = tb.tb_frame
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            funcname = frame.f_code.co_name
            sio.write(f"  File \"{filename}\", line {lineno}, in {funcname}\n")
            tb = tb.tb_next
            
        # 添加异常类型和消息
        sio.write(f"{ei[0].__name__}: {ei[1]}\n")
        
        stack_trace = sio.getvalue()
        sio.close()
        if stack_trace[-1:] == "\n":
            stack_trace = stack_trace[:-1]
        return stack_trace

if __name__ == "__main__":
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

    def exception(e: Exception, message: str = None):
        """ 自定义异常记录快捷方式 """
        # 获取调用者的框架信息
        caller_frame = inspect.currentframe().f_back
        filename = os.path.basename(caller_frame.f_code.co_filename)
        line_no = caller_frame.f_lineno
        
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 获取完整的异常堆栈
        sio = StringIO()
        traceback.print_exception(type(e), e, e.__traceback__, None, sio)
        stack_trace = sio.getvalue()
        sio.close()
        if stack_trace[-1:] == "\n":
            stack_trace = stack_trace[:-1]
        
        # 构建日志消息
        log_message = f"[{current_time}] {filename}:{line_no} - "
        if message:
            log_message += f"{message}: "
        log_message += f"\nException: {type(e).__name__}: {str(e)}\nStack trace:\n{stack_trace}"
        
        logger.error(log_message, exc_info=True)