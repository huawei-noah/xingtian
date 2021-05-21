# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

"""
| 字段/属性名称   | 使用格式            | 描述                                                         |
| --------------- | ------------------- | ------------------------------------------------------------ |
| asctime         | %(asctime)s         | 日志事件发生的时间--人类可读时间，如：2003-07-08 16:49:45,896 |
| created         | %(created)f         | 日志事件发生的时间--时间戳，就是当时调用time.time()函数返回的值 |
| relativeCreated | %(relativeCreated)d | 日志事件发生的时间相对于logging模块加载时间的相对毫秒数（目前还不知道干嘛用的） |
| msecs           | %(msecs)d           | 日志事件发生事件的毫秒部分                                   |
| levelname       | %(levelname)s       | 该日志记录的文字形式的日志级别（'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'） |
| levelno         | %(levelno)s         | 该日志记录的数字形式的日志级别（10, 20, 30, 40, 50）         |
| name            | %(name)s            | 所使用的日志器名称，默认是'root'，因为默认使用的是 rootLogger |
| message         | %(message)s         | 日志记录的文本内容，通过 `msg % args`计算得到的              |
| pathname        | %(pathname)s        | 调用日志记录函数的源码文件的全路径                           |
| filename        | %(filename)s        | pathname的文件名部分，包含文件后缀                           |
| module          | %(module)s          | filename的名称部分，不包含后缀                               |
| lineno          | %(lineno)d          | 调用日志记录函数的源代码所在的行号                           |
| funcName        | %(funcName)s        | 调用日志记录函数的函数名                                     |
| process         | %(process)d         | 进程ID                                                       |
| processName     | %(processName)s     | 进程名称，Python 3.1新增                                     |
| thread          | %(thread)d          | 线程ID                                                       |
| threadName      | %(thread)s          | 线程名称                                                     |
"""

import logging
import sys


class LoggingEngine:
    def __init__(self, level="debug", contents=None, logger_name=None):
        self.logging_level_dict = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }

        logging_level = self.logging_level_dict.get(level.lower(), logging.DEBUG)

        if contents is None:
            contents = ["asctime", "levelname", "funcName", "lineno", "message"]

        if logger_name is None:
            logger_name = 'logging_engine'

        logging_fmt = "%(asctime)s [%(filename)-15s | %(lineno)d] %(levelname)s: %(message)s"
        # logging_fmt = " - ".join([f"%({content})s" for content in contents])

        logger = logging.getLogger(logger_name)
        logger.setLevel(level=logging_level)
        formatter = logging.Formatter(logging_fmt)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        self.logger = logger
        self.logger_name = logger_name
        self.handlers = {}
        self.formatter = formatter

        self.import_log_funcs()

    def import_log_funcs(self):
        log_funcs = ['debug', 'info', 'warning', 'error', 'critical', 'exception']
        for func_name in log_funcs:
            func = getattr(self.logger, func_name)
            setattr(self, func_name, func)

    def add_file_output(self, filename: str, level='info', mode="w"):
        if filename not in self.handlers:
            handler = logging.FileHandler(filename, mode=mode, encoding='UTF-8')
            handler.setFormatter(self.formatter)
            handler.setLevel(self.logging_level_dict.get(level.lower(), logging.DEBUG))
            self.handlers[filename] = handler
            self.logger.addHandler(handler)

    def remove_file_handler(self, file_path):
        if file_path in self.handlers:
            self.logger.removeHandler(self.handlers.get(file_path))

    def debug(self, msg: str):
        pass

    def info(self, msg: str):
        pass

    def warning(self, msg: str):
        pass

    def error(self, msg: str):
        pass

    def critical(self, msg: str):
        pass

    def exception(self, msg: str):
        pass


logger = LoggingEngine(logger_name="glob_logging_engine",
                       level="info")


def test_log():
    log = LoggingEngine(level="debug",
                        contents=["asctime", "levelname", "filename", "lineno", "funcName", "message"])

    log.info("Hello World!")
