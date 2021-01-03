# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
# THE SOFTWARE.
"""Utils for printing."""

import sys
from time import time
import pprint
from absl import logging
LAST_PRINT = time()


def print_immediately(to_str):
    """Print some string immediately."""
    print(to_str)
    sys.stdout.flush()


def debug_within_interval(logs=None, interval=10, func=None, human_able=False, **kwargs):
    """Print with time interval."""
    global LAST_PRINT
    if time() - LAST_PRINT > interval:
        # print(func, **kwargs)
        if func and callable(func):
            func(**kwargs)
        if logs:
            logs_human = pprint.pformat(logs, indent=0, width=1) if human_able else logs
            logging.debug("{}".format(logs_human))

        LAST_PRINT = time()
        return True

    return False
