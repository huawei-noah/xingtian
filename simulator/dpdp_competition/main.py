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

import traceback
import datetime
import numpy as np
import sys

from src.conf.configs import Configs
from src.simulator.simulate_api import simulate
from src.utils.log_utils import ini_logger, remove_file_handler_of_logging
from src.utils.logging_engine import logger
# from naie.metrics import report

if __name__ == "__main__":
    # if you want to traverse all instances, set the selected_instances to []
    selected_instances = Configs.selected_instances

    if selected_instances:
        test_instances = selected_instances
    else:
        test_instances = Configs.all_test_instances

    score_list = []
    for idx in test_instances:
        # Initial the log
        log_file_name = f"dpdp_{datetime.datetime.now().strftime('%y%m%d%H%M%S')}.log"
        ini_logger(log_file_name)

        instance = "instance_%d" % idx
        logger.info(f"Start to run {instance}")

        try:
            score = simulate(Configs.factory_info_file, Configs.route_info_file, instance)
            score_list.append(score)
            logger.info(f"Score of {instance}: {score}")
        except Exception as e:
            logger.error("Failed to run simulator")
            logger.error(f"Error: {e}, {traceback.format_exc()}")
            score_list.append(sys.maxsize)

        # 删除日志句柄
        remove_file_handler_of_logging(log_file_name)

    avg_score = np.mean(score_list)
    # with report(True) as logs:
    #     logs.log_metrics('score', [avg_score])
    print(score_list)
    print(avg_score)
    print("Happy Ending")
