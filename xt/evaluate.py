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
"""DESC: Xingtian evaluate entrance."""

import copy
import sys
import glob
import os
import time
from datetime import datetime
import re
import yaml
import numpy as np
from collections import OrderedDict
from absl import logging
from xt.framework.broker_launcher import launch_broker

from xt.framework.evaluate_adapter import TesterManager, EvalResultSummary
from xt.framework.learner import patch_alg_within_config
from zeus.common.util.get_xt_config import check_if_patch_local_node

from zeus.common.util.hw_cloud_helper import sync_data_from_s3
from zeus.common.util.hw_cloud_helper import XT_HWC_WORKSPACE


TEST_MODEL_GAP = 5


def setup_evaluate_adapter(config, controller, s3_result_path=None):
    """Start test."""
    if "test_node_config" in config:
        manager = TesterManager(config, controller, s3_result_path)
        manager.start()
    else:
        manager = None
    return manager


def main(config_file, s3_result_path=None):
    """DESC: the entrance for evaluate model."""
    with open(config_file) as f:
        config = yaml.safe_load(f)
        config = check_if_patch_local_node(config, "evaluate")

        eval_info = config.get("benchmark", dict()).get("eval", dict())
        model_path = eval_info.get("model_path")
        if not model_path:
            raise ValueError("must config 'benchmark.eval.model_path' to evaluate!")
        eval_gap = eval_info.get("gap", TEST_MODEL_GAP)

    # fixme: unify the path among s3 and local
    # read model from s3, sync total model into local /cache/XT_HWC_WORKSPACE
    if str(model_path).startswith("s3://"):
        sync_data_from_s3(model_path, XT_HWC_WORKSPACE)
        # re-point the model path to the local machine
        model_path = XT_HWC_WORKSPACE

    # check and distribute test model
    test_model = []
    if os.path.isfile(model_path):
        test_model.append(model_path)
    elif os.path.isdir(model_path):
        _total_model_list = list(glob.glob(os.path.join(model_path, "actor_*")))
        _total_model_list.sort(
            reverse=True,
            key=lambda x: int(str(os.path.splitext(
                os.path.basename(x))[0]).split("_")[-1]))
        test_model.extend(
            [
                _model
                for _index, _model in enumerate(_total_model_list)
                if _index % eval_gap == 0
            ]
        )
    else:
        logging.error("Path: {} is not exist".format(model_path))
        sys.exit(1)
    logging.info("will test ({}) models.".format(len(test_model)))

    # global controller
    config = patch_alg_within_config(config.copy(), node_type="test_node_config")
    controller = launch_broker(config)

    # fixme:hard connect into the model info
    # unset summary within actor
    config["model_para"]["actor"]["summary"] = False

    model_info = config["model_para"].copy()
    config["alg_para"].update({"model_info": model_info})

    tester_manager = setup_evaluate_adapter(config, controller, s3_result_path)
    if tester_manager is None:
        logging.fatal("test config is error")
        sys.exit(1)

    # start controller after tester's msg queue ready.
    controller.start()

    for i in test_model:
        logging.info(i)

    divided_times = eval_info.get("model_divided_freq", 1)
    target_ids = ["train_id", "eval_episode_reward", "eval_step_reward",
                  "custom_criteria", "battle_won"]

    save_csv = "./eval_result_{}.csv".format(datetime.now().strftime("%y%m%d%H%M%S"))
    result_summary = EvalResultSummary(
        divide_time=divided_times, target_ids=target_ids, output_csv=save_csv)
    logging.info("save output into '{}'. ".format(save_csv))

    _eval_summary_verbose = eval_info.get("summary", False)

    loop_is_end = False
    try:
        for model_name in test_model:
            np_file = np.load(model_name)
            ordered_weights = OrderedDict(**np_file)

            pattern = re.compile(r'(?<=actor_)\d+')
            train_index = int(pattern.findall(model_name)[0])

            for _divide_index in range(divided_times):

                eval_id = "{}@{}".format(train_index, _divide_index)
                tester_manager.put_test_model(
                    {eval_id: copy.deepcopy(ordered_weights)})

        while True:
            time.sleep(3)
            result_vars = tester_manager.fetch_eval_result()
            result_summary.append(result_vars)
            single_ret = result_summary.check_and_archive()

            if _eval_summary_verbose and single_ret:
                print(single_ret, end="\r")

            if result_summary.check_finish_stats(target_count=len(test_model)):
                result_summary.close()
                controller.stop()
                print("\n")
                break

        loop_is_end = True

    except (KeyboardInterrupt, EOFError) as err:
        logging.warning("Get a KeyboardInterrupt, Stop early.")
        # handle close signal, with cleaning works.

    if loop_is_end:
        logging.info("Finished evaluate job normally.")

    os._exit(0)
