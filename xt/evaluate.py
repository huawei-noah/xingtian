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
"""
xingtian evaluate entrance.
"""
import sys
import glob
import os
import time

import yaml
from absl import logging
from xt.framework.broker_launcher import launch_broker

from xt.framework.evaluate_adapter import TesterManager
from xt.framework.learner import patch_alg_within_config
from xt.util.common import get_config_file
from xt.benchmark.tools.get_config import check_if_patch_local_node

from xt.util.hw_cloud_helper import sync_data_from_s3
from xt.util.hw_cloud_helper import XT_HWC_WORKSPACE


TEST_MODEL_GAP = 5


def setup_evaluate_adapter(config, broker_master, s3_result_path=None):
    """ start test """
    if "test_node_config" in config:
        manager = TesterManager(config, broker_master, s3_result_path)
        manager.start()
    else:
        manager = None
    return manager


def main(config_file, s3_result_path=None):
    """The entrance for evaluate model. """
    with open(config_file) as f:
        config = yaml.safe_load(f)
        config = check_if_patch_local_node(config, "evaluate")

        model_path = config["test_model_path"]
        eval_gap = (
            config.get("benchmark", dict())
            .get("eval", dict())
            .get("gap", TEST_MODEL_GAP)
        )
    # fixme: unify the path among s3 and local
    # read model from s3, sync total model into local /cache/XT_HWC_WORKSPACE
    if str(model_path).startswith("s3://"):
        sync_data_from_s3(model_path, XT_HWC_WORKSPACE)
        # re-point the model path to the local machine
        model_path = XT_HWC_WORKSPACE

    # global broker_master
    config = patch_alg_within_config(config.copy())
    broker_master = launch_broker(config)

    # fixme:hard connect into the model info
    model_info = config["model_para"].copy()
    config["alg_para"].update({"model_info": model_info})

    tester_manager = setup_evaluate_adapter(config, broker_master, s3_result_path)
    if tester_manager is None:
        logging.fatal("test config is error")
        sys.exit(1)

    # distribute test model
    test_model = []
    if os.path.isfile(model_path):
        test_model.append(model_path)
    elif os.path.isdir(model_path):
        _total_model_list = list(glob.glob(os.path.join(model_path, "*.h5")))
        _total_model_list.sort(reverse=True)
        test_model.extend(
            [
                _model
                for _index, _model in enumerate(_total_model_list)
                if _index % eval_gap == 0
            ]
        )
    else:
        logging.warning("model_path: {} is not exist".format(model_path))
        sys.exit(1)
    logging.info("will test ({}) models.".format(len(test_model)))
    for i in test_model:
        logging.info(i)

    loop_is_end = False
    try:
        for model_name in test_model:
            tester_manager.put_test_model([model_name])

        while True:
            time.sleep(3)
            if tester_manager.check_finish_stat(target_model_count=len(test_model)):
                broker_master.stop()
                break
        loop_is_end = True

    except (KeyboardInterrupt, EOFError) as err:
        logging.warning("Get a KeyboardInterrupt, Stop early.")
        # handle close signal, with cleaning works.

    if loop_is_end:
        logging.info("Finished evaluate job normally.")

    os._exit(0)


if __name__ == "__main__":
    main(get_config_file())
