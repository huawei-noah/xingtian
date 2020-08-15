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
# THE SOFTWARE
"""
xingtian train entrance.
"""
import os
import signal
import sys
import time
from subprocess import Popen
import pprint

from absl import logging
import yaml
import zmq

from xt.evaluate import setup_evaluate_adapter
from xt.framework.broker_launcher import launch_broker
from xt.framework.learner import setup_learner, patch_alg_within_config
from xt.framework.explorer import setup_explorer
from xt.util.common import get_config_file
from xt.benchmark.tools.get_config import parse_xt_multi_case_paras, \
    check_if_patch_local_node

TRAIN_PROCESS_LIST = list()


def _makeup_learner(config_info, data_url, verbosity):
    """make up a learner instance, and build the relation with broker"""

    config_info = patch_alg_within_config(config_info.copy())

    _exp_params = pprint.pformat(config_info, indent=0, width=1,)
    logging.info("init learner with:\n{}\n".format(_exp_params))

    broker_master = launch_broker(config_info)
    eval_adapter = setup_evaluate_adapter(config_info, broker_master, verbosity)

    # fixme: split the relation between learner and tester
    learner = setup_learner(config_info, eval_adapter, data_url)

    learner.send_predict = broker_master.register("predict")
    learner.send_train = broker_master.register("train")
    learner.stats_deliver = broker_master.register("stats_msg")
    learner.send_broker = broker_master.recv_local_q
    learner.start()

    broker_master.main_task = learner

    env_num = config_info.get("env_num")
    for i in range(env_num):
        setup_explorer(broker_master, config_info, i)
    return broker_master


def start_train(config_file, train_task,
                data_url=None, try_times=5, verbosity="info"):
    """ start train"""

    with open(config_file) as f:
        config_info = yaml.safe_load(f)

    config_info = check_if_patch_local_node(config_info, train_task)

    for _ in range(try_times):
        try:
            return _makeup_learner(config_info, data_url, verbosity)

        except zmq.error.ZMQError as err:
            logging.error("catch: {}, \n try with times-{}".format(err, _))
            continue
        except BaseException as ex:
            logging.exception(ex)
            os.system("pkill -9 fab")
            sys.exit(3)


def handle_multi_case(sig, frame):
    """ Catch <ctrl+c> signal for clean stop """

    global TRAIN_PROCESS_LIST
    for p in TRAIN_PROCESS_LIST:
        p.send_signal(signal.SIGINT)

    time.sleep(1)
    os._exit(0)


def main(config_file, train_task, s3_path=None, verbosity="info"):
    """do train task with single case """
    broker_master = start_train(config_file, train_task,
                                data_url=s3_path, verbosity=verbosity)
    loop_is_end = False
    try:
        broker_master.main_loop()
        loop_is_end = True
    except (KeyboardInterrupt, EOFError) as ex:
        logging.warning("Get a KeyboardInterrupt, Stop early.")
    except BaseException as ex:
        logging.exception(ex)
        logging.warning("Get a Exception, Stop early.")

    # handle close signal, with cleaning works.
    broker_master.main_task.train_worker.logger.save_to_json()
    broker_master.stop()

    # fixme: make close harmonious between broker master & slave
    time.sleep(2)
    if loop_is_end:
        logging.info("Finished train job normally.")

    os._exit(0)


# train with multi case
def write_conf_file(config_folder, config):
    """ write config to file """
    with open(config_folder, "w") as f:
        yaml.dump(config, f)


def makeup_multi_case(config_file, s3_path):
    """ run multi case """
    signal.signal(signal.SIGINT, handle_multi_case)
    # fixme: setup with archive path
    if os.path.isdir("log") is False:
        os.makedirs("log")
    if os.path.isdir("tmp_config") is False:
        os.makedirs("tmp_config")

    ret_para = parse_xt_multi_case_paras(config_file)
    config_file_base_name = os.path.split(config_file)[-1]

    for i, para in enumerate(ret_para):
        if i > 9:
            logging.fatal("only support 10 parallel case")
            break

        tmp_config_file = "{}_{}".format(config_file_base_name, i)
        config_file = os.path.join("tmp_config", tmp_config_file)
        write_conf_file(config_file, para)

        abs_config_file = os.path.abspath(config_file)

        log_file = os.path.join("log", "log_{}.log".format(tmp_config_file))

        TRAIN_PROCESS_LIST.append(
            launch_train_with_shell(
                abs_config_file, s3_path=s3_path, stdout2file=log_file
            )
        )

    while True:
        time.sleep(100)


def launch_train_with_shell(abs_config_file, s3_path=None, stdout2file="./xt.log"):
    """ run train process """
    cmd = "import xt; from xt.train import main; main('{}', {})".format(
        abs_config_file, s3_path
    )
    logging.info("start launching train with cmd: \n{}".format(cmd))
    file_out = open(stdout2file, "w")

    process_instance = Popen(
        ["python3", "-c", cmd],
        stdout=file_out,
    )
    time.sleep(1)
    return process_instance


if __name__ == "__main__":
    main(get_config_file())
