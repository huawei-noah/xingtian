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
"""DESC: Xingtian train entrance."""

import os
import signal
import sys
import time
from subprocess import Popen
import pprint
import copy

from absl import logging
import yaml
import zmq

from xt.evaluate import setup_evaluate_adapter
from xt.framework.broker_launcher import launch_broker
from xt.framework.learner import setup_learner, patch_alg_within_config
from xt.framework.explorer import setup_explorer
from zeus.common.util.logger import StatsRecorder, VERBOSITY_MAP
from zeus.common.util.get_xt_config import parse_xt_multi_case_paras, \
    check_if_patch_local_node, get_pbt_set


TRAIN_PROCESS_LIST = list()


def _makeup_learner(config_info, data_url, verbosity):
    """Make up a learner instance and build the relation with broker."""
    config_info = patch_alg_within_config(config_info.copy(), node_type="node_config")

    _exp_params = pprint.pformat(config_info, indent=0, width=1,)
    logging.info("init learner with:\n{}\n".format(_exp_params))

    controller = launch_broker(config_info, verbosity=verbosity)
    eval_adapter = setup_evaluate_adapter(config_info, controller, verbosity)

    # fixme: split the relation between learner and tester
    _use_pbt, pbt_size, env_num, _pbt_config = get_pbt_set(config_info)

    if _use_pbt:
        metric_store = controller.register("pbt_metric", "store")
        weights_store = controller.register("pbt_weights", "store")
    else:
        metric_store, weights_store = None, None

    for _learner_id in range(pbt_size):
        learner = setup_learner(config_info, eval_adapter, _learner_id, data_url)

        controller.register("predict{}".format(learner.name), "send", learner.send_predict)
        learner.send_train = controller.register("train{}".format(learner.name), "send")
        learner.stats_deliver = controller.register("stats_msg{}".format(learner.name), "send")
        learner.send_broker = controller.register("recv{}".format(learner.name), "recv")
        controller.register("recv_predict{}".format(learner.name), "recv", learner.send_broker_predict)

        # update the learner <--relationship--> explorer ids
        eid_start = _learner_id*env_num
        learner.explorer_ids = list(range(eid_start, eid_start+env_num)) if _use_pbt else None

        # add this learner into population.
        if _use_pbt:
            learner.add_to_pbt(_pbt_config, metric_store, weights_store)

        setup_broker_stats(learner, controller)
        controller.add_task(learner)
        time.sleep(0.01)

    # start learner, after the data within broker stabilization.
    controller.start()
    time.sleep(0.01)

    for _learner in controller.tasks:
        _learner.start()

    for _index, _learner in enumerate(controller.tasks):
        config_of_learner = copy.deepcopy(config_info)
        config_of_learner.update({"learner_postfix": _learner.name})
        # unset summary within actor
        config_of_learner["model_para"]["actor"]["summary"] = False

        for env_index_per_pbt in range(env_num):
            # [0, 1, env_num-1] , [env_num, env_num+1, env_num*2-1], ...,
            # [env_num*(tasks_num-1), ..., env_num*tasks_num-1]
            env_id = _index * env_num + env_index_per_pbt
            setup_explorer(_learner.send_broker, config_of_learner, env_id)

        time.sleep(0.01)
    return controller


def setup_broker_stats(task_stub, to_broker):
    """Setup stats for each task."""
    stats_obj = StatsRecorder(
        msg_deliver=task_stub.stats_deliver,
        bm_args=task_stub.bm_args,
        workspace=task_stub.workspace,
        bm_board=task_stub.bm_board,
        name=task_stub.name
    )
    to_broker.stats.add_stats_recorder(task_stub.name, stats_obj)


def start_train(config_info, train_task,
                data_url=None, try_times=5, verbosity="info"):
    """Start training."""
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
    """Catch <ctrl+c> signal for clean stop."""
    global TRAIN_PROCESS_LIST
    for p in TRAIN_PROCESS_LIST:
        p.send_signal(signal.SIGINT)

    time.sleep(1)
    os._exit(0)


def train(config_info, train_task, s3_path, verbosity="info"):
    if verbosity in VERBOSITY_MAP.keys():
        logging.set_verbosity(VERBOSITY_MAP[verbosity])
        pass
    else:
        logging.warning("un-known logging level-{}".format(verbosity))

    controller = start_train(config_info, train_task,
                             data_url=s3_path, verbosity=verbosity)
    loop_is_end = False
    try:
        controller.tasks_loop()
        loop_is_end = True
    except (KeyboardInterrupt, EOFError) as ex:
        logging.warning("Get a KeyboardInterrupt, Stop early.")
    except BaseException as ex:
        logging.exception(ex)
        logging.warning("Get a Exception, Stop early.")

    # handle close signal, with cleaning works.
    for _task in controller.tasks:
        _task.train_worker.logger.save_to_json()
    controller.stop()

    # fixme: make close harmonious between controller & broker
    time.sleep(2)
    if loop_is_end:
        logging.info("Finished train job normally.")

    os._exit(0)


def main(config_file, train_task, s3_path=None, verbosity="info"):
    """Do train task with single case."""
    with open(config_file) as f:
        config_info = yaml.safe_load(f)

    train(config_info, train_task, s3_path, verbosity)


# train with multi case
def write_conf_file(config_folder, config):
    """Write config to file."""
    with open(config_folder, "w") as f:
        yaml.dump(config, f)


def makeup_multi_case(config_file, s3_path):
    """Run multi cases."""
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
    """Run train process."""
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
