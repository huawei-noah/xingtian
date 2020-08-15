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
Learner module cover the training process within the RL problems.
"""
import os
import threading
from time import time
from copy import deepcopy
import numpy as np
from absl import logging
from collections import deque

from xt.util.logger import Logger, StatsRecorder
from xt.util.profile_stats import PredictStats
from xt.framework.trainer import build_alg_with_trainer
from xt.benchmark.tools.evaluate_xt import (
    make_workspace_if_not_exist,
    parse_benchmark_args,
)

from xt.benchmark.visualize import BenchmarkBoard
from xt.framework.comm.message import message, get_msg_data, set_msg_info, set_msg_data, get_msg_info
from xt.util.common import bytes_to_str
from xt.util.hw_cloud_helper import mox_makedir_if_not_existed, sync_data_to_s3


class Learner(object):
    def __init__(
            self,
            alg_para,
            env_para,
            agent_para,
            test_master=None,
            data_url=None,
            benchmark_info=None,
    ):
        self.alg_para = deepcopy(alg_para)
        self.process_num = self.alg_para.get("process_num", 1)

        self.test_master = test_master

        self.train_worker = None

        self.send_train = None
        self.send_predict = None
        self.send_broker = None
        self.stats_deliver = None

        self.train_lock = threading.Lock()
        self.alg = None
        self.trainer = None
        self.shared_buff = None

        self.bm_args = parse_benchmark_args(
            env_para, alg_para, agent_para, benchmark_info
        )
        _model_dir = ["models", "benchmark"]
        self._workspace, _archive, _job = make_workspace_if_not_exist(
            self.bm_args, _model_dir
        )
        self.bm_board = BenchmarkBoard(_archive, _job)

        self.model_path = os.path.join(self._workspace, _model_dir[0])
        logging.info(
            "{} \nworkspace: \n\t{} \n"
            "model will save under path: \n\t{} \n"
            "".format("*" * 10, self._workspace, self.model_path)
        )

        self.max_step = agent_para.get("agent_config", {}).get("complete_step")

        # For Cloud
        self.s3_path = None
        if data_url is not None:
            self.s3_path = os.path.join(data_url, _model_dir[0])
            mox_makedir_if_not_existed(self.s3_path)

    def async_predict(self):
        """ create predict thread """
        predict = [
            PredictThread(
                i,
                self.alg,
                self.send_predict,
                self.send_broker,
                self.stats_deliver,
                self.train_lock,
            )
            for i in range(2)
        ]
        predict_thread = [threading.Thread(target=t.predict) for t in predict]

        for t in predict_thread:
            t.setDaemon(True)
            t.start()

    def setup_stats_recorder(self):
        """setup an independent thread to record profiling information."""
        stats_thread = StatsRecorder(
            msg_deliver=self.stats_deliver,
            bm_args=self.bm_args,
            workspace=self._workspace,
            bm_board=self.bm_board,
        )
        stats_thread.setDaemon(True)
        stats_thread.start()

    def init_async_train(self):
        """ create train worker """
        self.train_worker = TrainWorker(
            self.send_train,
            self.alg,
            self.train_lock,
            self.model_path,
            self.send_broker,
            self.s3_path,
            self.max_step,
            self.stats_deliver,
            self.test_master,
        )

    def submit_algorithm(self, alg_instance, trainer_obj, shared_buff):
        """submit an algorithm, to update algorithm instance description."""
        self.alg = alg_instance
        self.trainer = trainer_obj
        self.shared_buff = shared_buff

    def start(self):
        """ start all system """
        alg, trainer_obj, shared_list = build_alg_with_trainer(
            deepcopy(self.alg_para), self.send_broker, self.model_path, self.process_num
        )
        self.submit_algorithm(alg, trainer_obj, shared_list)

        self.async_predict()
        self.init_async_train()
        self.setup_stats_recorder()

    def main_loop(self):
        self.train_worker.train()

    def __del__(self):
        if self.bm_board:
            self.bm_board.close()


class TrainWorker(object):
    def __init__(
            self,
            train_q,
            alg,
            lock,
            model_path,
            model_q,
            s3_path,
            max_step,
            stats_deliver,
            test_master=None,
    ):
        self.train_q = train_q
        self.alg = alg
        self.lock = lock
        self.model_path = model_path
        self.model_q = model_q
        self.actor_reward = dict()
        self.rewards = []
        self.s3_path = s3_path
        self.max_step = max_step
        self.actual_step = 0
        self.won_in_episodes = deque(maxlen=256)
        self.train_count = 0

        self.stats_deliver = stats_deliver
        self.test_master = test_master

        self.logger = Logger(os.path.dirname(model_path))

    def _dist_model(self, dist_model_name=("none", "none"), save_index=-1):
        """dist model tool"""
        ctr_info = self.alg.dist_model_policy.get_dist_info(save_index)

        # Not do distribute model with empty list
        if isinstance(ctr_info, list):
            for _ctr in ctr_info:
                to_send_data = message(dist_model_name, cmd="dist_model", **_ctr)
                self.model_q.send(to_send_data)
        else:
            to_send_data = message(dist_model_name, cmd="dist_model", **ctr_info)
            self.model_q.send(to_send_data)

    def train(self):
        """ train model """
        total_count = 0  # if on the off policy, total count > train count
        save_count = 0

        if not self.alg.async_flag:
            _model = self.alg.save(self.model_path, 0)
            full_model_name = [os.path.join(self.model_path, i) for i in _model]
            self._dist_model(dist_model_name=full_model_name)
        while True:
            for _tf_val in range(self.alg.prepare_data_times):
                logging.debug("wait data for preparing-{}...".format(_tf_val))
                with self.logger.wait_sample_timer:
                    data = self.train_q.recv()
                with self.logger.prepare_data_timer:
                    data = bytes_to_str(data)
                    self.record_reward(data)
                    self.alg.prepare_data(data["data"], ctr_info=data["ctr_info"])
                logging.debug("finished prepare data-{}.".format(_tf_val))
                # support sync model before

            if self.max_step and self.actual_step >= self.max_step:
                break

            total_count += 1
            if not self.alg.train_ready(total_count, dist_dummy_model=self._dist_model):
                continue

            with self.lock, self.logger.train_timer:
                logging.debug("start train process-{}.".format(self.train_count))
                loss = self.alg.train(episode_num=total_count)

            if type(loss) in (float, np.float64, np.float32, np.float16, np.float):
                self.logger.record(train_loss=loss)

            self.train_count += 1
            if self.alg.checkpoint_ready(self.train_count):
                with self.lock:
                    if not self.alg.sync_weights:
                        _model = self.alg.save(self.model_path, save_count)
                        # fixme: weights to eval
                        full_model_name = [os.path.join(self.model_path, i) for i in _model]
                    else:
                        full_model_name = self.alg.get_weights()

                    if not self.alg.async_flag:
                        # logging.debug("put full_model_name: {}".format(full_model_name))
                        self._dist_model(dist_model_name=full_model_name, save_index=save_count)

                    # For Cloud
                    if self.s3_path is not None:
                        for name in full_model_name:
                            _model_name = os.path.split(name)[-1]
                            logging.debug(
                                "sync model:{} to s3:{}".format(_model_name, self.s3_path)
                            )
                            sync_data_to_s3(name, os.path.join(self.s3_path, _model_name))

                save_count += 1

                # we only eval saved model
                # fixme: move evaluate logic inside the evaluator
                if self.test_master:
                    self.test_master.call_if_eval(
                        full_model_name[0],
                        self.train_count,
                        self.actual_step,
                        self.logger.elapsed_time,
                        self.logger.train_reward,
                        loss,
                    )
                    eval_ret = self.test_master.fetch_eval_result()
                    if eval_ret:
                        logging.debug("eval stats: {}".format(eval_ret))
                        self.stats_deliver.send(
                            {"data": eval_ret, "is_bm": True}, block=True
                        )
                if save_count % 10 == 9:
                    logging.debug("train count: {}".format(self.train_count))
                    self.stats_deliver.send(self.logger.get_new_info(), block=True)

    def record_reward(self, train_data):
        """ record reward in train """
        broker_id = get_msg_info(train_data, 'broker_id')
        explorer_id = get_msg_info(train_data, 'explorer_id')
        agent_id = get_msg_info(train_data, 'agent_id')
        key = (broker_id, explorer_id, agent_id)

        self.alg.dist_model_policy.add_processed_ctr_info(key)
        data_dict = get_msg_data(train_data)

        # update multi agent train reward without done flag
        if self.alg.alg_name in ("ppo_share_weights",):
            self.actual_step += len(data_dict["done"])
            self.logger.record(
                step=self.actual_step,
                train_reward=np.sum(data_dict["reward"]),
                train_count=self.train_count,
            )
            return
        elif self.alg.alg_name in ("QMixAlg", ):  # fixme: unify the record op
            self.actual_step += np.sum(data_dict["filled"])
            self.won_in_episodes.append(data_dict.pop("battle_won"))
            self.logger.update(explore_won_rate=np.nanmean(self.won_in_episodes))

            self.logger.record(
                step=self.actual_step,
                train_reward=np.sum(data_dict["reward"]),
                train_count=self.train_count,
            )
            return

        if key not in self.actor_reward.keys():
            self.actor_reward[key] = 0.0

        data_length = len(data_dict["done"])  # fetch the train data length
        for data_index in range(data_length):
            reward = data_dict["reward"][data_index]
            done = data_dict["done"][data_index]
            info = data_dict["info"][data_index]
            self.actual_step += 1
            if isinstance(info, dict):
                self.actor_reward[key] += info.get("eval_reward", reward)
                done = info.get("real_done", done)

            if done:
                self.logger.record(
                    step=self.actual_step,
                    train_count=self.train_count,
                    train_reward=self.actor_reward[key],
                )
                self.actor_reward[key] = 0.0


class PredictThread(object):
    def __init__(self, thread_id, alg, request_q, reply_q, stats_deliver, lock):
        self.alg = alg
        self.thread_id = thread_id
        self.request_q = request_q
        self.reply_q = reply_q
        self.lock = lock

        self.stats_deliver = stats_deliver
        self._report_period = 200

        self._stats = PredictStats()

    def predict(self):
        """ predict action """
        while True:

            start_t0 = time()
            data = self.request_q.recv()
            state = get_msg_data(data)
            self._stats.obs_wait_time += time() - start_t0

            start_t1 = time()
            with self.lock:
                action = self.alg.predict(state)
            self._stats.inference_time += time() - start_t1

            set_msg_info(data, cmd="predict_reply")
            set_msg_data(data, action)
            self.reply_q.send(data)

            self._stats.iters += 1
            if self._stats.iters > self._report_period:
                _report = self._stats.get()
                self.stats_deliver.send(_report, block=True)


def patch_alg_within_config(config):
    """combine the algorithm parameters"""
    alg_para = config["alg_para"].copy()
    agent_para = config["agent_para"]
    model_info = config["model_para"]

    node_config = config["node_config"]

    # for quickly run 2s_vs_1sc map
    env_attr = {
        "state_shape": 27,
        # obs_shape with been extend with action&agent id in algorithm!
        "obs_shape": 17,
        "n_actions": 7,
        "n_agents": 2,
        "episode_limit": 300,
        "api_type": "standalone",
        "agent_ids": [0],
    }

    if "alg_config" not in alg_para:
        alg_para["alg_config"] = dict()
        
    alg_para["alg_config"].update(
        {
            "instance_num": config["env_num"] * len(node_config),
            "agent_num": agent_para.get("agent_num", 1),
            "env_attr": env_attr,
        }
    )
    config.update({"alg_para": alg_para})

    # update env attr into model info
    if "model_config" not in model_info["actor"].keys():
        model_info["actor"].update({"model_config": dict()})
    model_info["actor"]["model_config"].update(env_attr)
    alg_para["model_info"] = model_info

    return config


def setup_learner(config, test_master, data_url=None):
    """ start learner """
    env_para = config["env_para"]
    agent_para = config["agent_para"]
    alg_para = deepcopy(config["alg_para"])
    model_info = alg_para["model_info"]

    # set actor.type as learner
    model_info["actor"].update({"type": "learner"})

    # add benchmark id
    bm_info = config.get("benchmark")

    learner = Learner(
        alg_para,
        env_para,
        agent_para,
        test_master=test_master,
        data_url=data_url,
        benchmark_info=bm_info,
    )

    learner.config_info = config

    return learner
