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
"""Create a module cover the training process within the RL problems."""

import os
import threading
from time import time
from copy import deepcopy
import numpy as np
from absl import logging
from collections import deque
from zeus.visual.tensorboarder import SummaryBoard
from zeus.common.util.evaluate_xt import make_workspace_if_not_exist, parse_benchmark_args
from xt.environment import env_builder
from zeus.common.ipc.message import message, get_msg_data, set_msg_info, set_msg_data, get_msg_info
from xt.framework.trainer import build_alg_with_trainer
from zeus.common.util.common import bytes_to_str
from zeus.common.util.hw_cloud_helper import mox_makedir_if_not_existed
from zeus.common.util.logger import Logger, StatsRecorder
from zeus.common.util.profile_stats import PredictStats, TimerRecorder


class Learner(object):
    """Learner manage the train-processing of whole RL pipe-line."""

    def __init__(
            self,
            alg_para,
            env_para,
            agent_para,
            eval_adapter=None,
            data_url=None,
            benchmark_info=None,
    ):
        self.alg_para = deepcopy(alg_para)
        self.process_num = self.alg_para.get("process_num", 1)

        self.eval_adapter = eval_adapter

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

        self.bm_board = SummaryBoard(_archive, _job)
        self.model_path = os.path.join(self._workspace, _model_dir[0])

        logging.info("{}\nworkspace:\n\t{}\n".format("*" * 10, self._workspace))

        self.max_step = agent_para.get("agent_config", {}).get("complete_step")

        # For Cloud
        self.s3_path = None
        if data_url is not None:
            self.s3_path = os.path.join(data_url, _model_dir[0])
            mox_makedir_if_not_existed(self.s3_path)

    def async_predict(self):
        """Create predict thread."""
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
        """Create an independent thread to record profiling information."""
        stats_thread = StatsRecorder(
            msg_deliver=self.stats_deliver,
            bm_args=self.bm_args,
            workspace=self._workspace,
            bm_board=self.bm_board,
        )
        stats_thread.setDaemon(True)
        stats_thread.start()

    def init_async_train(self):
        """Create train worker."""
        self.train_worker = TrainWorker(
            self.send_train,
            self.alg,
            self.train_lock,
            self.model_path,
            self.send_broker,
            self.s3_path,
            self.max_step,
            self.stats_deliver,
            self.eval_adapter,
        )

    def submit_algorithm(self, alg_instance, trainer_obj, shared_buff):
        """Submit an algorithm, to update algorithm instance description."""
        self.alg = alg_instance
        self.trainer = trainer_obj
        self.shared_buff = shared_buff

    def start(self):
        """Start all system."""
        alg, trainer_obj, shared_list = build_alg_with_trainer(
            deepcopy(self.alg_para), self.send_broker, self.model_path, self.process_num
        )
        self.submit_algorithm(alg, trainer_obj, shared_list)

        self.async_predict()
        self.init_async_train()
        self.setup_stats_recorder()

    def main_loop(self):
        """Run with while True, cover the working loop."""
        self.train_worker.train()
        # user operation after train process

    def __del__(self):
        if self.bm_board:
            self.bm_board.close()


class TrainWorker(object):
    """TrainWorker Process manage the trajectory data set and optimizer."""

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
            eval_adapter=None,
    ):
        self.train_q = train_q
        self.alg = alg
        self.lock = lock
        self.model_path = model_path
        self.model_q = model_q
        self.actor_reward = dict()
        self.actor_trajectory = dict()
        self.rewards = []
        self.s3_path = s3_path
        self.max_step = max_step
        self.actual_step = 0
        self.won_in_episodes = deque(maxlen=256)
        self.train_count = 0

        self.stats_deliver = stats_deliver
        self.e_adapter = eval_adapter

        self.logger = Logger(os.path.dirname(model_path))
        self._metric = TimerRecorder("leaner_model", maxlen=50,
                                     fields=("fix_weight", "send"))

    def _dist_policy(self, weight=None, save_index=-1, dist_cmd="explore"):
        """Distribute model tool."""
        ctr_info = self.alg.dist_model_policy.get_dist_info(save_index)

        if isinstance(ctr_info, dict):
            ctr_info = [ctr_info]

        for _ctr in ctr_info:
            to_send_data = message(weight, cmd=dist_cmd, **_ctr)
            self.model_q.send(to_send_data)

    def _handle_eval_process(self, loss):
        if self.e_adapter and self.e_adapter.if_eval(self.train_count):
            weights = self.alg.get_weights()
            self.e_adapter.to_eval(weights,
                                   self.train_count,
                                   self.actual_step,
                                   self.logger.elapsed_time,
                                   self.logger.train_reward,
                                   loss)
            eval_ret = self.e_adapter.fetch_eval_result()
            if eval_ret:
                logging.debug("eval stats: {}".format(eval_ret))
                self.stats_deliver.send({"data": eval_ret, "is_bm": True}, block=True)

    def train(self):
        """Train model."""
        total_count = 0  # if on the off policy, total count > train count
        save_count = 0

        if not self.alg.async_flag:
            policy_weight = self.alg.get_weights()
            self._dist_policy(weight=policy_weight)

        while True:
            for _tf_val in range(self.alg.prepare_data_times):
                logging.debug("wait data for preparing-{}...".format(_tf_val))
                with self.logger.wait_sample_timer:
                    data = self.train_q.recv()
                with self.logger.prepare_data_timer:
                    data = bytes_to_str(data)
                    self.record_reward(data)
                    self.alg.prepare_data(data["data"], ctr_info=data["ctr_info"])
                logging.debug("Prepared data-{}.".format(_tf_val))
                # support sync model before

            if self.max_step and self.actual_step >= self.max_step:
                break

            total_count += 1
            if not self.alg.train_ready(total_count, dist_dummy_model=self._dist_policy):
                continue

            with self.lock, self.logger.train_timer:
                logging.debug("start train process-{}.".format(self.train_count))
                loss = self.alg.train(episode_num=total_count)
                self.train_count += 1

            if type(loss) in (float, np.float64, np.float32, np.float16, np.float):
                self.logger.record(train_loss=loss)

            # The requirement of distribute model is checkpoint ready.
            # if self.alg.checkpoint_ready(self.train_count):
            with self.lock:
                if self.alg.if_save(self.train_count):
                    _name = self.alg.save(self.model_path, self.train_count)
                    # logging.debug("to save model: {}".format(_name))

            self._handle_eval_process(loss)

            if not self.alg.async_flag and self.alg.checkpoint_ready(
                    self.train_count):

                _save_t1 = time()
                policy_weight = self.alg.get_weights()
                self._metric.append(fix_weight=time() - _save_t1)

                _dist_st = time()
                self._dist_policy(policy_weight, save_count)
                self._metric.append(send=time() - _dist_st)
                self._metric.report_if_need()

            save_count += 1
            if save_count % 5 == 1:
                self.stats_deliver.send(self.logger.get_new_info(), block=True)

    def record_reward(self, train_data):
        """Record reward in train."""
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
            self.actor_trajectory[key] = 0

        data_length = len(data_dict["done"])  # fetch the train data length
        for data_index in range(data_length):
            reward = data_dict["reward"][data_index]
            done = data_dict["done"][data_index]
            info = data_dict["info"][data_index]
            self.actual_step += 1
            if isinstance(info, dict):
                self.actor_reward[key] += info.get("eval_reward", reward)
                self.actor_trajectory[key] += 1
                done = info.get("real_done", done)

            if done:
                self.logger.record(
                    step=self.actual_step,
                    train_count=self.train_count,
                    train_reward=self.actor_reward[key],
                )

                logging.debug("{} epi reward-{}. with len-{}".format(
                    key, self.actor_reward[key], self.actor_trajectory[key]))

                self.actor_reward[key] = 0.0
                self.actor_trajectory[key] = 0


class PredictThread(object):
    """Predict Worker for async algorithm."""

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
        """Predict action."""
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

            # logging.debug("msg to explore: ", data)
            self.reply_q.send(data)

            self._stats.iters += 1
            if self._stats.iters > self._report_period:
                _report = self._stats.get()
                self.stats_deliver.send(_report, block=True)


def patch_model_config_by_env_info(config):
    model_info = config["model_para"]
    if "model_config" not in model_info["actor"].keys():
        model_info["actor"].update({"model_config": dict()})
    model_config = model_info["actor"]["model_config"]

    env = env_builder(**config["env_para"])
    env_info = env.get_env_info()
    model_config.update({"action_type": env_info.get("action_type")})
    env.close()

    return model_info


def patch_alg_within_config(config):
    """combine the algorithm parameters"""
    alg_para = config["alg_para"].copy()
    agent_para = config["agent_para"]
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

    model_info = patch_model_config_by_env_info(config)
    # update env attr into model info
    model_info["actor"]["model_config"].update(env_attr)
    alg_para["model_info"] = model_info
    config.update({"alg_para": alg_para})

    return config


def setup_learner(config, eval_adapter, data_url=None):
    """Start learner."""
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
        eval_adapter=eval_adapter,
        data_url=data_url,
        benchmark_info=bm_info,
    )

    learner.config_info = config

    return learner
