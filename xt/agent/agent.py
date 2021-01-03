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
DESC: Agent module contains all the interaction operations between algorithm and environment.

User could implement the infer_action and handle_env_feedback functions.
"""
from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np

from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.profile_stats import AgentStats


class Agent(object):
    """Build Agent Base."""

    def __init__(self, env, alg, agent_config, recv_explorer, send_explorer, **kwargs):
        self.env = env
        self.alg = alg
        self.agent_config = deepcopy(agent_config)
        self.recv_explorer = recv_explorer
        self.send_explorer = send_explorer
        self.action_dim = alg.action_dim
        self._id = agent_config["agent_id"]

        self.transition_data = defaultdict()
        self.trajectory = defaultdict(list)
        self.max_step = agent_config.get("max_steps", 2000)
        self.infer_if_remote = False
        self.alive = True
        self.keep_seq_len = False

        self.sync_weights_count = 0
        self.broadcast_weights_interval = 1

        self._stats = AgentStats()

    def clear_transition(self):
        pass

    def clear_trajectory(self):
        self.trajectory.clear()

    def get_trajectory(self, last_pred=None):
        """Get trajectory"""
        # Need copy, when run with explore time > 1,
        # if not, will clear trajectory before sent.
        trajectory = message(self.trajectory.copy())
        # trajectory = message(deepcopy(self.trajectory))
        set_msg_info(trajectory, agent_id=self.id)

        return trajectory

    def add_to_trajectory(self, transition_data):
        for k, val in transition_data.items():
            self.trajectory[k].append(val)

    def infer_action(self, state, use_explore):
        """
        Infer an action with the new state.

        User could convert the state into special model's input on there.

        :param state:
        :param use_explore:
            1) False, alg would predict with local model;
            2) True, sync: local predict with model; async: predict with remote
        :return:
        """
        action = self.alg.predict(state)
        self.transition_data.update(
            {"cur_state": state, "action": action}
        )
        if use_explore:
            pass

        raise NotImplementedError

    def do_one_interaction(self, raw_state, use_explore=True):
        """
        Use the Agent do one interaction.

        User could re-write the infer_action and handle_env_feedback functions.
        :param raw_state:
        :param use_explore:
        :return:
        """
        _start0 = time()
        action = self.infer_action(raw_state, use_explore)
        self._stats.inference_time += time() - _start0

        _start1 = time()
        next_raw_state, reward, done, info = self.env.step(action, self.id)
        self._stats.env_step_time += time() - _start1
        self._stats.iters += 1

        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return next_raw_state

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        self.transition_data.update(
            {"next_state": next_raw_state, "reward": reward, "done": done, "info": info}
        )
        raise NotImplementedError

    def run_one_episode(self, use_explore, need_collect):
        """
        Do interaction with max steps in each episode.

        :param use_explore:
        :param need_collect: if collect the total transition of each episode.
        :return:
        """
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.get_init_state(self.id)

        self._stats.reset()

        for _ in range(self.max_step):
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                if not self.keep_seq_len:
                    break
                self.env.reset()
                state = self.env.get_init_state()

        last_pred = self.alg.predict(state)
        return self.get_trajectory(last_pred)

    def sum_trajectory_reward(self):
        """Return the sum of trajectory reward."""
        return {self.id: {
            "epi_reward": np.sum(self.trajectory["reward"]),
            "step_reward": np.mean(self.trajectory["reward"])
        }}

    def calc_custom_evaluate(self):
        """
        Do some custom evaluate process on the whole trajectory of current episode.

        User could overwrite this function to set special evaluate.
        Return a dictionary contains all the key:values by user defined.
        """
        return {self.id: {"custom_criteria": 0.0}}

    @staticmethod
    def post_process(agents):
        """
        Do some operations after all agents run an episode, which within the agent group.

        :param agents:
        :return:
        """
        return 0.0

    def reset(self):
        """
        Do nothing in the base Agent.

        User could do the special reset operation on their agent.
        :return:
        """
        pass

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, agent_id):
        """
        Set agent id.

        :param agent_id:
        :return:
        """
        self._id = agent_id

    def sync_model(self):
        """Fetch model from broker."""
        model_name = self.recv_explorer.recv()
        return model_name

    def get_perf_stats(self):
        """Get status after run once episode."""
        _stats_info = self._stats.get()

        mean_reward = getattr(self, "get_explore_mean_reward", None)
        if mean_reward and callable(mean_reward):
            explore_reward = mean_reward()
            _stats_info.update({"mean_explore_reward": explore_reward})

        return _stats_info
