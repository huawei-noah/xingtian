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
"""Build information flow agent with dqn, which sync model periodicity."""

from __future__ import division, print_function
from copy import deepcopy
import random
import numpy as np
from collections import defaultdict
from xt.agent import Agent
from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.register import Registers


@Registers.agent
class InfoFlowDqn(Agent):
    """Sumo Agent with DQN algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super(InfoFlowDqn, self).__init__(env, alg, agent_config, **kwargs)
        self.epsilon = agent_config["epsilon"]

        self.item_dim = agent_config.get("item_dim")
        self.episode_count = agent_config.get("complete_episode", 100000)

        # print("alg.alg_config", alg.alg_config)
        self.batch_size = alg.alg_config["batch_size"]

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore:
        :return: action value
        """
        # print("state, \n", state.keys(), type(state))
        if use_explore and np.random.rand() < self.epsilon:
            action = random.choice(state["candidate_items"])
        elif use_explore:
            num_candidates = len(state["candidate_items"])

            q_input = dict()
            q_input["user_input"] = np.tile(state["user"], (num_candidates, 1))
            q_input["history_click"] = np.tile(
                np.array(state["clicked_items"]).reshape(-1, self.item_dim * 5),
                (num_candidates, 1),
            )
            q_input["history_no_click"] = np.tile(
                np.array(state["viewed_items"]).reshape(-1, self.item_dim * 5),
                (num_candidates, 1),
            )
            q_input["item_input"] = np.array(state["candidate_items"])
            # q_values = self.alg.actor.model.predict_on_batch(q_input)
            q_values = self.alg.actor.predict(q_input)
            ltvs = np.array(q_values).reshape(-1)
            index = np.argmax(ltvs)
            item = np.array(state["candidate_items"])[index]

            action = item.tolist()
        else:  # evaluate
            # action = random.choice(state["candidate_items"])
            num_candidates = len(state["candidate_items"])
            q_input = dict()
            q_input["user_input"] = np.tile(state["user"], (num_candidates, 1))
            q_input["history_click"] = np.tile(
                np.array(state["clicked_items"]).reshape(-1, self.item_dim * 5),
                (num_candidates, 1),
            )
            q_input["history_no_click"] = np.tile(
                np.array(state["viewed_items"]).reshape(-1, self.item_dim * 5),
                (num_candidates, 1),
            )
            q_input["item_input"] = np.array(state["candidate_items"])
            # q_values = self.alg.actor.model.predict_on_batch(q_input)
            q_values = self.alg.actor.predict(q_input)
            ltvs = np.array(q_values).reshape(-1)
            index = np.argmax(ltvs)
            item = np.array(state["candidate_items"])[index]

            action = item.tolist()

        # update episode value if explore
        # if use_explore:
        #     self.epsilon -= 1.0 / self.episode_count
        #     self.epsilon = max(0.01, self.epsilon)

        # update transition data
        self.transition_data.update({"cur_state": state.copy(), "action": action})

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        self.transition_data.update(
            {"next_state": next_raw_state.copy(), "reward": reward, "done": done, "info": info}
        )
        return self.transition_data

    def clear_transition(self):
        self.transition_data = defaultdict()

    def get_trajectory(self, last_pred=None):
        """Get trajectory"""
        # Need copy, when run with explore time > 1,
        # if not, will clear trajectory before sent.
        # trajectory = message(self.trajectory.copy())
        trajectory = message(deepcopy(self.trajectory))
        set_msg_info(trajectory, agent_id=self.id)

        return trajectory

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

        done = False
        while not done:
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)
            # print("state: ", state)
            # print("transition: ", self.transition_data)
            if need_collect:
                self.add_to_trajectory(self.transition_data.copy())
                # self.add_to_trajectory(copy.deepcopy(self.transition_data))

            done = self.transition_data["done"]

        return self.get_trajectory()
