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
"""Build MuZero agent."""

import numpy as np

from xt.agent.agent import Agent
from xt.agent.muzero.default_config import NUM_SIMULATIONS, GAMMA, TD_STEP
from xt.agent.muzero.mcts import Mcts
from zeus.common.util.register import Registers
from zeus.common.util.common import import_config


@Registers.agent
class Muzero(Agent):
    """Build Agent with Muzero algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        import_config(globals(), agent_config)
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS

    def infer_action(self, state, use_explore):
        """
        Infer action.

        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        mcts = Mcts(self, state)
        if use_explore:
            mcts.add_exploration_noise(mcts.root)

        mcts.run_mcts()
        action = mcts.select_action()

        self.transition_data.update({"cur_state": state, "action": action})
        self.transition_data.update(mcts.get_info())

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        info.update({'eval_reward': reward})

        self.transition_data.update({
            "reward": reward,
            "done": done,
            "info": info
        })

        return self.transition_data

    def get_trajectory(self):
        self.data_proc()
        return super().get_trajectory()

    def data_proc(self):
        traj = self.trajectory
        value = traj["root_value"]
        reward = traj["reward"]
        dones = np.asarray(traj["done"])
        discounts = ~dones * GAMMA

        target_value = [reward[-1]] * len(reward)
        for i in range(len(reward) - 1):
            end_index = min(i + TD_STEP, len(reward) - 1)
            sum_value = value[end_index]

            for j in range(i, end_index)[::-1]:
                sum_value = reward[j] + discounts[j] * sum_value

            target_value[i] = sum_value

        self.trajectory["target_value"] = target_value

    def sync_model(self):
        ret_model_name = None
        while True:
            model_name = self.recv_explorer.recv(name=None, block=False)
            if model_name:
                ret_model_name = model_name
            else:
                break

        return ret_model_name

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

        return self.get_trajectory()
