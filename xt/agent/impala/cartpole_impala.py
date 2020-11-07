# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Build Cartpole agent for IMPALA algorithm."""

import numpy as np
from xt.agent.ppo.cartpole_ppo import CartpolePpo
from zeus.common.util.register import Registers


@Registers.agent
class CartpoleImpala(CartpolePpo):
    """Build Cartpole Agent with Impala algorithm."""

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        predict_val = self.alg.predict(next_raw_state)
        self.next_action = predict_val[0][0]
        self.next_value = predict_val[1][0]
        self.next_state = next_raw_state

        self.transition_data.update({
            "next_state": next_raw_state,
            "reward": reward,
            "next_value": self.next_value,
            "done": done,
            "info": info
        })

        return self.transition_data

    def data_proc(self):
        episode_data = self.trajectory
        states = episode_data["cur_state"]

        actions = np.asarray(episode_data["real_action"])
        actions = np.eye(self.action_dim)[actions.reshape(-1)]

        pred_a = np.asarray(episode_data["action"])
        states.append(episode_data["last_state"])

        states = np.asarray(states)
        self.trajectory["cur_state"] = states
        self.trajectory["real_action"] = actions
        self.trajectory["action"] = pred_a

    def add_to_trajectory(self, transition_data):
        for k, val in transition_data.items():
            if k is "next_state":
                self.trajectory.update({"last_state": val})
            else:
                if k not in self.trajectory.keys():
                    self.trajectory.update({k: [val]})
                else:
                    self.trajectory[k].append(val)

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

        for _ in range(self.max_step):
            self.clear_transition()

            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                self.env.reset()
                state = self.env.get_init_state(self.id)

        traj = self.get_trajectory()
        return traj
