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
"""Build CartPole agent for dqn algorithm."""

import random
import numpy as np

from xt.agent import Agent

from zeus.common.util.register import Registers
from zeus.common.ipc.message import message


@Registers.agent
class CartpoleDqn(Agent):
    """Build Cartpole Agent with DQN algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super(CartpoleDqn, self).__init__(env, alg, agent_config, **kwargs)
        self.epsilon = 1.0
        self.episode_count = agent_config.get("episode_count", 100000)

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore: Used True, in train, False in evaluate
        :return: action value
        """
        # if explore action
        if use_explore and random.random() < self.epsilon:
            action = np.random.randint(0, self.alg.action_dim)
        elif use_explore:  # explore with remote predict
            # Get Q values with deliver for each action.
            send_data = message(state, cmd="predict")
            self.send_explorer.send(send_data)
            action = self.recv_explorer.recv()
        else:  # don't explore, used in evaluate
            action = self.alg.predict(state)

        # update episode value
        if use_explore:
            self.epsilon -= 1.0 / self.episode_count
            self.epsilon = max(0.01, self.epsilon)

        # update transition data
        self.transition_data.update(
            {"cur_state": state, "action": action}
        )

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        self.transition_data.update({
            "next_state": next_raw_state,
            "reward": np.sign(reward) if use_explore else reward,
            "done": done,
            "info": info
        })

        # deliver this transition data to learner, trigger train process.
        if use_explore:
            train_data = {k: [v] for k, v in self.transition_data.items()}
            train_data = message(train_data, agent_id=self.id)
            self.send_explorer.send(train_data)

        return self.transition_data

    def sync_model(self):
        return None
