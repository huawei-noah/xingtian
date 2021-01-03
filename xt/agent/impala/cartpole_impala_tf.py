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
"""Optimize the CartPole agent for IMPALA algorithm with tf.graph."""

import numpy as np

from xt.agent.ppo.cartpole_ppo import CartpolePpo
from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.register import Registers


@Registers.agent
class CartpoleImpalaTf(CartpolePpo):
    """Cartpole Agent with IMPALA algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):

        super(CartpoleImpalaTf, self).__init__(
            env, alg, agent_config, **kwargs
        )
        self.keep_seq_len = True  # to keep max sequence length in explorer.
        self.next_logit = None

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore:
        :return: action value
        """
        if self.next_state is None:
            s_t = state
            predict_val = self.alg.predict(s_t)
            logit = predict_val[0][0]
            value = predict_val[1][0]
            action = predict_val[2][0]
        else:
            s_t = self.next_state
            logit = self.next_logit
            action = self.next_action
            value = self.next_value

        # update transition data
        self.transition_data.update({
            "cur_state": s_t,
            "logit": logit,
            # "value": value,
            "action": action
        })

        # print("logit, value, action: ", logit, value, action)
        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        predict_val = self.alg.predict(next_raw_state)
        self.next_logit = predict_val[0][0]
        self.next_value = predict_val[1][0]
        self.next_action = predict_val[2][0]

        self.next_state = next_raw_state

        self.transition_data.update({
            # "next_state": next_raw_state,
            "reward": reward,
            # "next_value": self.next_value,
            "done": done,
            "info": info
        })

        return self.transition_data

    def get_trajectory(self, last_pred=None):
        for _data_key in ("cur_state", "logit", "action"):
            self.trajectory[_data_key] = np.asarray(self.trajectory[_data_key])

        self.trajectory["action"].astype(np.int32)
        # self.trajectory["cur_state"].astype(np.int32)
        # print(self.trajectory)

        trajectory = message(self.trajectory)
        set_msg_info(trajectory, agent_id=self.id)
        return trajectory

    def sync_model(self):
        model_name = "none"
        try:
            while True:
                model_name = self.recv_explorer.recv(block=False)
        except:
            pass
        return model_name
