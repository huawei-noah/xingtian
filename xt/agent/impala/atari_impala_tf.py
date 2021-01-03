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
"""Optimize the Atiri agent for IMPALA algorithm with tf.graph."""

import numpy as np

from xt.agent.impala.cartpole_impala import CartpoleImpala
from xt.agent.impala.cartpole_impala_tf import CartpoleImpalaTf
from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.register import Registers


@Registers.agent
class AtariImpalaTf(CartpoleImpala):
    """Build Atari agent with IMPALA algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):

        super(AtariImpalaTf, self).__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True  # to keep max sequence length in explorer.
        self.next_logit = None
        self.broadcast_weights_interval = 2
        self.sync_weights_count = 0

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore:
        :return: action value
        """
        if self.next_state is None:
            s_t = np.expand_dims((state.astype("int16") - 128).astype("int8"), axis=0)
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
        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        """Overwrite handle env feedback."""
        next_state = (next_raw_state.astype("int16") - 128).astype('int8')
        predict_val = self.alg.predict(np.expand_dims(next_state, axis=0))

        self.next_logit = predict_val[0][0]
        self.next_value = predict_val[1][0]
        self.next_action = predict_val[2][0]

        self.next_state = next_state

        info.update({'eval_reward': reward})
        done = info.get('real_done', done)  # fixme: real done

        self.transition_data.update({
            # "next_state": next_raw_state,
            # "reward": np.sign(reward) if use_explore else reward,
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

        trajectory = message(self.trajectory.copy())
        set_msg_info(trajectory, agent_id=self.id)
        return trajectory

    def sync_model(self):
        model_name = "none"
        self.sync_weights_count += 1
        if self.sync_weights_count >= self.broadcast_weights_interval:
            model_name = self.recv_explorer.recv(block=True)
            self.sync_weights_count = 0

            try:
                while True:
                    model_name = self.recv_explorer.recv(block=False)
            except:
                pass

        return model_name
