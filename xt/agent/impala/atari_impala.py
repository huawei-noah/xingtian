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
import numpy as np

from xt.agent.impala.cartpole_impala import CartpoleImpala
from zeus.common.util.register import Registers
from zeus.common.ipc.message import message, set_msg_info


@Registers.agent
class AtariImpala(CartpoleImpala):
    """Build Atari agent with IMPALA algorithm."""

    def infer_action(self, state, use_explore):
        state = state.astype('uint8')
        real_action = super().infer_action(state, use_explore)

        return real_action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        next_state = next_raw_state.astype('uint8')
        predict_val = self.alg.predict(next_state)

        self.next_action = predict_val[0][0]
        self.next_value = predict_val[1][0]
        self.next_state = next_state

        info.update({'eval_reward': reward})

        self.transition_data.update({
            "next_state": next_state,
            "reward": np.sign(reward) if use_explore else reward,
            "next_value": self.next_value,
            "done": done,
            "info": info
        })

        return self.transition_data
