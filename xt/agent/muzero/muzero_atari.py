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
from xt.agent.muzero.muzero import Muzero
from xt.agent.muzero.mcts import Mcts
from xt.agent.muzero.default_config import NUM_SIMULATIONS, GAMMA, TD_STEP
from zeus.common.util.register import Registers


@Registers.agent
class MuzeroAtari(Muzero):
    """ Agent with Muzero algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS
        self.keep_seq_len = True

    def infer_action(self, state, use_explore):
        """
        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        state = state.astype('uint8')
        action = super().infer_action(state, use_explore)

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):

        next_raw_state = next_raw_state.astype('uint8')
        super().handle_env_feedback(next_raw_state, reward, done, info, use_explore)

        return self.transition_data
