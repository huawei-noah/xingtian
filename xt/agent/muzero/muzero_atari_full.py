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
from xt.agent.muzero.default_config import NUM_SIMULATIONS
from zeus.common.util.register import Registers


@Registers.agent
class MuzeroAtariFull(Muzero):
    """ Agent with Muzero algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.num_simulations = NUM_SIMULATIONS
        self.history_acton = np.zeros((96, 96, 32)).astype('uint8')

    def infer_action(self, state, use_explore):
        """
        We then run a Monte Carlo Tree Search using only action sequences and the
        model learned by the networks.
        """
        state = state.astype('uint8')
        action_plane = np.asarray(self.history_acton).astype('uint8')
        # print("all shape", state.shape, action_plane.shape)
        state = np.concatenate((state, action_plane), axis=-1)

        mcts = Mcts(self, state)
        if use_explore:
            mcts.add_exploration_noise(mcts.root)

        mcts.run_mcts()
        action = mcts.select_action()

        action_plane = np.full((96, 96, 1), action * 14, dtype='uint8')
        self.history_acton = np.roll(self.history_acton, shift=-1, axis=-1)
        self.history_acton[..., -action_plane.shape[-1]:] = action_plane

        self.transition_data.update({"cur_state": state, "action": action})
        self.transition_data.update(mcts.get_info())

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        if done:
            self.history_acton = np.zeros((96, 96, 32)).astype('uint8')
        super().handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        return self.transition_data
