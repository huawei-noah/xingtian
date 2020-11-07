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
import gym

from xt.environment.environment import Environment


class AtariBaseEnv(Environment, gym.Wrapper):
    """Create atari base wrapper including noop reset and repeat action."""

    def init_env(self, env_info):
        env = gym.make(env_info["name"])
        gym.Wrapper.__init__(self, env)

        self.state_buffer = np.zeros((2, ) + env.observation_space.shape, dtype=np.uint8)
        self.repeat_times = 4
        self.max_noop_times = 30
        self.noop_action = 0

        return env

    def reset(self):
        """Create reset environment and take random noop action."""
        self.env.reset()

        repeat_noop_times = self.unwrapped.np_random.randint(1, self.max_noop_times + 1)
        for _ in range(repeat_noop_times):
            state, _, done, _ = self.env.step(self.noop_action)
            if done:
                state = self.env.reset()

        return state

    def step(self, action, agent_index=0):
        """Take repeat action."""
        total_reward = 0.0
        done = None
        for i in range(self.repeat_times):
            state, reward, done, info = self.env.step(action)
            if i == self.repeat_times - 2:
                self.state_buffer[0] = state
            if i == self.repeat_times - 1:
                self.state_buffer[1] = state
            total_reward += reward
            if done:
                break

        max_frame = self.state_buffer.max(axis=0)

        return max_frame, total_reward, done, info


class AtariRealDone(Environment, gym.Wrapper):
    """Create atari real done wrapper, reset environment when real done."""

    def init_env(self, env):
        gym.Wrapper.__init__(self, env)

        self.lives = 0
        self.real_done = True
        return env

    def reset(self):
        if self.real_done:
            state = self.env.reset()
        else:
            state, _, done, _ = self.env.step(0)
            if done:
                state = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()

        return state

    def step(self, action, agent_index=0):
        state, reward, done, info = self.env.step(action)

        self.real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True

        self.lives = lives
        info.update({'real_done': self.real_done})
        return state, reward, done, info


class AtariFireRest(Environment, gym.Wrapper):
    def init_env(self, env):
        gym.Wrapper.__init__(self, env)
        return env

    def reset(self):
        """Take action after reset."""
        state = self.env.reset()
        state, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        state, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()

        return state


def make_atari(env_info):
    env = AtariBaseEnv(env_info)
    env = AtariRealDone(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = AtariFireRest(env)

    return env
