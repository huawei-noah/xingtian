"""
Use multiagent environment from smac.

```
def get_env_info(self):
    env_info = {"state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit}
    return env_info
```
"""

import os
import sys

from absl import logging
from smac.env import MultiAgentEnv, StarCraft2Env
from xt.environment.environment import Environment
from zeus.common.util.register import Registers

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


@Registers.env
class StarCraft2Xt(Environment):
    """Make starcraft II simulation into xingtian's environment."""

    def init_env(self, env_info):
        logging.debug("init env with: {}".format(env_info))
        print(env_info)
        sys.stdout.flush()
        _info = env_info.copy()
        if "agent_num" in _info.keys():
            _info.pop("agent_num")
        return StarCraft2Env(**_info)

    def reset(self):
        """
        Reset the environment. starcraft env need get obs & global status.

        :return: None
        """
        self.env.reset()
        return None

    def step(self, action, agent_index=0):
        """Make a simplest step in starcraft."""
        reward, done, info = self.env.step(action)
        return reward, done, info

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_obs(self):
        return self.env.get_obs()

    def get_env_info(self):
        """Return environment's basic information."""
        self.reset()
        env_attr = self.env.get_env_info()
        env_attr.update(
            {"api_type": self.api_type, }
        )
        # update the agent ids, will used in the weights map.
        # default work well with the sumo multi-agents
        # starcraft multi-agents consider as a batch agents.
        env_attr.update({"agent_ids": [0, ]})

        return env_attr
