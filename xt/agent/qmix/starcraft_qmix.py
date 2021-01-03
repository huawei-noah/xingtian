"""Build starCraft agent with QMix algorithm."""

from functools import partial
from time import time

from xt.agent import Agent
from xt.algorithm.qmix.episode_buffer_np import EpisodeBatchNP
from xt.algorithm.qmix.transforms import OneHotNp
from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.register import Registers


@Registers.agent
class StarCraftQMix(Agent):
    """DESC: QMix combines multi-agents into one batch data set."""

    def __init__(self, env, alg, agent_config, **kwargs):
        """Set explore sun-graph in agent, and update max_steps with env.info."""
        env_info = env.get_env_info()

        agent_config.update({"max_steps": env_info["episode_limit"]})
        super().__init__(env, alg, agent_config, **kwargs)

        self.timestamp_per_agent = 0
        self.t_env = 0
        self.n_episode = 0
        self.episode_limit = env_info["episode_limit"]

        groups = {"agents": env_info["n_agents"]}
        preprocess_np = {
            "actions": ("actions_onehot", [OneHotNp(out_dim=env_info["n_actions"])])
        }
        self.batch = None
        self.new_batch = self.setup(
            scheme=alg.scheme, groups=groups, preprocess=preprocess_np)

        self._info = dict(battle_won=False)
        self._reward = 0

    def setup(self, scheme, groups, preprocess):
        return partial(
            EpisodeBatchNP,
            scheme,
            groups,
            1,  # Note: batch size must be 1 in a episode
            self.episode_limit + 1,
            preprocess=preprocess,
        )

    def reset(self):
        self.alg.reset_hidden_state()
        self.batch = self.new_batch()
        self.env.reset()
        self.timestamp_per_agent = 0

        self._info = dict(battle_won=False)
        self._reward = 0

    def infer_action(self, state, use_explore):
        """Rewrite with predict_with_selector."""
        pass

    def do_one_interaction(self, raw_state, use_explore=True):
        """Overwrite with obs and global states."""
        # get stats, avail_actions and obs from environment.
        pre_transition_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(pre_transition_data, ts=self.timestamp_per_agent)

        _start0 = time()
        actions = self.alg.predict_with_selector(
            self.batch, self.timestamp_per_agent, self.t_env, not use_explore)
        self._stats.inference_time += time() - _start0

        _start1 = time()
        reward, done, info = self.env.step(actions[0], self.id)

        self._stats.env_step_time += time() - _start1
        self._stats.iters += 1

        self.handle_env_feedback(actions, reward, done, info, use_explore)

    def handle_env_feedback(self, actions, reward, done, info, use_explore):
        """Overwrite env feedback."""
        post_transition_data = {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(done != info.get("episode_limit", False),)],
        }

        self.batch.update(post_transition_data, ts=self.timestamp_per_agent)

        self.timestamp_per_agent += 1
        if done:  # record the env info for evaluate or others
            self._info.update(info)
        self.transition_data.update({"done": done})
        self._reward += reward

        return post_transition_data

    def run_one_episode(self, use_explore, need_collect):
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.get_init_state(self.id)

        self._stats.reset()

        for _ in range(self.max_step):
            self.clear_transition()
            self.do_one_interaction(state, use_explore)

            if self.transition_data["done"]:
                break

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.timestamp_per_agent)

        # Select actions in the last stored state
        actions = self.alg.predict_with_selector(
            self.batch, t_ep=self.timestamp_per_agent,
            t_env=self.t_env, test_mode=not use_explore
        )

        self.batch.update({"actions": actions}, ts=self.timestamp_per_agent)

        self.t_env += self.timestamp_per_agent  # NOTE: each agent only has self.t_env

        return self.get_trajectory()

    def get_trajectory(self):
        transition = self.batch.data.transition_data
        transition.update(self._info.copy())  # record win rate within train
        trajectory = message(transition)
        set_msg_info(trajectory, agent_id=self.id)
        return trajectory

    def sum_trajectory_reward(self):
        """Return the sum of trajectory reward."""
        return {self.id: self._reward}

    def calc_custom_evaluate(self):
        """Calculate the win rate."""
        return {self.id: self._info.copy()}
