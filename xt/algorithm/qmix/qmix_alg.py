"""
Implement the qmix algorithm with tensorflow, also thanks to the pymarl repo.

https://github.com/oxwhirl/pymarl  with pytorch implemented.

"""

import os
import sys
from types import SimpleNamespace as SN

import numpy as np

from absl import logging
# from attrdict import AttrDict
from xt.model.tf_compat import tf
from xt.algorithm import Algorithm
from xt.algorithm.algorithm import ZFILL_LENGTH
from xt.algorithm.qmix.episode_buffer_np import EpisodeBatchNP, ReplayBufferNP
from xt.algorithm.qmix.transforms import OneHotNp
from zeus.common.util.register import Registers


class DecayThenFlatSchedule(object):
    """Schedule from pymarl."""

    def __init__(self, start, finish, time_length, decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = ((-1) * self.time_length / np.log(self.finish)
                                if self.finish > 0 else 1)

    def eval(self, t):
        """Schedule with eval times."""
        val = 0
        if self.decay in ["linear"]:
            val = max(self.finish, self.start - self.delta * t)
        elif self.decay in ["exp"]:
            val = min(self.start, max(self.finish, np.exp(-t / self.exp_scaling)))
        else:
            raise KeyError("invalid decay-{} configured".format(self.decay))
        return val


class EpsilonGreedyActionSelector(object):
    """Create epsilon greedy action selector from pymarl."""

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args["epsilon_start"],
            args["epsilon_finish"],
            args["epsilon_anneal_time"],
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Assume agent_inputs is a batch of Q-Values for each agent bav."""
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        # masked_q_values = agent_inputs.clone()
        # we make the `agent_inputs` value as a numpy !!!
        masked_q_values = agent_inputs.copy()
        # should never be selected!
        # masked_q_values[avail_actions == 0.0] = -float("inf")
        masked_q_values[avail_actions < 1e-6] = -float("inf")

        # random_numbers = th.rand_like(agent_inputs[:, :, 0])
        random_numbers = np.random.rand(*agent_inputs[:, :, 0].shape)
        # pick_random = (random_numbers < self.epsilon).long()
        pick_random = np.array(random_numbers < self.epsilon).astype(np.long)
        _batch_size, agent_num, action_dim = avail_actions.shape
        # avail_actions = avail_actions.reshape(-1, last_dim)
        avail_norm_p = np.array(
            avail_actions / np.expand_dims(avail_actions.sum(-1), -1)).astype(np.float)
        avail_norm_p = avail_norm_p.reshape(-1, action_dim)

        random_actions = np.array(
            [np.random.choice(action_dim, p=pi)
             for pi in avail_norm_p]).reshape((-1, agent_num))

        picked_actions = pick_random * random_actions + \
            (1 - pick_random) * masked_q_values.argmax(axis=2)
        # print(picked_actions, picked_actions.shape)
        # # [[5. 5.]] (1, 2)
        return picked_actions


@Registers.algorithm
class QMixAlg(Algorithm):
    """Implemente q-mix algorithm with tensorflow framework."""

    def __init__(
            self,
            model_info: dict,
            alg_config: dict,
            **kwargs,
    ):
        """
        Initialize.

        Consider the compatibility between trainer and explorer,
        QMix Algorithm could support the two scene
        """
        # avail_actions vary with env.map
        # alg_config = AttrDict(alg_config)
        logging.debug("get alg_config: {}".format(alg_config))

        env_info = alg_config["env_attr"]
        alg_config.update({
            "n_agents": env_info["n_agents"],
            "n_actions": env_info["n_actions"],
            "state_shape": env_info["state_shape"],
        })

        self.n_agents = alg_config["n_agents"]  # from env

        self.scheme = {
            "state": {
                "vshape": env_info["state_shape"]
            },
            "obs": {
                "vshape": env_info["obs_shape"],
                "group": "agents"
            },
            "actions": {
                "vshape": (1, ),
                "group": "agents",
                "dtype": np.int64
            },
            "avail_actions": {
                "vshape": (env_info["n_actions"], ),
                "group": "agents",
                "dtype": np.int32,
            },
            "reward": {
                "vshape": (1, )
            },
            "terminated": {
                "vshape": (1, ),
                "dtype": np.uint8
            },
            "actions_onehot": {
                "vshape": (env_info["n_actions"], ),
                "dtype": np.float32,
                "group": "agents",
            },
            # "filled": {"vshape": (1,), "dtype": tf.int64},
        }
        self.obs_shape = self._get_input_shape(alg_config, self.scheme)
        logging.debug("obs_shape: {}".format(self.obs_shape))

        logging.debug("update obs shape: {} --> {}".format(
            model_info["actor"]["model_config"]["obs_shape"], self.obs_shape))
        model_info["actor"]["model_config"]["obs_shape"] = self.obs_shape

        # NOTE: set graph scene, train as default
        model_info["actor"].update({"scene": kwargs.get("scene", "train")})

        super(QMixAlg, self).__init__(alg_name="QMixAlg",
                                      model_info=model_info["actor"],
                                      alg_config=alg_config)

        self.async_flag = False

        self.previous_state = None
        self.ph_hidden_states_in = None
        self.hidden_states_out = None

        self.params = None
        self.inputs = None
        self.out_actions = None

        self.avail_action_num = env_info["n_actions"]

        # use the episode limit as fix shape.
        self.fix_seq_length = env_info["episode_limit"]

        self.schedule = DecayThenFlatSchedule(
            alg_config["epsilon_start"],
            alg_config["epsilon_finish"],
            alg_config["epsilon_anneal_time"],
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

        # select action
        self.selector = EpsilonGreedyActionSelector(alg_config)

        # # mix
        # self.state_dim = int(np.prod(alg_config.state_shape))
        # self.embed_dim = alg_config.mixing_embed_dim

        # update target with period, while update explorer each times.
        self.last_target_update_episode = -9999.0

        self.groups = {"agents": env_info["n_agents"]}
        preprocess_np = {"actions": ("actions_onehot",
                                     [OneHotNp(out_dim=alg_config["n_actions"])])}

        self.buffer = ReplayBufferNP(
            self.scheme,
            self.groups,
            alg_config["buffer_size"],
            env_info["episode_limit"] + 1,
            preprocess=preprocess_np,
        )
        self.train_batch = None
        self.train_times = 0

    @staticmethod
    def _get_input_shape(alg_config, scheme):
        """Assemble input shape with alg_config, vary with last_action/agent_id."""
        input_shape = scheme["obs"]["vshape"]
        if alg_config["obs_last_action"]:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if alg_config["obs_agent_id"]:
            input_shape += alg_config["n_agents"]
        return input_shape

    def reset_hidden_state(self):
        self.actor.reset_hidden_state()

    def build_inputs(self, batch, t):
        """
        Build inputs.

        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        1. inference stage, use batch = 1,
        2. train stage, use batch = episode.limit

        Also, use numpy for combine the inputs data
        """
        _batch_size = batch.batch_size
        inputs = list()
        inputs.append(batch["obs"][:, t])  # b1av

        if self.alg_config["obs_last_action"]:
            if t == 0:
                inputs.append(np.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.alg_config["obs_agent_id"]:
            _ag_id = np.expand_dims(np.eye(self.n_agents), axis=0)  # add axis 0
            inputs.append(np.tile(_ag_id, (_batch_size, 1, 1)))  # broadcast_to

        inputs = np.expand_dims(np.concatenate(inputs, axis=-1), axis=1)

        return inputs

    def predict_with_selector(self, ep_batch, t_ep, t_env, test_mode):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_inputs = self.build_inputs(ep_batch, t_ep)

        out_val = self.actor.infer_actions(agent_inputs)

        select_actions = self.selector.select_action(
            out_val, avail_actions, t_env, test_mode=test_mode)
        # print("out_val: {}, select action: {}, avail_actions, {}, t_env:{}".format(
        #     out_val, select_actions, avail_actions, t_env))
        return select_actions

    def save(self, model_path, model_index):
        """Save qmix explore agent weight with saver."""
        model_name = os.path.join(
            model_path, "actor{}".format(str(model_index).zfill(ZFILL_LENGTH)))
        self.actor.save_explore_agent_weights(model_name)
        return [model_name]

    def restore(self, model_name=None, model_weights=None):
        """
        Restore the model with the priority: model_weight > model_name.
        owing to actor.set_weights would be faster than load model from disk.

        if user used multi model in one algorithm,
        they need overwrite this function.
        And, Caller make the name/weights valid.
        """
        if model_weights is not None:
            self.actor.set_weights(model_weights)
        else:
            logging.debug("{} try load model: {}".format(self.alg_name, model_name))
            self.actor.restore_explorer_variable(model_name)

    @staticmethod
    def _new_data_sn():
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def prepare_data(self, train_data, **kwargs):
        """Insert trajectory into buffer, and sample batch if meet required."""
        new_data = self._new_data_sn()
        # received train data batch size ==1, seq_length==limit episode
        for k, val in train_data.items():
            new_data.transition_data[k] = val

        deliver_batch = EpisodeBatchNP(
            self.scheme, self.groups, 1, self.fix_seq_length + 1, data=new_data)
        self.buffer.insert_episode_batch(deliver_batch)

        if self.buffer.can_sample(self.alg_config["batch_size"]):
            self.train_batch = self.buffer.sample(self.alg_config["batch_size"])
        else:
            self.train_batch = None

    def train(self, **kwargs):
        """Train with buffer sampled."""
        # , batch: EpisodeBatchNP, t_env: int, episode_num: int
        if not self.train_batch:
            return np.nan

        episode_num = kwargs.get("episode_num")
        if not episode_num:
            raise KeyError("need episode num to update target network")

        batch = self.train_batch

        # Truncate batch to only filled timesteps
        max_ep_t = batch.max_t_filled()
        logging.debug("episode sample with max_ep_t: {}".format(max_ep_t))
        # batch = batch[:, :max_ep_t]

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].astype(np.float32)
        mask = batch["filled"][:, :-1].astype(np.float32)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # [bs, seq_len, n_agents, obs_size] [32, 1, 2, 26] --> [32, 301, 2, 26]
        _inputs = [self.build_inputs(batch, t) for t in range(batch.max_seq_length)]

        batch_trajectories = np.concatenate(_inputs, axis=1)

        logging.debug("batch_trajectories.shape: {}".format(batch_trajectories.shape))
        logging.debug("rewards.shape: {}".format(rewards.shape))
        logging.debug("actions.shape: {}".format(actions.shape))
        logging.debug("terminated.shape: {}".format(terminated.shape))
        logging.debug("mask.shape: {}".format(mask.shape))
        logging.debug("avail_actions.shape: {}".format(avail_actions.shape))
        logging.debug("batch.max_seq_length: {}".format(batch.max_seq_length))
        logging.debug("batch.batch_size: {}".format(batch.batch_size))

        # to get action --> [32, 300, 2, 7]
        # [32*301*2, 26] --> [32*301*2, 7] --> [32, 301, 2, 7] --> [32, 300, 2, 7]
        # batch4train = batch_trajectories.reshape([-1, batch_trajectories.shape[-1]])

        self.train_times += 1
        loss_val = self.actor.train(
            batch_trajectories,
            list([max_ep_t for _ in range(batch.batch_size * self.n_agents)]),
            avail_actions,
            actions,
            batch["state"][:, :-1],
            batch["state"][:, 1:],
            rewards,
            terminated,
            mask,
        )

        # update explore agent after each train processing.
        self.actor.assign_explore_agent()

        # update target network as required.
        if (episode_num - self.last_target_update_episode) / self.alg_config["target_update_interval"] >= 1.0:
            self.actor.assign_targets()
            logging.debug("episode-{}, target Q network params replaced!".format(episode_num))
            print("episode-{}, target Q network params replaced!".format(episode_num))
            print(">>> train-{} use seq-len-{}".format(self.train_times, max_ep_t))
            sys.stdout.flush()
            self.last_target_update_episode = episode_num

        return loss_val

    def train_ready(self, elapsed_episode, **kwargs):
        """
        Support custom train logic.

        :return: train ready flag
        """
        # we set train ready as default
        # if elapsed_episode < self.alg_config["batch_size"]:
        if not self.buffer.can_sample(self.alg_config["batch_size"]):
            self._train_ready = False
            if not kwargs.get("dist_dummy_model"):
                raise KeyError("qmix need to dist dummy model.")
            # dist dummy model
            kwargs["dist_dummy_model"]()
        else:
            self._train_ready = True

        return self._train_ready
