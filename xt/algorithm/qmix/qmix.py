"""Implement the qmix algorithm with tensorflow, also thanks to the pymarl repo."""

from functools import partial
from time import time
import numpy as np
from absl import logging
from smac.env import MultiAgentEnv, StarCraft2Env
from xt.algorithm.qmix.episode_buffer_np import EpisodeBatchNP
from xt.algorithm.qmix.qmix_alg import DecayThenFlatSchedule, EpsilonGreedyActionSelector
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf


class QMixAlgorithm(object):
    """Create a Target network for calculating the maximum estimated Q-value in given action a."""

    def __init__(self, scheme, args, avail_action_num, seq_limit, dtype):
        # avail_actions vary with env.map
        self.n_agents = args.n_agents
        self.args = args
        self.dtype = dtype
        self.obs_shape = self._get_input_shape(scheme)
        logging.debug("obs_shape: {}".format(self.obs_shape))
        self.previous_state = None
        self.ph_hidden_states_in = None
        self.hidden_states_out = None

        self.params = None
        self.inputs = None
        self.out_actions = None
        self.avail_action_num = avail_action_num
        # 2s_vs_1sc , use the episode limit as fix shape.
        self.fix_seq_length = seq_limit

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

        # select action
        self.selector = EpsilonGreedyActionSelector(self.args)

        # mix
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        # self.global_state_dims = (1, 120)  # fixme: 2s3z

        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=self.graph)
        self.sess = sess

        self.gru_cell = None
        self.hi_out_val = None
        self.hi_out_val_default = None
        # self.hi_target_out_val = None
        self.grad_update = None  # train op

        self._explore_paras = None  # need update after each train process
        self.last_target_update_episode = 0

        self.ph_obs, self.agent_outs, self.hidden_outs = None, None, None
        self.ph_avail_action, self.ph_actions, self.ph_train_obs = None, None, None
        self.ph_train_obs_len, self.agent_explore_replace_op = None, None
        self.agent_train_replace_op, self.ph_train_states = None, None
        self.ph_train_target_states, self.q_tot, self.target_q_tot = None, None, None
        self.mix_train_replace_op = None
        self.ph_rewards, self.ph_terminated = None, None
        self.loss, self.ph_mask = None, None

    def _get_input_shape(self, scheme):
        """Assemble input shape."""
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_motivate_actions(self, agents_dim, avail_actions, t_env, test_mode=False):
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0
        # random_numbers = th.rand_like(agent_inputs[:, :, 0])
        random_numbers = np.random.rand(agents_dim)
        # pick_random = (random_numbers < self.epsilon).long()
        pick_random = np.array(random_numbers < self.epsilon).astype(np.long)

        # random_actions = Categorical(avail_actions.float()).sample().long()
        avail_action_len = avail_actions.shape[-1]
        avail_norm_to_np = np.array(avail_actions / avail_actions.sum(-1)).astype(np.float)
        random_actions = np.random.multinomial(avail_action_len, avail_norm_to_np).astype(np.long)
        return pick_random, random_actions

    def build_agent_net(
            self,
            inputs_obs,
            seq_max,
            obs_lengths,
            hidden_state_in=None,
    ):
        """Build default init_state for rnn."""
        fc1 = tf.layers.dense(
            inputs=inputs_obs,
            units=self.args.rnn_hidden_dim,
            activation=tf.nn.relu,
        )

        fc1 = tf.transpose(fc1, perm=[0, 2, 1, 3])
        print("\n fc1 before reshape: ", fc1)
        fc1 = tf.reshape(fc1, [-1, seq_max, self.args.rnn_hidden_dim])
        print("fc1 after reshape: ", fc1)

        gru_cell = tf.nn.rnn_cell.GRUCell(
            num_units=self.args.rnn_hidden_dim,  # dtype=self.dtype
        )
        # only record the gru cell once time, to init the hidden value.
        if not self.gru_cell:
            self.gru_cell = gru_cell

        # self.hidden_in_zero = self.gru_cell.zero_state(1, dtype=tf.float32)

        # https://blog.csdn.net/u010223750/article/details/71079036
        # tf.nn.dynamic_rnn
        rnn_output, hidden_state_out = tf.nn.dynamic_rnn(
            gru_cell,
            fc1,
            dtype=self.dtype,
            initial_state=hidden_state_in,
            sequence_length=obs_lengths,
            # sequence_length=[1, ]
        )
        print("rnn raw out: {} ".format(rnn_output))
        rnn_output = tf.reshape(rnn_output, [-1, self.n_agents, seq_max, self.args.rnn_hidden_dim])
        rnn_output = tf.transpose(rnn_output, perm=[0, 2, 1, 3])

        rnn_output = tf.reshape(rnn_output, [-1, self.args.rnn_hidden_dim])

        fc2_outputs = tf.layers.dense(
            inputs=rnn_output,
            units=self.args.n_actions,
            activation=None,
            # activation=tf.nn.relu,
        )

        out_actions = tf.reshape(fc2_outputs, (-1, self.n_agents, self.avail_action_num))
        print("out action: {} \n".format(out_actions))
        return out_actions, hidden_state_out

    def _build_mix_net2(self, agent_qs, states):
        """Build mixer architecture with two hyper embed."""
        hypernet_embed = self.args.hypernet_embed

        def hyper_w1(hyper_w1_input):
            """
            Create hyper_w1.

            input shape (none, state_dim)
            """
            with tf.variable_scope("hyper_w1"):
                hw0 = tf.layers.dense(inputs=hyper_w1_input, units=hypernet_embed, activation=tf.nn.relu)
                hw1 = tf.layers.dense(inputs=hw0, units=self.embed_dim * self.n_agents, activation=None)
                return hw1

        def hyper_w_final(hyper_w_final_input):
            """
            Create hyper_w_final.

            input shape (none, state_dim)
            """
            with tf.variable_scope("hyper_w_final"):
                hw_f0 = tf.layers.dense(
                    inputs=hyper_w_final_input,
                    units=hypernet_embed,
                    activation=tf.nn.relu,
                )
                hw_f1 = tf.layers.dense(inputs=hw_f0, units=self.embed_dim, activation=None)
                return hw_f1

        def hyper_b1(state_input):
            """State dependent bias for hidden layer."""
            with tf.variable_scope("hyper_b1"):
                return tf.layers.dense(inputs=state_input, units=self.embed_dim, activation=None)

        def val(state_input):
            """V(s) instead of a bias for the last layers."""
            with tf.variable_scope("val_for_bias"):
                val0 = tf.layers.dense(inputs=state_input, units=self.embed_dim, activation=tf.nn.relu)
                val2 = tf.layers.dense(inputs=val0, units=1, activation=None)
                return val2

        bs = agent_qs.get_shape().as_list()[0]
        states_reshaped = tf.reshape(states, (-1, self.state_dim))
        agent_qs_reshaped = tf.reshape(agent_qs, (-1, 1, self.n_agents))

        # firstly layer
        w1 = tf.math.abs(hyper_w1(states_reshaped))
        b1 = hyper_b1(states_reshaped)

        w1_reshaped = tf.reshape(w1, (-1, self.n_agents, self.embed_dim))
        b1_reshaped = tf.reshape(b1, (-1, 1, self.embed_dim))

        to_hidden_val = tf.math.add(tf.matmul(agent_qs_reshaped, w1_reshaped), b1_reshaped)
        hidden = tf.nn.elu(to_hidden_val)

        # second layer
        w_final = tf.math.abs(hyper_w_final(states_reshaped))
        w_final_reshaped = tf.reshape(w_final, (-1, self.embed_dim, 1))

        # state-dependent bias
        v = tf.reshape(val(states_reshaped), (-1, 1, 1))

        # compute final output
        y = tf.math.add(tf.matmul(hidden, w_final_reshaped), v)

        # reshape and return
        q_tot = tf.reshape(y, (bs, -1, 1))

        return q_tot

    def _build_action_selector(self, agent_inputs, avail_actions, ph_pick_random, ph_random_actions):
        """Calculate the explore action with numpy out of the graph."""
        masked_q_values = tf.identity(agent_inputs)
        negation_inf_val = tf.ones_like(masked_q_values) * -1e10
        masked_q_values = tf.where(avail_actions < 1e-5, negation_inf_val, masked_q_values)

        picked_actions = ph_pick_random * ph_random_actions + (1 - ph_pick_random) * tf.reduce_max(
            masked_q_values, reduction_indices=[2])
        return picked_actions

    def build_inputs(self, batch, t):
        """
        Build inputs.

        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        1. inference stage, use batch = 1,
        2. train stage, use batch = episode.limit

        Also, use numpy for combine the inputs data
        """
        bs = batch.batch_size
        inputs = list()
        inputs.append(batch["obs"][:, t])  # b1av
        # print("forward input.obs shape, ", np.shape(inputs[0])) # torch.Size([1, 5, 80])
        if self.args.obs_last_action:
            if t == 0:
                # tmp = batch["actions_onehot"][:, t]
                # print(tmp, np.shape(tmp), np.shape(batch["actions_onehot"]))
                inputs.append(np.zeros_like(batch["actions_onehot"][:, t]))
                # print(inputs)
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
            # print("forward input.onehot shape, ",
            #       np.shape(inputs[-1]), np.shape(batch["actions_onehot"]))
        if self.args.obs_agent_id:
            _ag_id = np.expand_dims(np.eye(self.n_agents), axis=0)  # add axis 0
            inputs.append(np.tile(_ag_id, (bs, 1, 1)))  # broadcast_to

        # print("inputs shape: ", [np.shape(i) for i in inputs])
        # inputs = np.concatenate(
        #     [x.reshape(bs * self.n_agents, -1) for x in inputs], axis=1
        # )
        # [batch_size, 1, agents, obs_size]
        inputs = np.expand_dims(np.concatenate(inputs, axis=-1), axis=1)

        # fixme: make to [batch_size, agent_num, seq_len, obs_size]
        # print("forward input shape, ", np.shape(inputs)) # torch.Size([5, 96])
        # print("inputs shape: ", inputs.shape)
        return inputs

    def build_actor_graph(self):
        """Build an actor graph used by the explorer."""
        with self.graph.as_default():
            self.ph_obs = tf.placeholder(tf.float32, shape=(1, 1, self.n_agents, self.obs_shape), name="obs")
            # self.ph_obs_len = tf.placeholder(tf.float32, shape=(None,), name="obs_len")

            self.ph_hidden_states_in = tf.placeholder(tf.float32,
                                                      shape=(None, self.args.rnn_hidden_dim),
                                                      name="hidden_in")

            with tf.variable_scope("explore_agent"):
                self.agent_outs, self.hidden_outs = self.build_agent_net(
                    inputs_obs=self.ph_obs,
                    seq_max=1,  # --------------------- 1, importance
                    obs_lengths=[1 for _ in range(self.n_agents)],
                    hidden_state_in=self.ph_hidden_states_in,
                )

            self._explore_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="explore_agent")

    def reset_hidden_state(self):
        """Reset hidden before start each episode."""
        self.hi_out_val = self.hi_out_val_default

    def get_explore_actions(self, ep_batch, t_ep, t_env, test_mode):
        """Get explore action with numpy."""
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_inputs = self.build_inputs(ep_batch, t_ep)
        # agent_inputs =

        out_val = self.infer_actions(agent_inputs)

        select_actions = self.selector.select_action(out_val, avail_actions, t_env, test_mode=test_mode)
        # print("out_val: {}, select action: {}, avail_actions, {}, t_env:{}".format(
        #     out_val, select_actions, avail_actions, t_env))
        return select_actions

    def infer_actions(self, agent_inputs):
        """Inference with tf.sess.run."""
        out_val, self.hi_out_val = self.sess.run(
            [self.agent_outs, self.hidden_outs],
            feed_dict={
                self.ph_obs: agent_inputs,
                # self.ph_obs_len: list(obs_len),
                self.ph_hidden_states_in: self.hi_out_val,
            },
        )
        return out_val

    @staticmethod
    def _gather4d_on_dim3(inputs, indices):
        """
        Gather 4dim tensor into 3dim, same to the pytorch.gather + sequeeze(3) function.

        :param inputs:
        :param indices:
        :return:
        """
        print("inputs: ", inputs)
        len_0d, len_1d, len_2d, len_3d = inputs.get_shape().as_list()
        print("len_0d, len_1d, len_2d, len_3d", len_0d, len_1d, len_2d, len_3d)
        inputs = tf.reshape(inputs, (-1, len_3d))
        calc_0d = inputs.get_shape()[0]

        flag_0d, flag_1d, flag_2d, flag_3d = indices.get_shape()
        indices = tf.reshape(indices, [-1, flag_3d])

        idx_matrix = tf.tile(tf.expand_dims(tf.range(0, len_3d, dtype=indices.dtype), 0), [calc_0d, 1])
        indices_t = tf.transpose(indices)
        idx_mask = tf.equal(idx_matrix, tf.transpose(indices_t))

        inputs = tf.reshape(tf.boolean_mask(inputs, idx_mask), [flag_0d, flag_1d, flag_2d])
        return inputs

    @staticmethod
    def _print_trainable_var_name(**kwargs):
        """Print trainable variable name."""
        for k, v in kwargs.items():
            print("{}: \n {}".format(k, list([t.name for t in v])))

    def build_train_graph(self):
        """
        Build train graph.

        train graph cannot connect-up to actor.graph,
        because of the different seq_max(1 vs limit)
        """
        with self.graph.as_default():
            self.ph_avail_action = tf.placeholder(
                tf.float32,
                shape=[
                    self.args.batch_size,
                    self.fix_seq_length + 1,
                    self.n_agents,
                    self.avail_action_num,
                ],
                name="avail_action",
            )

            self.ph_actions = tf.placeholder(
                tf.float32,
                shape=[self.args.batch_size, self.fix_seq_length, self.n_agents, 1],
                name="actions",
            )

            # agent_num = self.n_agents
            # seq_max = 300
            # -------eval rnn agent ------------------
            self.ph_train_obs = tf.placeholder(
                tf.float32,
                shape=(
                    self.args.batch_size,
                    self.fix_seq_length + 1,
                    self.n_agents,
                    self.obs_shape,
                ),
                name="train_obs",
            )
            self.ph_train_obs_len = tf.placeholder(tf.float32, shape=(None, ), name="train_obs_len")

            with tf.variable_scope("eval_agent"):
                trajectory_agent_outs, _ = self.build_agent_net(
                    inputs_obs=self.ph_train_obs,
                    seq_max=self.fix_seq_length + 1,  # --------------------- importance
                    obs_lengths=self.ph_train_obs_len,
                    hidden_state_in=None,  # with total trajectory, needn't hold hidden
                )

            with tf.variable_scope("target_agent"):
                tar_agent_outs_tmp, _ = self.build_agent_net(
                    inputs_obs=self.ph_train_obs,
                    # fix value, different between explore and train
                    seq_max=self.fix_seq_length + 1,
                    obs_lengths=self.ph_train_obs_len,
                    hidden_state_in=None,
                )
                target_trajectory_agent_outs = tf.stop_gradient(tar_agent_outs_tmp)

            _eval_agent_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="eval_agent")
            _target_agent_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="target_agent")

            with tf.variable_scope("soft_replacement"):
                self.agent_train_replace_op = [tf.assign(t, e) for t, e in zip(_target_agent_paras, _eval_agent_paras)]

                self.agent_explore_replace_op = [
                    tf.assign(t, e) for t, e in zip(self._explore_paras, _eval_agent_paras)
                ]

            self._print_trainable_var_name(
                _eval_agent_paras=_eval_agent_paras,
                _target_agent_paras=_target_agent_paras,
                _explore_paras=self._explore_paras,
            )

            # agent out to max q values
            # Calculate estimated Q-Values ----------------
            mac_out = tf.reshape(
                trajectory_agent_outs,
                [self.args.batch_size, self.fix_seq_length + 1, self.n_agents, -1],
            )
            print("mac_out: ", mac_out)
            chosen_action_qvals = self._gather4d_on_dim3(mac_out[:, :-1], self.ph_actions)  # -----

            # Calculate the Q-Values necessary for the target -----------

            target_mac_out = tf.reshape(
                target_trajectory_agent_outs,
                [self.args.batch_size, self.fix_seq_length + 1, self.n_agents, -1],
            )
            target_mac_out = target_mac_out[:, 1:]

            # Mask out unavailable actions
            # target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            indices = tf.equal(self.ph_avail_action[:, 1:], 0)
            # TypeError: Input 'e' of 'Select' Op has type float32 that
            # does not match type int32 of argument 't'.
            mask_val = tf.tile(
                [[[[-999999.0]]]],
                [
                    self.args.batch_size,
                    self.fix_seq_length,
                    self.n_agents,
                    self.avail_action_num,
                ],
            )
            print("indices: ", indices)
            print("mask_val: ", mask_val)
            print("target_mac_out: ", target_mac_out)

            target_mac_out = tf.where(indices, mask_val, target_mac_out)
            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = tf.stop_gradient(tf.identity(mac_out[:, 1:]))
                mac_out_detach = tf.where(indices, mask_val, mac_out_detach)
                cur_max_actions = tf.expand_dims(tf.argmax(mac_out_detach, axis=-1), -1)
                target_max_qvals = self._gather4d_on_dim3(target_mac_out, cur_max_actions)

            else:
                target_max_qvals = tf.reduce_max(target_mac_out, axis=[-1])

            # eval mixer ---------------
            self.ph_train_states = tf.placeholder(
                tf.float32,
                shape=(self.args.batch_size, self.fix_seq_length, self.state_dim),
                name="train_stats",
            )
            # target mixer -------------------
            self.ph_train_target_states = tf.placeholder(
                tf.float32,
                shape=(self.args.batch_size, self.fix_seq_length, self.state_dim),
                name="train_target_stats",
            )

            with tf.variable_scope("eval_mixer"):
                self.q_tot = self._build_mix_net2(chosen_action_qvals, self.ph_train_states)

            with tf.variable_scope("target_mixer"):
                q_tot_tmp = self._build_mix_net2(target_max_qvals, self.ph_train_target_states)
                self.target_q_tot = tf.stop_gradient(q_tot_tmp)

            _eval_mix_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="eval_mixer")
            _target_mix_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="target_mixer")

            with tf.variable_scope("soft_replacement"):
                self.mix_train_replace_op = [tf.assign(t, e) for t, e in zip(_target_mix_paras, _eval_mix_paras)]

            self._print_trainable_var_name(_eval_mix_paras=_eval_mix_paras, _target_mix_paras=_target_mix_paras)

            # --------
            self.ph_rewards = tf.placeholder(
                tf.float32,
                shape=(self.args.batch_size, self.fix_seq_length, 1),
                name="rewards",
            )
            self.ph_terminated = tf.placeholder(
                tf.float32,
                shape=(self.args.batch_size, self.fix_seq_length, 1),
                name="terminated",
            )
            self.ph_mask = tf.placeholder(
                tf.float32,
                shape=(self.args.batch_size, self.fix_seq_length, 1),
                name="mask",
            )

            print("self.ph_rewards: ", self.ph_rewards)
            print("self.args.gamma: ", self.args.gamma)
            print("self.ph_terminated: ", self.ph_terminated)
            print("self.target_q_tot: ", self.target_q_tot)

            # Calculate 1-step Q-Learning targets
            targets = (self.ph_rewards + self.args.gamma * (1.0 - self.ph_terminated) * self.target_q_tot)

            # Td-error
            td_error = self.q_tot - tf.stop_gradient(targets)

            # mask = mask.expand_as(td_error)  #fixme: default as same shape!

            # 0-out the targets that came from padded data
            masked_td_error = tf.multiply(td_error, self.ph_mask)

            self.loss = tf.reduce_sum(masked_td_error**2) / tf.reduce_sum(self.ph_mask)

            # # Optimise
            optimizer = tf.train.RMSPropOptimizer(self.args.lr, decay=0.95, epsilon=1.5e-7, centered=True)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            capped_gvs = [(
                grad if grad is None else tf.clip_by_norm(grad, clip_norm=self.args.grad_norm_clip),
                var,
            ) for grad, var in grads_and_vars]
            self.grad_update = optimizer.apply_gradients(capped_gvs)

    def _update_targets(self, episode_num):
        """
        Update weights periodically.

        1. from eval agent to target agent
        2. from target mixer to eval mixer
        :return:
        """
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            _a, _m = self.sess.run([self.agent_train_replace_op, self.mix_train_replace_op])
            print('episode ' + str(episode_num) + ', target Q network params replaced!')
            self.last_target_update_episode = episode_num

    def _update_explore_agent(self):
        """
        Update explore agent after each train process.

        :return:
        """
        _ = self.sess.run(self.agent_explore_replace_op)

    def save_explore_agent_weights(self, save_path):
        """Save explore agent weight for explorer."""
        explore_saver = tf.train.Saver({t.name: t for t in self._explore_paras})
        explore_saver.save(self.sess, save_path=save_path, write_meta_graph=False)
        # tf.train.list_variables(tf.train.latest_checkpoint(wp))

    def train_whole_graph(self, batch: EpisodeBatchNP, t_env: int, episode_num: int):

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

        # # # Calculate estimated Q-Values
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

        # writer = tf.summary.FileWriter(logdir="logdir", graph=self.graph)
        # writer.flush()

        _, loss_val = self.sess.run(
            [self.grad_update, self.loss],
            feed_dict={
                self.ph_train_obs: batch_trajectories,
                # Note: split trajectory with each agent.
                self.ph_train_obs_len: list(
                    [max_ep_t for _ in range(batch.batch_size * self.n_agents)]),
                self.ph_avail_action: avail_actions,
                self.ph_actions: actions,
                self.ph_train_states: batch["state"][:, :-1],
                self.ph_train_target_states: batch["state"][:, 1:],
                self.ph_rewards: rewards,
                self.ph_terminated: terminated,
                self.ph_mask: mask,
            },
        )
        logging.info("episode-{}, t_env-{}, train_loss: {}".format(episode_num, t_env, loss_val))

        # from tests.qmix.test_assign import print_mix_tensor_val, print_agent_tensor_val
        # print_agent_tensor_val(self.graph, self.sess, "before update explore agent")
        self._update_explore_agent()
        self.save_explore_agent_weights(save_path="./save_models/actor{}".format(episode_num))

        # print_agent_tensor_val(self.graph, self.sess, "after update explore agent")
        # print_mix_tensor_val(self.graph, self.sess, "before update target")
        self._update_targets(episode_num=episode_num)
        # print_mix_tensor_val(self.graph, self.sess, "after update target")

        return {"train_loss": loss_val}


class QMixAgent(object):
    """Create agent for 2s_vs_1sc."""

    def __init__(self, scheme, args):
        self.args = args
        self.scheme = scheme

        def env_fn(env, **kwargs) -> MultiAgentEnv:
            return env(**kwargs)

        sc2_env_func = partial(env_fn, env=StarCraft2Env)

        self.env = sc2_env_func(**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        print("limit seq: ", self.episode_limit)
        env_info = self.env.get_env_info()
        print("env_info: ", env_info)
        self.avail_action_num = env_info["n_actions"]
        self.t = 0
        self.t_env = 0
        self.n_episode = 0

        self.alg = QMixAlgorithm(self.scheme, self.args, self.avail_action_num, self.episode_limit, tf.float32)
        self.replay_buffer = None
        self.batch = None

        # self.bm_writer = BenchmarkBoard("logdir", "qmix_{}".format(
        #     strftime("%Y-%m-%d %H-%M-%S", localtime())))

    def setup(self, scheme, groups, preprocess):
        self.new_batch = partial(
            EpisodeBatchNP,
            scheme,
            groups,
            1,  # Note: batch size must be 1 in a episode
            self.episode_limit + 1,
            preprocess=preprocess,
        )
        self.alg.build_actor_graph()  # 1 only use for explore !

        self.alg.build_train_graph()

        # note: init with only once are importance!
        with self.alg.graph.as_default():
            self.alg.sess.run(tf.global_variables_initializer())

            self.alg.hi_out_val_default = self.alg.sess.run(
                self.alg.gru_cell.zero_state(self.args.n_agents, dtype=tf.float32))

        writer = tf.summary.FileWriter(logdir="logdir", graph=self.alg.graph)
        writer.flush()

    def reset(self):
        self.alg.reset_hidden_state()
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run_one_episode(self, test_mode=False):
        # time_info = [0, 0, 0]  # reset, interaction
        _t = time()
        self.reset()
        _reset_t = time() - _t

        terminated = False
        episode_return = 0
        env_step_list = []
        infer_time_list = []
        interaction_cycle, cycle_start = [], None

        def show_time(text, time_list):
            print("{} mean: {},  Hz-~{},  steps-{}, last-7 as: \n {}".format(
                text,
                np.mean(time_list[5:]),
                int(1.0 / np.mean(time_list)),
                len(time_list),
                time_list[-7:],
            ))
            return np.mean(time_list[5:])

        _start_explore = time()
        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this time step in a batch of size 1
            before_infer = time()
            actions = self.alg.get_explore_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            infer_time_list.append(time() - before_infer)

            before_env_step = time()
            reward, terminated, env_info = self.env.step(actions[0])
            env_step_list.append(time() - before_env_step)
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward, )],
                "terminated": [(terminated != env_info.get("episode_limit", False), )],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            if not cycle_start:
                cycle_start = time()
            else:
                interaction_cycle.append(time() - cycle_start)
                cycle_start = time()

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.alg.get_explore_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t
            self.n_episode += 1

        # # for time analysis
        # env_avg = show_time("env_step", env_step_list)
        # infer_avg = show_time("infer time", infer_time_list)
        # cycle_avg = show_time("--> cycle", interaction_cycle)
        # print(
        #     "env step proportion: {}, infer proportion:{}.".format(
        #         env_avg / cycle_avg, infer_avg / cycle_avg
        #     )
        # )
        logging.debug("t_env: {}, explore reward: {}".format(self.t_env, episode_return))
        # print("env_info: ", env_info)

        if env_info.get("battle_won"):
            print("\n", "*" * 50, "won once in {} mode! \n".format("TEST" if test_mode else "EXPLORE"))
            # self.bm_writer.insert_records(
        record_info_list = [("reset_time", _reset_t, self.n_episode),
                            ("interaction_time", time() - _start_explore, self.n_episode),
                            ("env_step_mean", np.mean(env_step_list), self.n_episode),
                            ("infer_mean", np.mean(infer_time_list), self.n_episode),
                            ("cycle_mean", np.mean(interaction_cycle), self.n_episode),
                            ("explore_reward", episode_return, self.t_env),
                            ("step_per_episode", self.t, self.n_episode)]

        return self.batch, record_info_list, env_info

    def train(self, batch_data, t_env, episode_num):
        info = self.alg.train_whole_graph(batch_data, t_env, episode_num)
        record_info = [("train_loss", info["train_loss"], self.t_env)]
        return record_info

        # self.bm_writer.insert_records([("train_loss", info["train_loss"], self.t_env)])
