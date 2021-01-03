"""
Qmix model architecture.

Implemented [Qmix](https://arxiv.org/pdf/1803.11485.pdf) graph with tensorflow.
For the restrict factor on dynamic shape of tensorflow.
The whole graph contains 5 sub-graph:
    1) explore agent
    2) eval agent with map.limit
    3) target agent with map.limit
    4) eval mixer with map.limit
    5) target mixer with map.limit
"""
from __future__ import division, print_function

import numpy as np

from xt.model.tf_utils import TFVariables
from absl import logging
from zeus.common.util.register import Registers
from xt.model.tf_compat import tf


@Registers.model
class QMixModel(object):
    """Define QMix model with tensorflow.graph."""

    def __init__(self, model_info):
        """
        Update default model.parameters with model info.

        owing to the big graph contains five sub-graph, while,
        explorer could work well with the explore.graph,
        Based on the least-cost principle,
        explorer could init the explore.graph;
        and, train process init the train.graph.
        """
        logging.debug("init qmix model with:\n{}".format(model_info))
        model_config = model_info.get("model_config", None)

        self.model_config = model_config

        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=self.graph)
        self.sess = sess

        # start to fetch parameters
        self.gamma = model_config.get("gamma", 0.99)
        self.lr = model_config.get("lr", 0.0005)
        self.grad_norm_clip = model_config.get("grad_norm_clip", 10)

        self.n_agents = model_config["n_agents"]
        self.obs_shape = model_config["obs_shape"]
        self.rnn_hidden_dim = model_config["rnn_hidden_dim"]

        seq_limit = model_config["episode_limit"]
        self.fix_seq_length = seq_limit  # use the episode limit as fix shape.

        self.n_actions = model_config["n_actions"]

        self.batch_size = model_config["batch_size"]
        self.avail_action_num = model_config["n_actions"]
        self.state_dim = int(np.prod(model_config["state_shape"]))
        self.embed_dim = model_config["mixing_embed_dim"]

        self.use_double_q = model_config.get("use_double_q", True)
        # fetch parameters from configure ready

        with self.graph.as_default():
            # placeholder work with tf.sess.run
            # buffer for explore
            # note: 4-d make same significance with train operation !
            self.ph_obs = tf.placeholder(
                tf.float32, shape=(1, 1, self.n_agents, self.obs_shape), name="obs")

            self.ph_hidden_states_in = tf.placeholder(
                tf.float32, shape=(None, self.rnn_hidden_dim), name="hidden_in")
            self.agent_outs, self.hidden_outs = None, None
            self._explore_paras = None
            self.gru_cell = None
            self.hi_out_val = None

            # placeholder for train
            self.ph_avail_action = tf.placeholder(
                tf.float32,
                shape=[
                    self.batch_size,
                    self.fix_seq_length + 1,
                    self.n_agents,
                    self.avail_action_num,
                ],
                name="avail_action",
            )

            self.ph_actions = tf.placeholder(
                tf.float32,
                shape=[self.batch_size, self.fix_seq_length, self.n_agents, 1],
                name="actions",
            )

            self.ph_train_obs = tf.placeholder(
                tf.float32,
                shape=(
                    self.batch_size,
                    self.fix_seq_length + 1,
                    self.n_agents,
                    self.obs_shape,
                ),
                name="train_obs",
            )
            self.ph_train_obs_len = tf.placeholder(
                tf.float32, shape=(None, ), name="train_obs_len")

            # eval mixer ---------------
            self.ph_train_states = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length, self.state_dim),
                name="train_stats",
            )
            # target mixer -------------------
            self.ph_train_target_states = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length, self.state_dim),
                name="train_target_stats",
            )

            self.q_tot, self.target_q_tot = None, None

            self.ph_rewards = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length, 1),
                name="rewards",
            )
            self.ph_terminated = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length, 1),
                name="terminated",
            )
            self.ph_mask = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length, 1),
                name="mask",
            )

            self.loss, self.grad_update = None, None

            # graph weights update
            self.agent_train_replace_op = None
            self.agent_explore_replace_op = None
            self.mix_train_replace_op = None

        # init graph
        self.g_type = model_info.get("scene", "explore")

        self.build_actor_graph()  # NOTE: build actor always
        if self.g_type == "train":
            self.build_train_graph()

        # note: init with only once are importance!
        with self.graph.as_default():
            self.actor_var = TFVariables([self.agent_outs, self.hidden_outs], self.sess)

            self.sess.run(tf.global_variables_initializer())
            self.hi_out_val_default = self.sess.run(
                self.gru_cell.zero_state(self.n_agents, dtype=tf.float32))

            # max_to_keep = 5 default, may been remove when to evaluate
            self.explore_saver = tf.train.Saver({
                t.name: t for t in self._explore_paras}, max_to_keep=100,)

    def build_actor_graph(self):
        """Build explorer graph with minimum principle."""
        with self.graph.as_default():
            with tf.variable_scope("explore_agent"):
                self.agent_outs, self.hidden_outs = self.build_agent_net(
                    inputs_obs=self.ph_obs,
                    seq_max=1,  # 1, importance for inference
                    obs_lengths=[1 for _ in range(self.n_agents)],
                    hidden_state_in=self.ph_hidden_states_in,
                )

            self._explore_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="explore_agent")

    def build_agent_net(self, inputs_obs, seq_max, obs_lengths, hidden_state_in):
        """
        Build agent architecture.

        could work well among explorer & train with different sequence.
        """
        fc1 = tf.layers.dense(
            inputs=inputs_obs,
            units=self.rnn_hidden_dim,
            activation=tf.nn.relu,
        )

        fc1 = tf.transpose(fc1, perm=[0, 2, 1, 3])
        logging.debug("fc1 before reshape: {}".format(fc1))
        fc1 = tf.reshape(fc1, [-1, seq_max, self.rnn_hidden_dim])
        logging.debug("fc1 after reshape: {}".format(fc1))

        gru_cell = tf.nn.rnn_cell.GRUCell(
            num_units=self.rnn_hidden_dim,  # dtype=self.dtype
        )
        # only record the gru cell once time, to init the hidden value.
        if not self.gru_cell:
            self.gru_cell = gru_cell

        # tf.nn.dynamic_rnn could be work well with different-length sequence
        rnn_output, hidden_state_out = tf.nn.dynamic_rnn(
            gru_cell,
            fc1,
            dtype=tf.float32,
            initial_state=hidden_state_in,
            sequence_length=obs_lengths,
        )

        logging.debug("rnn raw out: {} ".format(rnn_output))
        rnn_output = tf.reshape(
            rnn_output, [-1, self.n_agents, seq_max, self.rnn_hidden_dim])
        rnn_output = tf.transpose(rnn_output, perm=[0, 2, 1, 3])

        rnn_output = tf.reshape(rnn_output, [-1, self.rnn_hidden_dim])

        fc2_outputs = tf.layers.dense(
            inputs=rnn_output,
            units=self.n_actions,
            activation=None,
        )

        out_actions = tf.reshape(
            fc2_outputs, (-1, self.n_agents, self.avail_action_num))
        logging.debug("out action: {}".format(out_actions))

        return out_actions, hidden_state_out

    def reset_hidden_state(self):
        """Reset hidden state with value assign."""
        self.hi_out_val = self.hi_out_val_default

    def infer_actions(self, agent_inputs):
        """Unify inference api."""
        out_val, self.hi_out_val = self.sess.run(
            [self.agent_outs, self.hidden_outs],
            feed_dict={
                self.ph_obs: agent_inputs,
                self.ph_hidden_states_in: self.hi_out_val,
            },
        )
        return out_val

    def gather_custom(self, inputs, indices):
        indices = tf.cast(indices, tf.uint8)
        one_hot = tf.squeeze(
            tf.one_hot(indices=indices, depth=self.n_actions, on_value=1.,
                       off_value=0., axis=-1, dtype=tf.float32),
            axis=-2)
        mul_test = tf.multiply(inputs, one_hot)
        # reduce_sum_val = tf.reduce_sum(mul_test, axis=-1, keep_dims=True)
        reduce_sum_val = tf.reduce_sum(mul_test, axis=-1)
        return reduce_sum_val

    def _build_mix_net2(self, agent_qs, states):
        hypernet_embed = self.model_config["hypernet_embed"]

        def hyper_w1(hyper_w1_input):
            """
            Create hyper_w1.

            input shape (none, state_dim)
            """
            with tf.variable_scope("hyper_w1"):
                hw0 = tf.layers.dense(inputs=hyper_w1_input,
                                      units=hypernet_embed,
                                      activation=tf.nn.relu)
                hw1 = tf.layers.dense(inputs=hw0,
                                      units=self.embed_dim * self.n_agents,
                                      activation=None)
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
                hw_f1 = tf.layers.dense(inputs=hw_f0,
                                        units=self.embed_dim,
                                        activation=None)
                return hw_f1

        def hyper_b1(state_input):
            """State dependent bias for hidden layer."""
            with tf.variable_scope("hyper_b1"):
                return tf.layers.dense(inputs=state_input,
                                       units=self.embed_dim,
                                       activation=None)

        def val(state_input):
            """V(s) instead of a bias for the last layers."""
            with tf.variable_scope("val_for_bias"):
                val0 = tf.layers.dense(inputs=state_input,
                                       units=self.embed_dim,
                                       activation=tf.nn.relu)
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

        to_hidden_val = tf.math.add(
            tf.matmul(agent_qs_reshaped, w1_reshaped), b1_reshaped)
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

    @staticmethod
    def _print_trainable_var_name(**kwargs):
        """Print trainable variable name."""
        for k, v in kwargs.items():
            logging.info("{}: \n {}".format(k, list([t.name for t in v])))

    def build_train_graph(self):
        """
        Build train graph.

        Because of the different seq_max(1 vs limit),
        train graph cannot connect-up to actor.graph directly.
        Hence, we build an explore sub-graph and train sub-graph,
        which sync with tf.assign between two collections.
        :return:
        """
        with self.graph.as_default():
            with tf.variable_scope("eval_agent"):
                trajectory_agent_outs, _ = self.build_agent_net(
                    inputs_obs=self.ph_train_obs,
                    seq_max=self.fix_seq_length + 1,  # importance
                    obs_lengths=self.ph_train_obs_len,
                    hidden_state_in=None,  # total trajectory, needn't hold hidden
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
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_agent")
            _target_agent_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_agent")

            with tf.variable_scope("soft_replacement"):
                self.agent_train_replace_op = [
                    tf.assign(t, e) for t, e in zip(_target_agent_paras,
                                                    _eval_agent_paras)]

                self.agent_explore_replace_op = [
                    tf.assign(t, e) for t, e in zip(self._explore_paras,
                                                    _eval_agent_paras)
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
                [self.batch_size, self.fix_seq_length + 1, self.n_agents, -1],
            )
            logging.debug("mac_out: {}".format(mac_out))
            chosen_action_qvals = self.gather_custom(mac_out[:, :-1],
                                                     self.ph_actions)

            # Calculate the Q-Values necessary for the target -----------
            target_mac_out = tf.reshape(
                target_trajectory_agent_outs,
                [self.batch_size, self.fix_seq_length + 1, self.n_agents, -1],
            )
            target_mac_out = target_mac_out[:, 1:]

            # Mask out unavailable actions
            # target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            indices = tf.equal(self.ph_avail_action[:, 1:], 0)
            mask_val = tf.tile(
                [[[[-999999.0]]]],
                [
                    self.batch_size,
                    self.fix_seq_length,
                    self.n_agents,
                    self.avail_action_num,
                ],
            )
            logging.debug("indices:{}, mask_val:{}, target mac out:{}".format(
                indices, mask_val, target_mac_out))

            target_mac_out = tf.where(indices, mask_val, target_mac_out)

            if self.use_double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = tf.stop_gradient(tf.identity(mac_out[:, 1:]))
                mac_out_detach = tf.where(indices, mask_val, mac_out_detach)
                cur_max_actions = tf.expand_dims(
                    tf.argmax(mac_out_detach, axis=-1), -1)
                target_max_qvals = self.gather_custom(target_mac_out,
                                                      cur_max_actions)
            else:
                target_max_qvals = tf.reduce_max(target_mac_out, axis=[-1])

            # eval mixer ---------------
            with tf.variable_scope("eval_mixer"):
                self.q_tot = self._build_mix_net2(chosen_action_qvals,
                                                  self.ph_train_states)

            with tf.variable_scope("target_mixer"):
                q_tot_tmp = self._build_mix_net2(target_max_qvals,
                                                 self.ph_train_target_states)
                self.target_q_tot = tf.stop_gradient(q_tot_tmp)

            _eval_mix_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_mixer")
            _target_mix_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_mixer")

            with tf.variable_scope("soft_replacement"):
                self.mix_train_replace_op = [
                    tf.assign(t, e) for t, e in zip(_target_mix_paras,
                                                    _eval_mix_paras)]

            self._print_trainable_var_name(_eval_mix_paras=_eval_mix_paras,
                                           _target_mix_paras=_target_mix_paras)

            # Calculate 1-step Q-Learning targets
            targets = (self.ph_rewards +
                       self.gamma * (1.0 - self.ph_terminated) * self.target_q_tot)

            # Td-error
            td_error = self.q_tot - tf.stop_gradient(targets)

            # mask = mask.expand_as(td_error)  #fixme: default as same shape!

            # 0-out the targets that came from padded data
            masked_td_error = tf.multiply(td_error, self.ph_mask)

            self.loss = tf.reduce_sum(masked_td_error**2) / tf.reduce_sum(self.ph_mask)

            # Optimise
            optimizer = tf.train.RMSPropOptimizer(
                self.lr, decay=0.95, epsilon=1.5e-7, centered=True)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            capped_gvs = [(
                grad if grad is None else tf.clip_by_norm(
                    grad, clip_norm=self.grad_norm_clip),
                var,
            ) for grad, var in grads_and_vars]
            self.grad_update = optimizer.apply_gradients(capped_gvs)

    def assign_targets(self):
        """
        Update weights periodically.

        1. from eval agent to target agent
        2. from target mixer to eval mixer
        :return:
        """
        _a, _m = self.sess.run([self.agent_train_replace_op,
                                self.mix_train_replace_op])

    def assign_explore_agent(self):
        """
        Update explore agent after each train process.

        :return:
        """
        _ = self.sess.run(self.agent_explore_replace_op)

    def save_explore_agent_weights(self, save_path):
        """Save explore agent weight for explorer."""
        # explore_saver = tf.train.Saver({t.name: t for t in self._explore_paras})
        self.explore_saver.save(
            self.sess, save_path=save_path, write_meta_graph=False)
        # tf.train.list_variables(tf.train.latest_checkpoint(wp))

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        with self.graph.as_default():
            self.actor_var.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        with self.graph.as_default():
            return self.actor_var.get_weights()

    def restore_explorer_variable(self, model_name):
        """Restore explorer variable with tf.train.checkpoint."""
        reader = tf.train.NewCheckpointReader(model_name)
        var_names = reader.get_variable_to_shape_map().keys()
        result = {}
        for n in var_names:
            result[n] = reader.get_tensor(n)
            logging.debug("read variable-{} from file:{}".format(n, model_name))
        with self.sess.as_default():  # must been sess
            for var_key in self._explore_paras:
                try:
                    var_key.load(result[var_key.name])
                    logging.debug("load {} success".format(var_key.name))
                except BaseException as err:
                    raise KeyError("update {} error:{}".format(var_key.name, err))

    def train(
            self,
            batch_trajectories,
            train_obs_len,
            avail_actions,
            actions,
            cur_stats,
            target_stats,
            rewards,
            terminated,
            mask):
        """
        Train with the whole graph.

        Update explorer.graph after each train process, and target as required.

        :param batch_trajectories:
        :param train_obs_len: list([max_ep for _ in range(batch.batch_size * n_agents)]
        :param avail_actions: avail action from environment
        :param actions: actual actions within trajectory
        :param cur_stats: batch["state"][:, :-1]
        :param target_stats: batch["state"][:, 1:]
        :param rewards:
        :param terminated:
        :param mask:
        :return:
        """
        _, loss_val = self.sess.run(
            [self.grad_update, self.loss],
            feed_dict={
                self.ph_train_obs: batch_trajectories,
                # Note: split trajectory with each agent.
                self.ph_train_obs_len: train_obs_len,
                self.ph_avail_action: avail_actions,
                self.ph_actions: actions,
                self.ph_train_states: cur_stats,
                self.ph_train_target_states: target_stats,
                self.ph_rewards: rewards,
                self.ph_terminated: terminated,
                self.ph_mask: mask,
            },
        )
        logging.debug("train_loss: {}".format(loss_val))
        return loss_val
