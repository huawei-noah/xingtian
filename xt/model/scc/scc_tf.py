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
"""
SCC model architecture.
Implemented [SCC](https://arxiv.org/abs/2106.00285) graph with tensorflow.
For the restrict factor on dynamic shape of tensorflow.
The whole graph contains 4 sub-graph:
    1) explore agent
    2) eval agent with map.limit
    3) eval mixer with map.limit
    4) target mixer with map.limit
"""
from __future__ import division, print_function
import random
from absl import logging
import numpy as np
from zeus.common.util.register import Registers
from xt.model.tf_compat import tf
from xt.model.tf_utils import TFVariables


@Registers.model
class SCCModel(object):
    """Define SCC model with tensorflow.graph."""

    def __init__(self, model_info):
        """
        Update default model.parameters with model info.
        owing to the big graph contains five sub-graph, while,
        explorer could work well with the explore.graph,
        Based on the least-cost principle,
        explorer could init the explore.graph;
        and, train process init the train.graph.
        """
        logging.debug("init scc model with:\n{}".format(model_info))
        model_config = model_info.get("model_config", None)
        self.model_config = model_config
        map_name = model_config['map_name']
        agent_group_dict = {'2s3z':[2, 3],'3s5z':[3, 5], '3s5z_vs_3s6z':[3, 5], '1c3s5z':[1, 3, 5], 'MMM2':[1, 2, 7]}
        if map_name in agent_group_dict:
            self.agent_group = agent_group_dict[map_name]
        else:
            self.agent_group = [model_config["n_agents"]]

        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=self.graph)
        self.sess = sess

        # start to fetch parameters
        self.gamma = model_config.get("gamma", 0.99)
        self.c_lr = model_config.get("c_lr", 0.0005)
        self.a_lr = model_config.get("a_lr", 0.0005)
        self.mixer_grad_norm_clip = model_config.get(
            "mixer_grad_norm_clip", 10)
        self.actor_grad_norm_clip = model_config.get(
            "actor_grad_norm_clip", 10)
        self.n_agents = model_config["n_agents"]
        self.rnn_hidden_dim = model_config["rnn_hidden_dim"]
        seq_limit = model_config["episode_limit"]
        self.fix_seq_length = seq_limit  # use the episode limit as fix shape.
        self.n_actions = model_config["n_actions"]
        self.obs_shape = model_config["obs_shape"]
        self.batch_size = model_config["batch_size"]
        self.avail_action_num = model_config["n_actions"]
        self.state_dim = int(np.prod(model_config["state_shape"]))
        self.use_double_q = model_config.get("use_double_q", True)
        self.o_shape = self.obs_shape - self.n_actions - self.n_agents
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
            self.actor_target_values = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length, self.n_agents),
                name="mask",
            )
            self.mixer_state_with_action = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length,
                       (self.o_shape + self.n_actions) * self.n_agents),
                name="mixer_online_s_a",
            )
            self.next_mixer_state_with_action = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.fix_seq_length,
                       (self.o_shape + self.n_actions) * self.n_agents),
                name="mixer_target_s_a",
            )
            self.mixer_loss, self.mixer_grad_update, self.actor_loss, self.actor_grad_update = None, None, None, None
            # graph weights update
            self.agent_explore_replace_op = None
            self.mix_train_replace_op = None

        # init graph
        self.g_type = model_info.get("scene", "explore")
        self.build_actor_graph()  # NOTE: build actor always
        if self.g_type == "train":
            self.build_train_graph()

        # note: init with only once are importance!
        with self.graph.as_default():
            self.actor_var = TFVariables(
                [self.agent_outs, self.hidden_outs], self.sess)
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
        reduce_sum_val = tf.reduce_sum(mul_test, axis=-1)
        return reduce_sum_val

    def _build_mixer(self, mixer_state_with_action):
        bs = mixer_state_with_action.get_shape().as_list()[0]
        dense_unit_number = self.model_config["dense_unit_number"]
        enable_critic_multi_channel = self.model_config["enable_critic_multi_channel"]
        group = self.agent_group
        if not enable_critic_multi_channel:
            layer_1 = tf.layers.dense(
                mixer_state_with_action, units=dense_unit_number, activation=tf.nn.relu)
            layer_2 = tf.layers.dense(
                layer_1, units=dense_unit_number, activation=tf.nn.relu)
            v = tf.layers.dense(layer_2, units=1)
            return v
        else:
            group_num = len(group)
            layer_1 = [tf.keras.layers.Dense(
                units=dense_unit_number, activation=tf.nn.relu) for _ in range(len(group))]
            layer_2 = [tf.keras.layers.Dense(
                units=dense_unit_number, activation=tf.nn.relu) for _ in range(len(group))]
            reshaped_s_a = tf.reshape(
                mixer_state_with_action, (bs, -1, self.n_agents, (self.o_shape + self.n_actions)))
            agent_s_a = [reshaped_s_a[:, :, i, :]
                         for i in range(self.n_agents)]  

            group_hs = []
            for j in range(group_num):
                for i in range(sum(group[0:j]), sum(group[0:(j + 1)])):
                    group_hs.append(layer_2[j](layer_1[j](agent_s_a[i])))
            if self.model_config["channel_merge"] == 'concat':
                # bs,t,dim
                hs = tf.concat(group_hs, 2)
            elif self.model_config["channel_merge"] == 'add':
                hs = tf.add_n(group_hs)
            else:
                raise RuntimeError('Channel merge method is not correct')
            v = tf.layers.dense(hs, units=1)
            return v                                      

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
            _eval_agent_paras = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_agent")

            with tf.variable_scope("soft_replacement"):
                self.agent_explore_replace_op = [
                    tf.assign(t, e) for t, e in zip(self._explore_paras,
                                                    _eval_agent_paras)
                ]
            self._print_trainable_var_name(
                _eval_agent_paras=_eval_agent_paras,
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
            # Mask out unavailable actions
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
            logging.debug("indices:{}, mask_val:{}".format(
                indices, mask_val))
            if self.use_double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = tf.stop_gradient(tf.identity(mac_out[:, 1:]))
                mac_out_detach = tf.where(indices, mask_val, mac_out_detach)
                cur_max_actions = tf.expand_dims(
                    tf.argmax(mac_out_detach, axis=-1), -1)            
            else:
                raise RuntimeError('double q is needed')

            with tf.variable_scope("eval_mixer"):
                self.q_tot = self._build_mixer(self.mixer_state_with_action)
            with tf.variable_scope("target_mixer"):
                q_tot_tmp = self._build_mixer(
                    self.next_mixer_state_with_action)
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
            self.mixer_loss = tf.reduce_sum(
                masked_td_error**2) / tf.reduce_sum(self.ph_mask)
            self.onehot_action = tf.one_hot(tf.to_int32(tf.squeeze(
                self.ph_actions)), depth=self.n_actions)  
            alive_mask = tf.tile(self.ph_mask, [1, 1, self.n_agents])
            target_values = tf.reshape(
                self.actor_target_values, (self.batch_size, -1, self.n_agents)) * alive_mask
            self.act_q = tf.reshape(
                chosen_action_qvals, (self.batch_size, -1, self.n_agents)) * alive_mask
            self.sd_op = tf.squared_difference(self.act_q, target_values)
            self.actor_loss = tf.reduce_sum(
                self.sd_op) / tf.reduce_sum(alive_mask)
            # Optimise
            mixer_optimizer = tf.train.AdamOptimizer(self.c_lr)
            actor_optimizer = tf.train.RMSPropOptimizer(self.a_lr)
            if self.actor_grad_norm_clip > 0:
                mixer_grads_and_vars = mixer_optimizer.compute_gradients(
                    self.mixer_loss, _eval_mix_paras)
                actor_grads_and_vars = actor_optimizer.compute_gradients(
                    self.actor_loss, _eval_agent_paras)
                mixer_capped_gvs = [(
                    grad if grad is None else tf.clip_by_norm(
                        grad, clip_norm=self.mixer_grad_norm_clip),
                    var,
                ) for grad, var in mixer_grads_and_vars]
                actor_capped_gvs = [(
                    grad if grad is None else tf.clip_by_norm(
                        grad, clip_norm=self.actor_grad_norm_clip),
                    var,
                ) for grad, var in actor_grads_and_vars]
                self.mixer_grad_update = mixer_optimizer.apply_gradients(
                    mixer_capped_gvs)
                self.actor_grad_update = actor_optimizer.apply_gradients(
                    actor_capped_gvs)
            else:
                agrads = tf.gradients(self.actor_loss, _eval_agent_paras)
                cgrads = tf.gradients(self.mixer_loss, _eval_mix_paras)
                agrads = list(zip(agrads, _eval_agent_paras))
                cgrads = list(zip(cgrads, _eval_mix_paras))
                self.mixer_grad_update = mixer_optimizer.apply_gradients(
                    cgrads)
                self.actor_grad_update = actor_optimizer.apply_gradients(
                    agrads)

    def assign_targets(self):
        """
        Update weights periodically.

        from target mixer to eval mixer
        :return:
        """
        _m = self.sess.run([self.mix_train_replace_op])

    def assign_explore_agent(self):
        """
        Update explore agent after each train process.

        :return:
        """
        _ = self.sess.run(self.agent_explore_replace_op)

    def save_explore_agent_weights(self, save_path):
        """Save explore agent weight for explorer."""
        self.explore_saver.save(
            self.sess, save_path=save_path, write_meta_graph=False)

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
            logging.debug(
                "read variable-{} from file:{}".format(n, model_name))
        with self.sess.as_default():  # must been sess
            for var_key in self._explore_paras:
                try:
                    var_key.load(result[var_key.name])
                    logging.debug("load {} success".format(var_key.name))
                except BaseException as err:
                    raise KeyError(
                        "update {} error:{}".format(var_key.name, err))

    def get_mixer_output(self, critic_state):
        mixer_ouput = self.sess.run(self.q_tot, feed_dict={
                                    self.mixer_state_with_action: critic_state})
        return mixer_ouput

    def train(
            self,
            batch_trajectories,
            obs,
            train_obs_len,
            avail_actions,
            actions,
            cur_stats,
            target_stats,
            rewards,
            terminated,
            mask,
    ):
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

        bs = actions.shape[0]
        t_actions = actions.reshape(-1)
        one_hot_targets = np.eye(self.n_actions)[t_actions]
        ph_onehot_action = np.reshape(
            one_hot_targets, (bs, -1, self.n_agents, self.n_actions))
        mixer_state_with_action = np.concatenate(
            (obs[:, :-1], ph_onehot_action), -1)
        mixer_state_with_action = np.reshape(mixer_state_with_action,  (self.batch_size, -1, self.n_agents*(
            self.o_shape+self.n_actions)))  # bs, t ,n_agents* n_actions+obs_dim
        next_mixer_state_with_action = mixer_state_with_action
        next_mixer_state_with_action[:, :-
                                     1] = next_mixer_state_with_action[:, 1:]

        if self.n_agents > 2:
            target_q_val = self.get_ex_according_to_mcshap_mask(
                mixer_state_with_action, self.n_agents, self.o_shape, self.n_actions)
        else:
            target_q_val = self.get_ex_according_to_mask(
                mixer_state_with_action, self.n_agents, self.o_shape, self.n_actions)

        target_q_val = np.reshape(target_q_val, (bs, -1, self.n_agents))
        mixer_loss_val = self.train_mixer(batch_trajectories, train_obs_len, avail_actions, actions, cur_stats,
                                          target_stats, rewards, terminated, mask, mixer_state_with_action, next_mixer_state_with_action)
        actor_loss_val = self.train_policy(
            batch_trajectories, train_obs_len, avail_actions, actions, rewards, terminated, mask, target_q_val)
        loss = actor_loss_val + mixer_loss_val
        logging.debug("mixer_train_loss: {}".format(mixer_loss_val))
        logging.debug("actor_train_loss: {}".format(actor_loss_val))

        return loss

    def train_mixer(
            self,
            batch_trajectories,
            train_obs_len,
            avail_actions,
            actions,
            cur_stats,
            target_stats,
            rewards,
            terminated,
            mask,
            mixer_state_with_action,
            next_mixer_state_with_action,
    ):
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

        _, mixer_loss_val = self.sess.run(
            [self.mixer_grad_update, self.mixer_loss],
            feed_dict={
                self.ph_avail_action: avail_actions,
                self.ph_actions: actions,
                self.ph_rewards: rewards,
                self.ph_terminated: terminated,
                self.ph_mask: mask,
                self.mixer_state_with_action: mixer_state_with_action,
                self.next_mixer_state_with_action: next_mixer_state_with_action,
            },
        )
        logging.debug("mixer_train_loss: {}".format(mixer_loss_val))

        return mixer_loss_val

    def train_policy(
            self,
            batch_trajectories,
            train_obs_len,
            avail_actions,
            actions,
            rewards,
            terminated,
            mask,
            target_q_val
    ):
        """
        Train with the whole graph.

        Update explorer.graph after each train process, and target as required.

        :param batch_trajectories:
        :param train_obs_len: list([max_ep for _ in range(batch.batch_size * n_agents)]
        :param avail_actions: avail action from environment
        :param actions: actual actions within trajectory
        :param rewards:
        :param terminated:
        :param mask:
        :return:
        """

        _, actor_loss_val = self.sess.run(
            [self.actor_grad_update, self.actor_loss],
            feed_dict={
                self.ph_train_obs: batch_trajectories,
                # Note: split trajectory with each agent.
                self.ph_train_obs_len: train_obs_len,
                self.ph_avail_action: avail_actions,
                self.ph_actions: actions,
                self.ph_terminated: terminated,
                self.ph_mask: mask,
                self.actor_target_values: target_q_val,

            },
        )
        logging.debug("actor_train_loss: {}".format(actor_loss_val))

        return actor_loss_val

    # get Counterfactual Credits via Monte Carlo sampling, efficient when n_agents>3
    def get_ex_according_to_mcshap_mask(self, ep_critic_state, n_agents, n_obs, n_actions):
        ep_critic_state = np.array(ep_critic_state) 
        mc_times = self.model_config["mc_sample_times"]
        shapley_agents = []
        for i in range(n_agents):
            shapley_list = []
            for j in range(mc_times):
                agents_no = [x for x in range(n_agents)]
                agents_no.remove(i)
                sample_num = random.randint(1, n_agents-1)
                agents_no = random.sample(agents_no, sample_num)

                mask_with_i = np.ones_like(ep_critic_state)
                mask_without_i = np.ones_like(ep_critic_state)
                for ag in agents_no:
                    mask_with_i[:, :, ag * (n_obs + n_actions) +
                                n_obs: (ag + 1) * (n_obs + n_actions)] = 0
                    mask_without_i[:, :, ag * (n_obs + n_actions) +
                                   n_obs: (ag + 1) * (n_obs + n_actions)] = 0
                mask_without_i[:, :, i * (n_obs + n_actions) +
                               n_obs: (i + 1) * (n_obs + n_actions)] = 0

                v_with_i = self.get_mixer_output(
                    mask_with_i * ep_critic_state)  
                v_without_i = self.get_mixer_output(
                    mask_without_i * ep_critic_state)  
                marginal_i_in_combine = v_with_i - v_without_i
                shapley_list.append(marginal_i_in_combine)
            shapley_i = np.mean(np.stack(shapley_list, 0), 0) 
            shapley_agents.append(shapley_i) 
        ex = np.stack(shapley_agents, 2)   

        return ex

    # get credits via counterfactual mask    
    def get_ex_according_to_mask(self, ep_critic_state, n_agents, n_obs, n_actions):
        ep_critic_state = np.array(ep_critic_state)  
        credit_agents = []
        for i in range(n_agents):
            mask_i = np.ones_like(ep_critic_state)
            mask_i[:, :, i * (n_obs + n_actions): (i + 1)
                   * (n_obs + n_actions)] = 0
            v_with_i = self.get_mixer_output(ep_critic_state) 
            v_without_i = self.get_mixer_output(
                mask_i * ep_critic_state)  
            credit_i = v_with_i - v_without_i  
            credit_agents.append(credit_i)  
        ex = np.stack(credit_agents, 2)   

        return ex
