#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defined the DQN network for information flow.
"""

import numpy as np

from xt.model import XTModel
from xt.model.tf_compat import (
    GRU,
    Adam,
    Dense,
    Embedding,
    Flatten,
    Input,
    K,
    Model,
    Reshape,
    concatenate,
    tf,
)
from xt.model.tf_utils import TFVariables
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from absl import logging


@Registers.model
class DqnInfoFlowModel(XTModel):
    """DQN Class for information flow."""

    def __init__(self, model_info):
        """Init Dqn model for information flow."""
        model_config = model_info.get("model_config", None)
        import_config(globals(), model_config)

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]

        self.tau = 0.01
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.vocab_size = model_info["vocab_size"]
        self.emb_dim = model_info["emb_dim"]
        self.user_dim = model_info["user_dim"]
        self.item_dim = model_info["item_dim"]

        self.input_type = model_info["input_type"]
        # logging.info("set input type: {}".format(self.input_type))

        self.embeddings = model_info["embeddings"]
        self.last_act = model_info["last_activate"]

        embedding_weights = np.loadtxt(self.embeddings, delimiter=",", dtype=float)
        self.embedding_initializer = tf.constant_initializer(embedding_weights)

        self.n_history_click = 5
        self.n_history_no_click = 5

        super().__init__(model_info)

    def create_model(self, model_info):
        """Create Deep-Q network."""

        user_input = Input(shape=(self.user_dim,), name="user_input", dtype=self.input_type)
        history_click_input = Input(
            shape=(self.n_history_click * self.item_dim), name="history_click",
            dtype=self.input_type
        )
        history_no_click_input = Input(
            shape=(self.n_history_no_click * self.item_dim), name="history_no_click",
            dtype=self.input_type
        )
        item_input = Input(shape=(self.item_dim,), name="item_input", dtype=self.input_type)
        shared_embedding = Embedding(
            self.vocab_size,
            self.emb_dim,
            name="Emb",
            mask_zero=True,
            embeddings_initializer=self.embedding_initializer,
            trainable=False,
        )  # un-trainable
        gru_click = GRU(self.item_dim * self.emb_dim)
        gru_no_click = GRU(self.item_dim * self.emb_dim)

        user_feature = Flatten()(shared_embedding(user_input))
        item_feature = Flatten()(shared_embedding(item_input))

        history_click_feature = Reshape(
            (self.n_history_click, self.item_dim * self.emb_dim)
        )(shared_embedding(history_click_input))
        history_click_feature = gru_click(history_click_feature)

        history_no_click_feature = Reshape(
            (self.n_history_no_click, self.item_dim * self.emb_dim)
        )(shared_embedding(history_no_click_input))
        history_no_click_feature = gru_no_click(history_no_click_feature)

        x = concatenate(
            [
                user_feature,
                history_click_feature,
                history_no_click_feature,
                item_feature,
            ]
        )
        x_dense1 = Dense(128, activation="relu")(x)
        x_dense2 = Dense(128, activation="relu")(x_dense1)
        # ctr_pred = Dense(1, activation="linear", name="q_value")(x_dense2)
        ctr_pred = Dense(1, activation=self.last_act, name="q_value")(x_dense2)
        model = Model(
            inputs=[
                user_input,
                history_click_input,
                history_no_click_input,
                item_input,
            ],
            outputs=ctr_pred,
        )
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        if self._summary:
            model.summary()

        self.user_input = tf.placeholder(
            dtype=self.input_type, name="user_input", shape=(None, self.user_dim)
        )
        self.history_click_input = tf.placeholder(
            dtype=self.input_type,
            name="history_click_input",
            shape=(None, self.n_history_click * self.item_dim),
        )
        self.history_no_click_input = tf.placeholder(
            dtype=self.input_type,
            name="history_no_click_input",
            shape=(None, self.n_history_no_click * self.item_dim),
        )
        self.item_input = tf.placeholder(
            dtype=self.input_type, name="item_input", shape=(None, self.item_dim)
        )

        self.ctr_predict = model(
            [
                self.user_input,
                self.history_click_input,
                self.history_no_click_input,
                self.item_input,
            ]
        )
        self.actor_var = TFVariables([self.ctr_predict], self.sess)

        self.sess.run(tf.initialize_all_variables())
        return model

    def train(self, state, label, batch_size, verbose=False):
        """Train the model."""
        with self.graph.as_default():
            K.set_session(self.sess)
            history = self.model.fit(
                state, label, batch_size=batch_size, verbose=verbose
            )
            return history.history["loss"][0]

    def predict(self, state):
        """
        Do predict use the newest model.

        :param state:
        :return:
        """
        with self.graph.as_default():
            # K.set_session(self.sess)
            # return np.array(self.model.predict_on_batch(state)).reshape(-1)
            feed_dict = {
                self.user_input: state["user_input"],
                self.history_click_input: state["history_click"],
                self.history_no_click_input: state["history_no_click"],
                self.item_input: state["item_input"],
            }
            return np.array(self.sess.run(self.ctr_predict, feed_dict)).reshape(-1)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        # split keras and xingtian npz
        with self.graph.as_default():
            K.set_session(self.sess)
            if isinstance(weights, dict) and self.actor_var:
                self.actor_var.set_weights(weights)
            else:  # keras
                self.model.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        with self.graph.as_default():
            K.set_session(self.sess)
            return self.model.get_weights()

    def load_model(self, model_name):
        if self.actor_var and str(model_name).endswith(".npz"):
            self.actor_var.set_weights_with_npz(model_name)
        else:
            with self.graph.as_default():
                K.set_session(self.sess)
                self.model.load_weights(model_name)
                # self.model.load_weights(model_name, by_name=True)
