#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the calculate of Vtrace.

Follows codes refer to
`https://github.com/deepmind/scalable_agent/blob/master/vtrace_test.py`

"""

import unittest

import numpy as np
import tensorflow as tf
from parameterized import parameterized

from xt.model.impala import vtrace


def fill_shaped_arange(*shape):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape) / 10.0


def calc_softmax(logits):
    elogits = np.exp(logits)
    print("sum.shape: ", np.sum(elogits, axis=-1, keepdims=True).shape)
    return elogits / np.sum(elogits, axis=-1, keepdims=True)


def calc_log_prob_from_logits_and_actions(logits, actions):
    prob = calc_softmax(logits)
    # one_hot_mask = np.eye(np.shape(logits)[-1])[actions] == 1
    one_hot_mask = actions[..., None] == np.arange(np.shape(logits)[-1])
    ret = np.log(prob)[one_hot_mask].reshape(*np.shape(logits)[:-1])
    return ret


def _ground_truth_calculation(
    behaviour_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=None,
    clip_pg_rho_threshold=None,
):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    behaviour_actions_log_probs = calc_log_prob_from_logits_and_actions(
        behaviour_policy_logits, actions
    )

    target_actions_log_probs = calc_log_prob_from_logits_and_actions(
        target_policy_logits, actions
    )

    log_rhos = target_actions_log_probs - behaviour_actions_log_probs

    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

    values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (
                np.prod(discounts[s:t], axis=0)
                * np.prod(cs[s:t], axis=0)
                * clipped_rhos[t]
                * (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t])
            )
            print(" {}-{} discount prod: {}".format(s, t, np.prod(discounts[s:t], axis=0)))
        print("v_s: ", v_s, np.shape(v_s))
        vs.append(v_s)
    vs = np.stack(vs, axis=0)

    pg_advantages = clipped_pg_rhos * (
        rewards
        + discounts * np.concatenate([vs[1:], bootstrap_value[None, :]], axis=0)
        - values
    )

    return vs, pg_advantages


class VtraceTest(unittest.TestCase):
    @parameterized.expand([("Batch1", 1), ("Batch4", 4)])
    def test_from_logits(self, name, batch_size):
        """Tests V-trace against ground truth data calculated in python."""
        seq_len = 5
        num_actions = 7
        values = {
            "behaviour_policy_logits": fill_shaped_arange(
                seq_len, batch_size, num_actions
            ),
            "target_policy_logits": fill_shaped_arange(
                seq_len, batch_size, num_actions
            ),
            "actions": np.random.randint(
                0, num_actions - 1, size=(seq_len, batch_size)
            ),
            # T, B where B_i: [0.9 / (i+1)] * T
            "discounts": np.array(
                [[0.9 / (b + 1) for b in range(batch_size)] for _ in range(seq_len)],
                dtype=np.float32,
            ),
            "rewards": fill_shaped_arange(seq_len, batch_size),
            "values": fill_shaped_arange(seq_len, batch_size) / batch_size,
            "bootstrap_value": fill_shaped_arange(batch_size) + 1.0,
        }
        print("discount: \n", values["discounts"])

        placeholders = {
            # T, B, NUM_ACTIONS
            "behaviour_policy_logits": tf.placeholder(
                dtype=tf.float32, shape=[seq_len, batch_size, num_actions]
            ),
            # T, B, NUM_ACTIONS
            "target_policy_logits": tf.placeholder(
                dtype=tf.float32, shape=[seq_len, batch_size, num_actions]
            ),
            "actions": tf.placeholder(dtype=tf.int32, shape=[seq_len, batch_size]),
            "discounts": tf.placeholder(dtype=tf.float32, shape=[seq_len, batch_size]),
            "rewards": tf.placeholder(dtype=tf.float32, shape=[seq_len, batch_size]),
            "values": tf.placeholder(dtype=tf.float32, shape=[seq_len, batch_size]),
            "bootstrap_value": tf.placeholder(dtype=tf.float32, shape=[batch_size]),
        }

        feed_dict = {placeholders[k]: v for k, v in values.items()}

        ret_vs, ret_pg_avd = vtrace.from_logic_outputs(
            clip_importance_sampling_threshold=None,
            clip_pg_importance_sampling_threshold=None,
            **placeholders
        )
        with tf.Session() as sess:
            vs, pg_adv = sess.run([ret_vs, ret_pg_avd], feed_dict=feed_dict)
            print("vs: \n", vs, "\n\npg_adv\n", pg_adv)

        # calculate true value
        true_vs, true_pg_adv = _ground_truth_calculation(**values)
        print("-" * 10)
        print("true_vs: \n", true_vs, "\n\ntrue_pg_adv\n", true_pg_adv)

        diff_vs = np.sum(np.square(true_vs - vs))
        diff_pg_adv = np.sum(np.square(true_pg_adv - pg_adv))

        print("diff: ", diff_vs, diff_pg_adv)
        assert diff_pg_adv < 1e-5, "diff pg_adv failed"
        assert diff_vs < 1e-5, "diff vs failed"


if __name__ == "__main__":
    unittest.main()
