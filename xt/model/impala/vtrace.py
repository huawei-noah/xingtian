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
"""Functions to compute V-trace for off-policy actor-learner architecture.

The following codes refers to DeepMind/scale_agent:
https://github.com/deepmind/scalable_agent/blob/master/vtrace.py

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

from __future__ import absolute_import, division, print_function

from xt.model.tf_compat import tf


def from_logic_outputs(behaviour_policy_logic_outputs,
                       target_policy_logic_outputs,
                       actions,
                       discounts,
                       rewards,
                       values,
                       bootstrap_value,
                       clip_importance_sampling_threshold=1.0,
                       clip_pg_importance_sampling_threshold=1.0):
    """
    Calculate vtrace with logic outputs.

    :param behaviour_policy_logic_outputs: behaviour_policy_logic_outputs
    :param target_policy_logic_outputs: target_policy_logic_outputs
    :param actions:
    :param discounts:
    :param rewards:
    :param values:
    :param bootstrap_value:
    :param clip_importance_sampling_threshold:
    :param clip_pg_importance_sampling_threshold:
    :return:
    """
    behaviour_policy_logic_outputs = tf.convert_to_tensor(behaviour_policy_logic_outputs, dtype=tf.float32)
    target_policy_logic_outputs = tf.convert_to_tensor(target_policy_logic_outputs, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    # support [T, B, Action_dimension]
    behaviour_policy_logic_outputs.shape.assert_has_rank(3)
    target_policy_logic_outputs.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)

    target_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=target_policy_logic_outputs,
                                                                      labels=actions)

    behaviour_log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behaviour_policy_logic_outputs,
                                                                         labels=actions)

    # log importance sampling weight
    importance_sampling_weights = tf.exp(target_log_prob - behaviour_log_prob)

    clipped_importance_sampling_weight = tf.minimum(clip_importance_sampling_threshold, importance_sampling_weights)
    clipped_pg_importance_sampling_weight = tf.minimum(clip_pg_importance_sampling_threshold,
                                                       importance_sampling_weights)

    # coefficient, similar to the 'trace cutting'
    coefficient = tf.minimum(1.0, importance_sampling_weights)

    next_values = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)

    # temporal difference, as the fixed point
    deltas = clipped_importance_sampling_weight * (rewards + discounts * next_values - values)
    sequences = (deltas, discounts, coefficient)

    # calculate Vtrace with tf.scan, and set reverse: True, back --> begin
    def scan_fn(cumulative_value, sequence_item):
        _delta, _discount, _coefficient = sequence_item
        return _delta + _discount * _coefficient * cumulative_value

    last_values = tf.zeros_like(bootstrap_value)
    temporal_difference = tf.scan(
        fn=scan_fn,
        elems=sequences,
        initializer=last_values,
        parallel_iterations=1,
        back_prop=False,
        reverse=True,
    )

    value_of_states = tf.add(temporal_difference, values)
    # Advantage for policy gradient.
    value_of_next_state = tf.concat([value_of_states[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    pg_advantages = clipped_pg_importance_sampling_weight * (rewards + discounts * value_of_next_state - values)

    value_of_states = tf.stop_gradient(value_of_states)
    pg_advantages = tf.stop_gradient(pg_advantages)
    return value_of_states, pg_advantages
