from xt.model.tf_compat import tf


def actor_loss_with_entropy(dist, adv, old_log_p, behavior_action, clip_ratio, ent_coef):
    """Calculate actor loss with entropy."""
    action_log_prob = dist.log_prob(behavior_action)
    ratio = tf.exp(action_log_prob - old_log_p)

    surr_loss_1 = ratio * adv
    surr_loss_2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    surr_loss = tf.reduce_mean(tf.minimum(surr_loss_1, surr_loss_2))

    ent = dist.entropy()
    ent = tf.reduce_mean(ent)

    return -surr_loss - ent_coef * ent


def critic_loss(target_v, out_v, old_v, val_clip):
    """Use clipped value loss as default."""
    vf_losses1 = tf.square(out_v - target_v)
    val_pred_clipped = old_v + tf.clip_by_value(out_v - old_v, -val_clip, val_clip)
    vf_losses2 = tf.square(val_pred_clipped - target_v)
    vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    return vf_loss
