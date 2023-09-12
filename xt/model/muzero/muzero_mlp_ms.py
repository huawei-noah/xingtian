from mindspore import nn
from mindspore.nn import Dense
from xt.model.muzero.muzero_model_ms import MuzeroModelMS
from xt.model.muzero.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers

# pylint: disable=W0201


@Registers.model
class MuzeroMlpMS(MuzeroModelMS):
    """Docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        super().__init__(model_info)

    def create_rep_network(self):
        return RepNet(self.state_dim)

    def create_policy_network(self):
        return PolicyNet(self.value_support_size, self.action_dim)

    def create_dyn_network(self):
        return DynNet(self.action_dim, self.reward_support_size)


class RepNet(nn.Cell):
    def __init__(self, state_dim):
        super().__init__()
        self.hidden = Dense(state_dim[-1],
                            HIDDEN1_UNITS,
                            activation="relu",
                            weight_init="XavierUniform")
        self.out_rep = Dense(
            HIDDEN1_UNITS,
            HIDDEN2_UNITS,
            activation="relu",
            weight_init="XavierUniform")

    def construct(self, x):
        out = self.hidden(x)
        out = self.out_rep(out)
        return out


class PolicyNet(nn.Cell):
    def __init__(self, value_support_size, action_dim):
        super().__init__()
        self.hidden = Dense(
            HIDDEN2_UNITS,
            HIDDEN1_UNITS,
            activation="relu",
            weight_init="XavierUniform")
        self.out_v = Dense(
            HIDDEN1_UNITS,
            value_support_size,
            activation="softmax",
            weight_init="XavierUniform")
        self.out_p = Dense(
            HIDDEN1_UNITS,
            action_dim,
            activation="softmax",
            weight_init="XavierUniform")

    def construct(self, x):
        hidden = self.hidden(x)
        out_v = self.out_v(hidden)
        out_p = self.out_p(hidden)
        return out_p, out_v


class DynNet(nn.Cell):
    def __init__(self, action_dim, reward_support_size):
        super().__init__()
        self.hidden = Dense(
            HIDDEN2_UNITS + action_dim,
            HIDDEN1_UNITS,
            activation="relu",
            weight_init="XavierUniform")
        self.out_h = Dense(
            HIDDEN1_UNITS,
            HIDDEN2_UNITS,
            activation="relu",
            weight_init="XavierUniform")
        self.out_r = Dense(
            HIDDEN1_UNITS,
            reward_support_size,
            activation="softmax",
            weight_init="XavierUniform")

    def construct(self, x):
        hidden = self.hidden(x)
        out_h = self.out_h(hidden)
        out_r = self.out_r(hidden)
        return out_h, out_r
