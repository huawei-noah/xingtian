from xt.model.ms_compat import ms, Dense, Conv2d, Flatten, ReLU, Cell
from xt.model.muzero.muzero_model_ms import MuzeroModelMS
from xt.model.muzero.default_config import HIDDEN_OUT
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers

# pylint: disable=W0201


@Registers.model
class MuzeroCnnMS(MuzeroModelMS):
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


class RepNet(Cell):
    def __init__(self, state_dim):
        super().__init__()
        self.convlayer1 = Conv2d(state_dim[-1],
                                 32,
                                 (8,
                                  8),
                                 stride=(4,
                                         4),
                                 pad_mode="valid",
                                 has_bias=True,
                                 weight_init="XavierUniform")
        self.convlayer2 = Conv2d(32, 32, (4, 4), stride=(2, 2),
                                 pad_mode="valid", has_bias=True,
                                 weight_init="XavierUniform")
        self.convlayer3 = Conv2d(32, 64, (3, 3), stride=(1, 1),
                                 pad_mode="valid", has_bias=True,
                                 weight_init="XavierUniform")
        self.relu = ReLU()
        self.flattenlayer = Flatten()
        dim = (
            (((state_dim[0] - 4) // 4 - 2) // 2 - 2)
            * (((state_dim[1] - 4) // 4 - 2) // 2 - 2)
            * 64
        )
        self.denselayer = Dense(
            dim,
            HIDDEN_OUT,
            activation="relu",
            weight_init="XavierUniform")

    def construct(self, x: ms.Tensor):
        out = x.transpose((0, 3, 1, 2)).astype("float32") / 255.
        out = self.convlayer1(out)
        out = self.relu(out)
        out = self.convlayer2(out)
        out = self.relu(out)
        out = self.convlayer3(out)
        out = self.relu(out)
        out = self.flattenlayer(out)
        out = self.denselayer(out)
        return out


class PolicyNet(Cell):
    def __init__(self, value_support_size, action_dim):
        super().__init__()
        self.hidden = Dense(
            HIDDEN_OUT,
            128,
            activation="relu",
            weight_init="XavierUniform")
        self.out_v = Dense(
            128,
            value_support_size,
            activation="softmax",
            weight_init="XavierUniform")
        self.out_p = Dense(
            128,
            action_dim,
            activation="softmax",
            weight_init="XavierUniform")

    def construct(self, x):
        hidden = self.hidden(x)
        out_v = self.out_v(hidden)
        out_p = self.out_p(hidden)
        return out_p, out_v


class DynNet(Cell):
    def __init__(self, action_dim, reward_support_size):
        super().__init__()
        self.hidden1 = Dense(
            HIDDEN_OUT + action_dim,
            256,
            activation="relu",
            weight_init="XavierUniform")
        self.hidden2 = Dense(256, 128, activation="relu",
                             weight_init="XavierUniform")
        self.out_h = Dense(
            128,
            HIDDEN_OUT,
            activation="relu",
            weight_init="XavierUniform")
        self.out_r = Dense(
            128,
            reward_support_size,
            activation="softmax",
            weight_init="XavierUniform")

    def construct(self, x):
        hidden = self.hidden1(x)
        hidden = self.hidden2(hidden)
        out_h = self.out_h(hidden)
        out_r = self.out_r(hidden)
        return out_h, out_r
