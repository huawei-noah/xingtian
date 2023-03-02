from xt.model.model_utils_ms import ACTIVATION_MAP_MS, get_mlp_backbone_ms, get_mlp_default_settings_ms
from xt.model.ppo.default_config import MLP_SHARE_LAYERS
from xt.model.ppo.ppo_ms import PPOMS
from xt.model.tf_compat import tf
from zeus.common.util.register import Registers
from xt.model.ms_utils import MSVariables

@Registers.model
class PpoMlpMS(PPOMS):
    """Build PPO MLP network."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config')

        self.vf_share_layers = model_config.get('VF_SHARE_LAYERS', MLP_SHARE_LAYERS)
        self.hidden_sizes = model_config.get('hidden_sizes', get_mlp_default_settings_ms('hidden_sizes'))
        activation = model_config.get('activation', get_mlp_default_settings_ms('activation'))
        try:
            self.activation = ACTIVATION_MAP_MS[activation]
        except KeyError:
            raise KeyError('activation {} not implemented.'.format(activation))

        super().__init__(model_info)

    def create_model(self, model_info):
        net = get_mlp_backbone_ms(self.state_dim, self.action_dim, self.hidden_sizes, self.activation,
                                 self.vf_share_layers, self.verbose, dtype=self.input_dtype)
        self.actor_var = MSVariables(net)
        return net
