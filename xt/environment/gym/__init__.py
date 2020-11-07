from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete


def infer_action_type(action_space):
    if isinstance(action_space, Box):
        return 'DiagGaussian'
    elif isinstance(action_space, Discrete):
        return 'Categorical'
    elif isinstance(action_space, (MultiBinary, MultiDiscrete)):
        return 'MultiCategorical'
    else:
        raise KeyError("unknown action space: {}".format(action_space))
