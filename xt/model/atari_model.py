"""Model architecture for atari game."""


def get_atari_filter(shape):
    """Get default model set for atari environments."""
    shape = list(shape)
    # (out_size, kernel, stride)
    filters_84x84 = [
        [16, 8, 4],
        [32, 4, 2],
        [256, 11, 1],
    ]
    filters_42x42 = [
        [16, 4, 2],
        [32, 4, 2],
        [256, 11, 1],
    ]
    if len(shape) == 3 and shape[:2] == [84, 84]:
        return filters_84x84
    elif len(shape) == 3 and shape[:2] == [42, 42]:
        return filters_42x42
    else:
        raise ValueError("Without default architecture for obs shape {}".format(shape))
