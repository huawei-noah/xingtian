import os

from config.conf import ResourceConfig, TrainingConfig


def get_latest_weight_path():
    """
    return: latest_weight_path, absolution path
    """
    weightlist = os.listdir(ResourceConfig.model_dir)
    weightlist = [i[:-6] for i in weightlist if '.index' in i]
    if len(weightlist) == 0:
        raise Exception('no pre_trained weights')
    latest_netname = sorted(weightlist)[-1]
    return os.path.join(ResourceConfig.model_dir, latest_netname)


def cycle_lr(step):
    index = (step // 500) % len(TrainingConfig.lr)
    return TrainingConfig.lr[index]
