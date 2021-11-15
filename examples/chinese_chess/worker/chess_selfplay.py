import os
import sys

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import numpy as np
from config import conf
from worker.game import DistributedSelfPlayGames


def self_play_gpu(gpu_num, play_times=np.inf, history_selfplay_output_dir=None, epoch=0):
    cn = DistributedSelfPlayGames(
        gpu_num=gpu_num,
        n_playout=conf.SelfPlayConfig.train_playout,
        recoard_dir=conf.ResourceConfig.distributed_datadir,
        c_puct=conf.TrainingConfig.c_puct,
        distributed_dir=conf.ResourceConfig.model_dir,
        dnoise=True,
        is_selfplay=True,
        play_times=play_times,
    )
    
    cn.play(data_url=history_selfplay_output_dir, epoch=epoch)


def self_play_cpu(play_times=np.inf, history_selfplay_output_dir=None, epoch=0):
    cn = DistributedSelfPlayGames(
        gpu_num=None,
        n_playout=conf.SelfPlayConfig.train_playout,
        recoard_dir=conf.ResourceConfig.distributed_datadir,
        c_puct=conf.TrainingConfig.c_puct,
        distributed_dir=conf.ResourceConfig.model_dir,
        dnoise=True,
        is_selfplay=True,
        play_times=play_times,
    )

    cn.play(data_url=history_selfplay_output_dir, epoch=epoch)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    self_play_gpu(0, 200)
