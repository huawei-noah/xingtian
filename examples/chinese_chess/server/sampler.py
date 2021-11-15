import os
import random
import time
import logging

from config.conf import TrainingConfig, ResourceConfig


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


class Sampler:
    def __init__(self):
        self.distribute_dir = ResourceConfig.distributed_datadir
        self.all_games = TrainingConfig.sample_games

    @staticmethod
    def is_full(directory):
        """
        Args:
            directory: the path to dir
        Return:
            is the dir has 5000 files
        """
        files = os.listdir(directory)
        return len(files) > ResourceConfig.block_min_games - 1

    def sample(self):
        """
        Return sampled games path
        """
        # sample data
        block_dirs = os.listdir(self.distribute_dir)
        # waiting enough data for training
        data_flag = True
        while data_flag:
            block_dirs = os.listdir(self.distribute_dir)
            if len(block_dirs) < ResourceConfig.train_min_block:
                logging.info('waiting for self_play data')
                time.sleep(60)
            else:
                data_flag = False
        block_dirs = [int(_) for _ in block_dirs]
        block_dirs = sorted(block_dirs, reverse=True)
        block_dirs = [str(_) for _ in block_dirs]
        if not self.is_full(os.path.join(self.distribute_dir, block_dirs[0])):
            block_dirs = block_dirs[1:]
        if len(block_dirs) > ResourceConfig.train_max_block:
            # run the newest model
            block_dirs = block_dirs[:ResourceConfig.train_max_block]
        blocks_num = len(block_dirs)
        games_pre_block = int(self.all_games / blocks_num)
        filelist = []
        for i in range(blocks_num):
            block_dir = os.path.join(self.distribute_dir, block_dirs[i])
            block_files = os.listdir(block_dir)
            selected_files = random.sample(block_files, games_pre_block)
            filelist += [os.path.join(block_dir, i) for i in selected_files]
        return filelist
