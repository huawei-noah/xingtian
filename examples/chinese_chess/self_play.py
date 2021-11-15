"""
chinese chess self play worker starter
1. upload game data, download weights / nash res
2. self play
"""
import os
import time
import argparse
import subprocess
import json

import logging
from multiprocessing import Process

import moxing as mox
from config.conf import ResourceConfig, SelfPlayConfig, EvaluateConfig
from lib.utils import init_dir, sorted_custom, get_latest_weight_path


project_path = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


def self_play_process(gpu_num):
    while True:
        try:
            logging.info('self_play_chosen_model gpu_num {}'.format(gpu_num))
            worker_self_play = os.path.join(project_path, 'self_play_worker.py')
            logging.info('excute command python {} -gpu_num {}'.format(worker_self_play, gpu_num))
            subprocess.run('{} {} -gpu_num {}'.format(ResourceConfig.python_executor, worker_self_play, gpu_num), shell=False)
        except BaseException:
            logging.error('self-play error')


def self_play(args):
    ps = []

    if args.worker_type == 'gpu':
        for gpu_num in range(SelfPlayConfig.gpu_num):
            for i in range(SelfPlayConfig.num_proc_each_gpu):
                p = Process(target=self_play_process, name="worker" + str(i), args=(gpu_num,))
                ps.append(p)
    elif args.worker_type == 'yigou':
        for gpu_num in range(1):
            for i in range(8):
                p = Process(target=self_play_process, name="worker" + str(i), args=(gpu_num,))
                ps.append(p)
    elif args.worker_type == 'yigou_cpu':
        for i in range(SelfPlayConfig.yigou_cpu_proc_num):
            p = Process(target=self_play_process, name="worker" + str(i), args=(None,))
            ps.append(p)
    else:
        for i in range(SelfPlayConfig.cpu_proc_num):
            p = Process(target=self_play_process, name="worker" + str(i), args=(None,))
            ps.append(p)

    for i in ps:
        i.start()

    for i in ps:
        i.join()

    logging.info('self done !!!')


def download_weight_core():
    nash_res_dirs = os.listdir(ResourceConfig.nash_battle_local_dir)
    nash_res_dirs.sort()
    lastest_nash_res_file = os.path.join(
        ResourceConfig.nash_battle_local_dir, nash_res_dirs[-1], ResourceConfig.nash_res_json)
    with open(lastest_nash_res_file, 'r') as f:
        nash_res = json.load(f)
    [players, _] = nash_res['nash_res']
    models_needed = players
    models_needed.sort()

    models_local = os.listdir(ResourceConfig.model_dir)
    models_local = [model[:-6] for model in models_local if '.index' in model]
    models_local.sort()

    # download weight
    for model in models_needed:
        if model not in models_local:
            file1 = model + '.data-00000-of-00001'
            file2 = model + '.meta'
            file3 = model + '.index'
            mox.file.copy(
                os.path.join(ResourceConfig.pool_weights_yundao_dir, file1),
                os.path.join(ResourceConfig.model_dir, file1)
            )
            mox.file.copy(
                os.path.join(ResourceConfig.pool_weights_yundao_dir, file2),
                os.path.join(ResourceConfig.model_dir, file2)
            )
            mox.file.copy(
                os.path.join(ResourceConfig.pool_weights_yundao_dir, file3),
                os.path.join(ResourceConfig.model_dir, file3)
            )
            logging.info('----------------- doneload weight {}'.format(model))

    # keep the latest 5 models
    models_local = models_local[:-5]
    # deleting model
    for model in models_local:
        if model not in models_needed:
            file1 = model + '.data-00000-of-00001'
            file2 = model + '.meta'
            file3 = model + '.index'
            os.remove(os.path.join(ResourceConfig.model_dir, file1))
            os.remove(os.path.join(ResourceConfig.model_dir, file2))
            os.remove(os.path.join(ResourceConfig.model_dir, file3))
            logging.info('----------------- deleting weight {}'.format(model))


def download_weight():
    """
    only download models in model pool list and delete model not needed
    """
    while True:
        try:
            download_weight_core()
        except:
            logging.error('download weight error')

        time.sleep(SelfPlayConfig.self_play_download_weight_dt)


def download_nash_file_core():
    """
    download nash res with and without bot
    """
    local_file_dirs = os.listdir(ResourceConfig.nash_battle_local_dir)
    local_file_dirs.sort()
    yundao_file_dirs = mox.file.list_directory(ResourceConfig.nash_battle_yundao_dir)
    yundao_file_dirs.sort()
    yundao_file_dirs = yundao_file_dirs[-2:]
    for single_dir in yundao_file_dirs:
        if single_dir not in local_file_dirs:
            try:
                mox.file.copy(
                    os.path.join(
                        ResourceConfig.nash_battle_yundao_dir,
                        single_dir,
                        ResourceConfig.nash_res_json
                    ),
                    os.path.join(
                        ResourceConfig.nash_battle_local_dir,
                        single_dir,
                        ResourceConfig.nash_res_json
                    )
                )
            except BaseException:
                logging.error('nash_res_json not found')

            try:
                mox.file.copy(
                    os.path.join(
                        ResourceConfig.nash_battle_yundao_dir,
                        single_dir,
                        ResourceConfig.nash_res_bot_json
                    ),
                    os.path.join(
                        ResourceConfig.nash_battle_local_dir,
                        single_dir,
                        ResourceConfig.nash_res_bot_json
                    )
                )
            except BaseException:
                logging.error('nash_res_bot_json not found')


def download_nash_file():
    while True:
        try:
            download_nash_file_core()
        except:
            logging.error('download nash error')

        time.sleep(SelfPlayConfig.self_play_download_weight_dt)


def upload_game_data():
    """
    upload game every conf.self_play_upload_data_dt seconds
    """
    while True:
        try:
            new_files = os.listdir(ResourceConfig.distributed_datadir)

            yundao_new_data_dirs = mox.file.list_directory(ResourceConfig.new_data_yundao_dir)
            yundao_new_data_dirs = sorted_custom(yundao_new_data_dirs)
            latest_yundao_new_data_dir = \
                os.path.join(ResourceConfig.new_data_yundao_dir, yundao_new_data_dirs[-1])

            for file in new_files:
                mox.file.copy(
                    os.path.join(ResourceConfig.distributed_datadir, file),
                    os.path.join(latest_yundao_new_data_dir, file)
                )

            for file in new_files:
                os.remove(os.path.join(ResourceConfig.distributed_datadir, file))
                
            if len(mox.file.list_directory(latest_yundao_new_data_dir)) > ResourceConfig.block_min_games and \
                    not mox.file.exists(os.path.join(
                        ResourceConfig.new_data_yundao_dir,
                        str(int(yundao_new_data_dirs[-1]) + 1)
                    )):
                mox.file.make_dirs(os.path.join(
                    ResourceConfig.new_data_yundao_dir,
                    str(int(yundao_new_data_dirs[-1]) + 1)
                ))
        except:
            logging.error('upload game data error')

        time.sleep(SelfPlayConfig.self_play_upload_data_dt)


def process_files():
    weight_p = Process(target=download_weight)
    weight_p.start()

    nash_file_p = Process(target=download_nash_file)
    nash_file_p.start()

    game_data_p = Process(target=upload_game_data)
    game_data_p.start()


def init_self_play():
    download_nash_file_core()
    download_weight_core()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu_num', type=str, default='12')
    parser.add_argument('--worker_type', type=str, default='gpu')
    args, _ = parser.parse_known_args()

    init_dir()

    init_self_play()

    # moxing process
    process_files()

    # start self play
    self_play(args)


if __name__ == "__main__":
    main()
