"""
for nash battle
when new model generated, start battle

1. download weights, upload battle data / nash res / nash rpp

"""
import os
import time
import argparse
import subprocess
import json

import logging
from multiprocessing import Process

import moxing as mox
from config import conf
from lib.utils import init_dir, get_latest_weight_path, get_sorted_weight_list, get_model_pool_list, check_if_dir_exit
from nash.cal_nash_eq import NashEq


project_path = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


def download_latest_model(args):
    """
    download latest weight
    """
    if args.nash_index == 0:
        while True:
            done = False
            new_weight_names = []
            local_weights = os.listdir(conf.ResourceConfig.model_dir)
            local_weights.sort()
            latest_weight_stamp = get_latest_weight_path()

            yundao_weights_ori = mox.file.list_directory(os.path.join(conf.ResourceConfig.pool_weights_yundao_dir))
            yundao_weights = [weight for weight in yundao_weights_ori if '.index' in weight]
            yundao_weights.sort()

            yundao_latest_weight = yundao_weights[-1]
            yundao_latest_weight = yundao_latest_weight[:-6]
            if latest_weight_stamp >= yundao_latest_weight:
                time.sleep(conf.EvaluateConfig.nash_eva_waiting_dt)
                logging.info('waiting for weight')
                continue

            for weight in yundao_weights:
                if weight not in local_weights:
                    new_weight_names.append(weight.split('.')[0])

            if not new_weight_names:
                # no new weight then wait
                time.sleep(conf.EvaluateConfig.nash_eva_waiting_dt)
                logging.info('waiting for weight')
                continue

            new_weight_names.sort()
            weight = new_weight_names[-1]
            file1 = weight + '.data-00000-of-00001'
            file2 = weight + '.meta'
            file3 = weight + '.index'
            if file1 in yundao_weights_ori and file2 in yundao_weights_ori and file3 in yundao_weights_ori:
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.pool_weights_yundao_dir, file1),
                    os.path.join(conf.ResourceConfig.model_dir, file1)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.pool_weights_yundao_dir, file2),
                    os.path.join(conf.ResourceConfig.model_dir, file2)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.pool_weights_yundao_dir, file3),
                    os.path.join(conf.ResourceConfig.model_dir, file3)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.pool_weights_yundao_dir, file1),
                    os.path.join(conf.ResourceConfig.nash_weights_yundao_dir, file1)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.pool_weights_yundao_dir, file2),
                    os.path.join(conf.ResourceConfig.nash_weights_yundao_dir, file2)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.pool_weights_yundao_dir, file3),
                    os.path.join(conf.ResourceConfig.nash_weights_yundao_dir, file3)
                )
                done = True
                logging.info('----------------- doneload weight {}'.format(weight))

            if done:
                return
    else:
        while True:
            done = False
            new_weight_names = []
            local_weights = os.listdir(conf.ResourceConfig.model_dir)
            local_latest_weight_stamp = get_latest_weight_path()

            yundao_weights_ori = mox.file.list_directory(os.path.join(conf.ResourceConfig.nash_weights_yundao_dir))
            yundao_weights = [weight for weight in yundao_weights_ori if '.index' in weight]
            yundao_weights.sort()

            yundao_latest_weight = yundao_weights[-1]
            yundao_latest_weight = yundao_latest_weight[:-6]
            if local_latest_weight_stamp >= yundao_latest_weight:
                time.sleep(conf.EvaluateConfig.nash_eva_waiting_dt)
                logging.info('waiting for weight')
                continue

            for weight in yundao_weights:
                if weight not in local_weights:
                    new_weight_names.append(weight.split('.')[0])

            if not new_weight_names:
                # no new weight then wait
                time.sleep(conf.EvaluateConfig.nash_eva_waiting_dt)
                logging.info('waiting for weight')
                continue

            new_weight_names.sort()
            weight = new_weight_names[-1]
            file1 = weight + '.data-00000-of-00001'
            file2 = weight + '.meta'
            file3 = weight + '.index'
            if file1 in yundao_weights_ori and file2 in yundao_weights_ori and file3 in yundao_weights_ori:
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.nash_weights_yundao_dir, file1),
                    os.path.join(conf.ResourceConfig.model_dir, file1)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.nash_weights_yundao_dir, file2),
                    os.path.join(conf.ResourceConfig.model_dir, file2)
                )
                mox.file.copy(
                    os.path.join(conf.ResourceConfig.nash_weights_yundao_dir, file3),
                    os.path.join(conf.ResourceConfig.model_dir, file3)
                )
                done = True
                logging.info('----------------- doneload weight {}'.format(weight))

            if done:
                return


def nash_battle_process(gpu_num, worker_index, nash_index):
    try:
        logging.info('nash battle gpu_num {}'.format(gpu_num))
        worker_nash_battle = os.path.join(project_path, 'nash_evaluate_worker.py')
        logging.info('excute command python {} -gpu_num {} -worker_index {} -nash_index {}'.format(
            worker_nash_battle, gpu_num, worker_index, nash_index))
        subprocess.run('{} {} -gpu_num {} -worker_index {}  -nash_index {}'.format(
            conf.ResourceConfig.python_executor, worker_nash_battle, gpu_num, worker_index, nash_index), shell=False)
    except:
        logging.error('nahs battle error')


def nash_battle_done(latest_model_name, battle_models):
    for model in battle_models:
        battle_dir = os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_name, model)
        check_if_dir_exit(battle_dir)

        if len(os.listdir(battle_dir)) < conf.EvaluateConfig.nash_each_battle_num:
            return False

    return True


def upload_and_download_files(battle_models):
    """
    sharing battle data between battle nodes
    """
    latest_model_name = get_latest_weight_path()
    while True:
        if nash_battle_done(latest_model_name, battle_models):
            return

        # sharing battle file between nash_index 0 and other nodes
        for model in battle_models:
            local_battle_dir = os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_name, model)
            check_if_dir_exit(local_battle_dir)
            yundao_battle_dir = os.path.join(conf.ResourceConfig.nash_battle_yundao_share_dir, latest_model_name, model)
            if not mox.file.exists(yundao_battle_dir):
                mox.file.make_dirs(yundao_battle_dir)

            local_files = os.listdir(local_battle_dir)
            yundao_files = mox.file.list_directory(yundao_battle_dir)

            for file in local_files:
                if not mox.file.exists(os.path.join(yundao_battle_dir, file)):
                    logging.info('uploading {}'.format(os.path.join(yundao_battle_dir, file)))
                    mox.file.copy(os.path.join(local_battle_dir, file), os.path.join(yundao_battle_dir, file))

            for file in yundao_files:
                if not os.path.exists(os.path.join(local_battle_dir, file)):
                    logging.info('downloading {}'.format(os.path.join(yundao_battle_dir, file)))
                    mox.file.copy(os.path.join(yundao_battle_dir, file), os.path.join(local_battle_dir, file))

        time.sleep(conf.EvaluateConfig.nash_eva_waiting_dt)


def get_nash_battle_models():
    latest_model_stamp = get_latest_weight_path()
    local_nash_battle_list_dir = os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_stamp)
    check_if_dir_exit(local_nash_battle_list_dir)

    model_pool_list = get_model_pool_list()

    # add eliminated model (only when there are enough models)
    if len(get_sorted_weight_list()) > conf.EvaluateConfig.model_pool_size:
        previous_model_stamp = get_sorted_weight_list()[-2]
        eliminate_model_file_path = os.path.join(
            conf.ResourceConfig.nash_battle_local_dir,
            previous_model_stamp,
            conf.ResourceConfig.eliminated_model_json
        )
        if os.path.exists(eliminate_model_file_path):
            with open(eliminate_model_file_path, 'r') as f:
                eliminated_model_dict = json.load(f)
            eliminated_model = eliminated_model_dict['eliminated_model']
            if model_pool_list:
                model_pool_list.append(eliminated_model)
            else:
                model_pool_list = [eliminated_model]

    # add bot
    if model_pool_list:
        model_pool_list.append(conf.EvaluateConfig.bot_name)
    else:
        model_pool_list = [conf.EvaluateConfig.bot_name]

    return model_pool_list


def nash_battle(args):
    nash_battle_models = get_nash_battle_models()
    Process(target=upload_and_download_files, args=(nash_battle_models,)).start()

    ps = []

    worker_index = 0
    if args.worker_type == 'gpu':
        for gpu_num in range(conf.EvaluateConfig.gpu_num):
            for i in range(conf.EvaluateConfig.num_proc_each_gpu):
                p = Process(
                    target=nash_battle_process,
                    name="worker" + str(i),
                    args=(gpu_num, worker_index, args.nash_index,)
                )
                ps.append(p)
                worker_index += 1
    else:
        for i in range(int(args.cpu_num)):
            p = Process(target=nash_battle_process, name="worker" + str(i), args=(None, worker_index, args.nash_index,))
            ps.append(p)
            worker_index += 1

    for i in ps:
        i.start()

    for i in ps:
        i.join()

    logging.info('battle done !!!')


def maintain_pool_model(nash_res, model_stamps):
    if len(model_stamps) <= conf.EvaluateConfig.model_pool_size:
        return model_stamps

    [players, q] = nash_res
    players_with_q = []
    eliminated_players_with_q = []
    for i in range(len(players)):
        players_with_q.append([players[i], q[i]])
        if q[i] < 0.001:
            eliminated_players_with_q.append([players[i], 0])
        else:
            eliminated_players_with_q.append([players[i], q[i]])

    # remove q最小的 / 时间靠前的
    eliminated_players_with_q.sort(key=lambda x: (x[1], x[0]))
    eliminated_model, eliminated_model_q = eliminated_players_with_q.pop(0)
    for model_index in range(len(players_with_q)):
        if eliminated_model == players_with_q[model_index][0]:
            players_with_q.pop(model_index)
            break

    # save eliminate_model file
    lastest_model_stamp = get_latest_weight_path()
    local_eliminated_model_file_path = os.path.join(
        conf.ResourceConfig.nash_battle_local_dir,
        lastest_model_stamp,
        conf.ResourceConfig.eliminated_model_json
    )
    eliminated_model_dict = {
        'eliminated_model': eliminated_model
    }
    with open(local_eliminated_model_file_path, 'w+') as f:
        json.dump(eliminated_model_dict, f)

    model_pool = [player[0] for player in players_with_q]
    model_pool.sort()
    return model_pool


def get_eliminated_model_name():
    eliminated_model = None
    previous_model_stamp = get_sorted_weight_list()[-2]
    local_eliminated_model_file_path = os.path.join(
        conf.ResourceConfig.nash_battle_local_dir,
        previous_model_stamp,
        conf.ResourceConfig.eliminated_model_json
    )
    yundao_eliminated_model_file_path = os.path.join(
        conf.ResourceConfig.nash_battle_yundao_dir,
        previous_model_stamp,
        conf.ResourceConfig.eliminated_model_json
    )
    if os.path.exists(local_eliminated_model_file_path):
        with open(local_eliminated_model_file_path, 'r') as f:
            eliminated_model_dict = json.load(f)
            eliminated_model = eliminated_model_dict['eliminated_model']
    elif mox.file.exists(yundao_eliminated_model_file_path):
        mox.file.copy(yundao_eliminated_model_file_path, local_eliminated_model_file_path)
        with open(local_eliminated_model_file_path, 'r') as f:
            eliminated_model_dict = json.load(f)
            eliminated_model = eliminated_model_dict['eliminated_model']
    return eliminated_model


def cal_nash():
    """
    cal nash and rpp
    upload all res files
    """
    model_stamps = get_sorted_weight_list()

    # get model_pool_list
    model_pool_list = get_model_pool_list()

    latest_model_stamp = get_latest_weight_path()

    # cal nash
    nash = NashEq()
    nash_res, rate_table_pd, nash_res_with_bot, rate_table_bot_pd \
        = nash.build_rate_table(model_pool_list, latest_model_stamp, conf.ResourceConfig.nash_battle_local_dir)

    model_pool = maintain_pool_model(nash_res, model_stamps)

    # save model pool
    model_pool_file_local_path = os.path.join(
        conf.ResourceConfig.nash_battle_local_dir,
        latest_model_stamp,
        conf.ResourceConfig.model_pool_list_json
    )
    model_pool_dict = {
        'model_pool': model_pool
    }
    with open(model_pool_file_local_path, 'w+') as f:
        json.dump(model_pool_dict, f)

    # save nash res
    nash_res_file_local_path = os.path.join(
        conf.ResourceConfig.nash_battle_local_dir,
        latest_model_stamp,
        conf.ResourceConfig.nash_res_json
    )
    [players, q] = nash_res
    players = [_ for _ in players]
    q = [_ for _ in q]
    nash_res_dict = {
        'nash_res': [players, q]
    }
    with open(nash_res_file_local_path, 'w+') as f:
        json.dump(nash_res_dict, f)

    # save nash res
    nash_res_bot_file_local_path = os.path.join(
        conf.ResourceConfig.nash_battle_local_dir,
        latest_model_stamp,
        conf.ResourceConfig.nash_res_bot_json
    )
    [players_bot, q_bot] = nash_res_with_bot
    players_bot = [_ for _ in players_bot]
    q_bot = [_ for _ in q_bot]
    nash_res_bot_dict = {
        'nash_res_bot': [players_bot, q_bot]
    }
    with open(nash_res_bot_file_local_path, 'w+') as f:
        json.dump(nash_res_bot_dict, f)

    rate_table_pd.to_csv(
        os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_stamp, 'rate_table.csv'),
        encoding='utf-8',
        index=False,
        header=None
    )

    rate_table_bot_pd.to_csv(
        os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_stamp, 'rate_table_bot.csv'),
        encoding='utf-8',
        index=False,
        header=None
    )

    # cal rpp
    if len(model_stamps) > conf.EvaluateConfig.model_pool_size and get_eliminated_model_name():
        previous_eliminated_model = get_eliminated_model_name()
        last_rate_table_path = os.path.join(
            conf.ResourceConfig.nash_battle_local_dir,
            model_stamps[-2],
            'rate_table.csv'
        )
        if os.path.exists(last_rate_table_path):
            row_nash_q, col_nash_q, rate_tables, rpp, rate_table_rpp_pd = nash.rpp(
                last_rate_table_path,
                model_pool_list,
                previous_eliminated_model,
                latest_model_stamp,
                conf.ResourceConfig.nash_battle_local_dir
            )
            row_nash_q = [_ for _ in row_nash_q]
            col_nash_q = [_ for _ in col_nash_q]
            rpp_nash_res = [row_nash_q, col_nash_q]

            # save rpp nash res
            rpp_res_file_local_path = os.path.join(
                conf.ResourceConfig.nash_battle_local_dir,
                latest_model_stamp,
                'rpp.json'
            )
            rpp_res_dict = {
                'rpp_res': rpp_nash_res
            }
            with open(rpp_res_file_local_path, 'w+') as f:
                json.dump(rpp_res_dict, f)

            # save rate_table_rpp
            rate_table_rpp_pd.to_csv(
                os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_stamp, 'rate_table_rpp.csv'),
                encoding='utf-8',
                index=False,
                header=None
            )

            # save rpp
            rpp_dict = {
                'rpp': rpp
            }
            rpp_json_file_local = os.path.join(
                conf.ResourceConfig.nash_battle_local_dir,
                latest_model_stamp,
                'rpp_res.json'
            )
            with open(rpp_json_file_local, 'w+') as f:
                json.dump(rpp_dict, f)


def upload_all_res():
    # upload all res
    latest_model_stamp = get_latest_weight_path()
    local_nash_res_dir = os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_stamp)
    yundao_nash_res_dir = os.path.join(conf.ResourceConfig.nash_battle_yundao_dir, latest_model_stamp)
    mox.file.copy_parallel(local_nash_res_dir, yundao_nash_res_dir)


def download_model_pool_list(args):
    if args.nash_index != 0:
        try:
            previous_model_stamp = get_sorted_weight_list()[-2]
            mox.file.copy(
                os.path.join(conf.ResourceConfig.nash_battle_yundao_dir,
                             previous_model_stamp, conf.ResourceConfig.model_pool_list_json),
                os.path.join(conf.ResourceConfig.nash_battle_local_dir,
                             previous_model_stamp, conf.ResourceConfig.model_pool_list_json)
            )
        except:
            logging.error('download model pool list error')

        try:
            previous_model_stamp = get_sorted_weight_list()[-2]
            mox.file.copy(
                os.path.join(conf.ResourceConfig.nash_battle_yundao_dir,
                             previous_model_stamp, conf.ResourceConfig.eliminated_model_json),
                os.path.join(conf.ResourceConfig.nash_battle_local_dir,
                             previous_model_stamp, conf.ResourceConfig.eliminated_model_json)
            )
        except:
            logging.error('download eliminated_model_json error')


def init_nash(args):
    # download weights
    mox.file.copy_parallel(conf.ResourceConfig.nash_weights_yundao_dir, conf.ResourceConfig.model_dir)

    model_stamps = get_sorted_weight_list()
    logging.info('weights now {}'.format(model_stamps))

    # download nash battle files
    if args.nash_index == 0:
        mox.file.copy_parallel(conf.ResourceConfig.nash_battle_yundao_dir, conf.ResourceConfig.nash_battle_local_dir)

    download_model_pool_list(args)


def nash_eva(args):

    init_nash(args)

    # 初始化后立即nash battle 一次
    nash_battle(args)

    if args.nash_index == 0:
        try:
            cal_nash()
        except:
            logging.error('cal nash error')

        upload_all_res()

    while True:
        # wait for latest model
        download_latest_model(args)

        download_model_pool_list(args)

        nash_battle(args)

        if args.nash_index == 0:
            try:
                cal_nash()
            except BaseException:
                logging.error('cal nash error')

            upload_all_res()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu_num', type=str, default='32')
    parser.add_argument('--worker_type', type=str, default='gpu')
    parser.add_argument('--nash_index', type=int, default=0)  # nash worker 0 for cal nash
    args, _ = parser.parse_known_args()

    init_dir()

    # start self play
    nash_eva(args)


if __name__ == "__main__":
    main()
