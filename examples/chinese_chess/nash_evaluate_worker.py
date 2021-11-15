"""
nash battle worker,
input gpu num and worker index
latest model vs others (models in model pool, bot, eliminated model)
"""
import os
import argparse
import logging
import time
import json
import random
import copy
from multiprocessing import Process

import tensorflow as tf
import numpy as np

from bot.evaluate.single_evaluate_worker import CchessPlayer
from lib import cbf
from agent import resnet, players
from env.cchess_env import create_uci_labels
from env.game_state import GameState
from worker.game import count_piece
from config import conf
from lib.utils import get_latest_weight_path, check_if_dir_exit, get_sorted_weight_list, get_model_pool_list
import moxing as mox


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s][%(message)s]",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_model_to_battle(model_pool_list, args):
    """

    """
    if not model_pool_list:
        return [], []

    len_pre_node = int(len(model_pool_list) / conf.EvaluateConfig.nash_nodes_num)
    if args.nash_index < conf.EvaluateConfig.nash_nodes_num - 1:
        model_pool_this_nash_worker = model_pool_list[args.nash_index * len_pre_node: (args.nash_index + 1) * len_pre_node]
    else:
        model_pool_this_nash_worker = model_pool_list[args.nash_index * len_pre_node:]

    models_to_battle_dict = {}
    models_to_battle_other_workers_dict = {}
    models_to_battle_other_node_dict = {}

    nash_evaluate_worker_num = conf.EvaluateConfig.gpu_num * conf.EvaluateConfig.num_proc_each_gpu
    model_pool_index = 0
    model_pool_size = len(model_pool_this_nash_worker)
    for worker_index in range(nash_evaluate_worker_num):
        models_to_battle_dict[worker_index] = [model_pool_this_nash_worker[model_pool_index]]
        model_pool_index += 1
        if model_pool_index >= model_pool_size:
            model_pool_index = 0

    for worker_index in range(nash_evaluate_worker_num):
        models_to_battle_other_workers_dict[worker_index] = \
            list(set(model_pool_this_nash_worker) - set(models_to_battle_dict[worker_index]))

    for worker_index in range(nash_evaluate_worker_num):
        models_to_battle_other_node_dict[worker_index] = \
            list(set(model_pool_list) - set(model_pool_this_nash_worker))

    return (models_to_battle_dict[int(args.worker_index)],
            models_to_battle_other_workers_dict[int(args.worker_index)],
            models_to_battle_other_node_dict[int(args.worker_index)])


def get_nash_battle_list(args):
    latest_model_stamp = get_latest_weight_path()
    local_nash_battle_list_dir = os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_stamp)
    check_if_dir_exit(local_nash_battle_list_dir)

    model_pool_list = get_model_pool_list()
    models_to_battle, models_to_battle_other_workers, models_to_battle_other_node\
        = get_model_to_battle(model_pool_list, args)

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
            if models_to_battle_other_workers:
                models_to_battle_other_workers.append(eliminated_model)
            else:
                models_to_battle_other_workers = [eliminated_model]

    # add bot
    if models_to_battle_other_workers:
        models_to_battle_other_workers.append(conf.EvaluateConfig.bot_name)
    else:
        models_to_battle_other_workers = [conf.EvaluateConfig.bot_name]

    np.random.shuffle(models_to_battle_other_workers)
    models_to_battle.extend(models_to_battle_other_workers)

    np.random.shuffle(models_to_battle_other_node)
    models_to_battle.extend(models_to_battle_other_node)

    logging.info('models_to_battle {}'.format(models_to_battle))

    return models_to_battle


def play_till_end(player_w, player_b, player_w_name, player_b_name):
    gamestate = GameState()
    winner = 'peace'
    moves = []
    peace_round = 0
    remain_piece = count_piece(gamestate.statestr)

    past_str_action = {}

    while True:
        statestr_this_step = copy.deepcopy(gamestate.statestr)

        if gamestate.move_number % 2 == 0:
            player_name = player_w_name
            player = player_w
            opponent_player = player_b
            opponent_player_name = player_b_name
        else:
            player_name = player_b_name
            player = player_b
            opponent_player = player_w
            opponent_player_name = player_w_name

        no_act = None
        # if statestr_this_step in past_str_action.keys():
        #     no_act = past_str_action[statestr_this_step]

        start_time = time.time()
        if player_name == conf.EvaluateConfig.bot_name:
            red_move = True if gamestate.currentplayer == 'w' else False
            move = player.get_action(gamestate.statestr, gamestate.move_number, red_move=red_move)
        else:
            move = player.get_action(gamestate, no_act=no_act)

        if opponent_player_name != conf.EvaluateConfig.bot_name:
            opponent_player.oppponent_make_move(move)

        total_time = time.time() - start_time
        if move is None:
            winner = 'b' if gamestate.currentplayer == 'w' else 'w'
            break

        # logging.info('move {} {} {} play {} use {:.2f}s pr {} pid {} statestr {}'.format(
        #     gamestate.move_number,
        #     gamestate.currentplayer,
        #     player_name,
        #     move,
        #     total_time,
        #     peace_round,
        #     os.getpid(),
        #     gamestate.statestr)
        # )

        if player_name == conf.EvaluateConfig.bot_name:
            gamestate.do_move(move)
        moves.append(move)

        # add statestr_action history
        if statestr_this_step in past_str_action.keys():
            past_str_action[statestr_this_step].append(move)
        else:
            past_str_action[statestr_this_step] = [move]

        loop, winner_if_loop = gamestate.long_catch_or_looping()
        if loop:
            winner = winner_if_loop
            break

        game_end, winner_p = gamestate.game_end()
        if game_end:
            winner = winner_p
            break

        remain_piece_round = count_piece(gamestate.statestr)
        if remain_piece_round < remain_piece:
            remain_piece = remain_piece_round
            peace_round = 0
        else:
            peace_round += 1
        if peace_round > 120:
            winner = 'peace'
            break

    return winner, moves


class IcyPlayer:
    def __init__(self, gpu_core, weight_full_path):
        """
        gpu_core should be None or 1 2 3 etc
        """
        self.network_inuse = resnet.get_model(
            weight_full_path,
            create_uci_labels(),
            gpu_core=[gpu_core],
            filters=conf.TrainingConfig.network_filters,
            num_res_layers=conf.TrainingConfig.network_layers
        )
        self.network_player = None

    def set_up_network_player(
            self,
            player='w',
            is_selfplay=True,
            temp_round=conf.SelfPlayConfig.train_temp_round,
            repeat_noise=True,
            c_puct=conf.TrainingConfig.c_puct,
            dnoise=True,
    ):
        self.network_player = players.NetworkPlayer(
            player,
            self.network_inuse,
            n_playout=conf.EvaluateConfig.val_playout,
            c_puct=c_puct,
            dnoise=dnoise,
            is_selfplay=is_selfplay,
            temp_round=temp_round,
            repeat_noise=repeat_noise,
            play=False
        )

    def restore_model(self, weight_full_path):
        (sess, graph), ((X, training), (net_softmax, value_head)) = self.network_inuse
        with graph.as_default():
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.restore(sess, weight_full_path)

    def get_action(self, state, no_act=None):
        move, score = self.network_player.make_move(state=state, actual_move=True, allow_legacy=True, no_act=no_act)
        return move

    def oppponent_make_move(self, move):
        self.network_player.oppoent_make_move(move, allow_legacy=True)


def nash_battle(gpu_num, nash_battle_models):
    player_bot = None

    latest_model_name = get_latest_weight_path()
    # get latest model
    network_player_main = \
        IcyPlayer(gpu_num, os.path.join(conf.ResourceConfig.model_dir, latest_model_name))

    network_player_battle = None

    for model in nash_battle_models:
        try:
            local_nash_battle_dir = os.path.join(conf.ResourceConfig.nash_battle_local_dir, latest_model_name, model)
            check_if_dir_exit(local_nash_battle_dir)

            battle_num = len(os.listdir(local_nash_battle_dir))
            if battle_num >= conf.EvaluateConfig.nash_each_battle_num:
                continue

            if model == conf.EvaluateConfig.bot_name:
                player2 = CchessPlayer(gpu_num, conf.EvaluateConfig.cchess_playout, infer=True)
                player_bot = player2
                player2_name = conf.EvaluateConfig.bot_name
            else:
                if not network_player_battle:
                    network_player_battle = \
                        IcyPlayer(gpu_num, os.path.join(conf.ResourceConfig.model_dir, model))
                else:
                    network_player_battle.restore_model(os.path.join(conf.ResourceConfig.model_dir, model))
                player2 = network_player_battle
                player2_name = conf.EvaluateConfig.icy_player_name

            player_w = network_player_main
            player_b = player2
            player_w_name = conf.EvaluateConfig.icy_player_name
            player_w_name_for_file = 'new'
            player_b_name = player2_name
            player_b_name_for_file = 'last'
            while battle_num < conf.EvaluateConfig.nash_each_battle_num:
                if random.random() < 0.5:
                    player_w, player_b = player_b, player_w
                    player_w_name, player_b_name = player_b_name, player_w_name
                    player_w_name_for_file, player_b_name_for_file = player_b_name_for_file, player_w_name_for_file

                if player_w_name == conf.EvaluateConfig.icy_player_name:
                    player_w.set_up_network_player('w', temp_round=-1, repeat_noise=False, c_puct=5, dnoise=True,
                                                   is_selfplay=False)
                if player_b_name == conf.EvaluateConfig.icy_player_name:
                    player_b.set_up_network_player('b', temp_round=-1, repeat_noise=False, c_puct=5, dnoise=True,
                                                   is_selfplay=False)

                winner, moves = play_till_end(player_w, player_b, player_w_name, player_b_name)

                stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
                date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
                cbfile = cbf.CBF(
                    black=player_b_name,
                    red=player_w_name,
                    date=date,
                    site='北京',
                    name='noname',
                    datemodify=date,
                    redteam=player_w_name,
                    blackteam=player_b_name,
                    round='第一轮'
                )
                cbfile.receive_moves(moves)

                randstamp = random.randint(0, 1000)
                cbffilename = '{}_{}_mcts-mcts_{}-{}_{}.cbf'.format(
                    stamp, randstamp, player_w_name_for_file, player_b_name_for_file, winner)
                cbfile_path = os.path.join(local_nash_battle_dir, cbffilename)
                cbfile.dump(cbfile_path)

                battle_num = len(os.listdir(local_nash_battle_dir))
                logging.info('{} vs {} battle {} games'.format(latest_model_name, model, battle_num))
        except:
            logging.error('battle model {} error'.format(model))

    logging.info('begin to deleting ai')
    if player_bot:
        player_bot.close()
        del player_bot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="worker self play script")
    parser.add_argument("-gpu_num", type=str, default='0')
    parser.add_argument("-worker_index", type=int, default=0)
    parser.add_argument("-nash_index", type=int, default=0)
    args = parser.parse_args()

    worker_type = 'gpu'
    gpu_num = 0
    if args.gpu_num.lower() == 'none':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        worker_type = 'cpu'
        gpu_num = None
    else:
        gpu_num = int(args.gpu_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    logging.info('worker nash battle gpu_num {}  worker_index {} '.format(args.gpu_num, args.worker_index))

    nash_battle_models = get_nash_battle_list(args)
    nash_battle(gpu_num, nash_battle_models)
