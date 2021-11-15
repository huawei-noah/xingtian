"""
chinese chess single self play thread
self play all the time, every self_play_games_one_time games change models
"""
import os
import argparse
import time
import random
import json
import logging

import tensorflow as tf
import numpy as np

from lib import cbf
from config import conf
from env.cchess_env import create_uci_labels
from agent import resnet, players
from env.game_state import GameState
from lib.utils import get_latest_weight_path
from bot.evaluate.single_evaluate_worker import CchessPlayer


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s][%(message)s]",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def count_piece(state_str):
    pieceset = {
        'A',
        'B',
        'C',
        'K',
        'N',
        'P',
        'R',
        'a',
        'b',
        'c',
        'k',
        'n',
        'p',
        'r'
    }
    return sum([1 for single_chessman in state_str if single_chessman in pieceset])


class Game(object):
    def __init__(self, white, black, white_name, black_name, verbose=True):
        self.white = white
        self.black = black
        self.white_name = white_name
        self.black_name = black_name

        self.verbose = verbose
        self.gamestate = GameState()
        self.total_time = 0
        self.steps = 0

    def play_till_end(self):
        winner = 'peace'
        moves = []
        peace_round = 0
        remain_piece = count_piece(self.gamestate.statestr)
        while True:
            self.steps += 1
            start_time = time.time()

            if self.gamestate.move_number % 2 == 0:
                player_name = self.white_name
                player = self.white
                opponent_player = self.black
                opponent_player_name = self.black_name
            else:
                player_name = self.black_name
                player = self.black
                opponent_player = self.white
                opponent_player_name = self.white_name

            if player_name == "bot":
                red_move = True if self.gamestate.currentplayer == 'w' else False
                move = player.get_action(self.gamestate.statestr, self.gamestate.move_number, red_move=red_move)
                score = None
            else:
                move, score = player.make_move(self.gamestate, actual_move=True, allow_legacy=True)
            if opponent_player_name != "bot":
                opponent_player.oppoent_make_move(move, allow_legacy=True)

            if move is None:
                winner = 'b' if self.gamestate.currentplayer == 'w' else 'w'
                break

            if player_name == "bot":
                self.gamestate.do_move(move)
            moves.append(move)

            loop, winner_if_loop = self.gamestate.long_catch_or_looping()
            if loop:
                winner = winner_if_loop
                break

            # if self.verbose:
            total_time = time.time() - start_time
            self.total_time += total_time

            # logging.info('time average {}'.format(round(self.total_time / self.steps, 2)))
            # logging.info('move {} {} play {} score {} use {:.2f}s pr {} pid {}'.format(
            #     self.gamestate.move_number,
            #     player_name,
            #     move,
            #     score,
            #     total_time,
            #     peace_round,
            #     os.getpid())
            # )

            game_end, winner_p = self.gamestate.game_end()
            if game_end:
                winner = winner_p
                break

            remain_piece_round = count_piece(self.gamestate.statestr)
            if remain_piece_round < remain_piece:
                remain_piece = remain_piece_round
                peace_round = 0
            else:
                peace_round += 1
            if peace_round > conf.SelfPlayConfig.non_cap_draw_round:
                winner = 'peace'
                break
        logging.info('one game done, time average {} game moves {}'.format(
            round(self.total_time / self.steps, 2), len(moves)))
        return winner, moves


class NetworkPlayGame(Game):
    def __init__(self, network_w, network_b, name_w, name_b, **xargs):
        if name_w == 'bot':
            whiteplayer = network_w
        else:
            whiteplayer = players.NetworkPlayer('w', network_w, **xargs)
        if name_b == 'bot':
            blackplayer = network_b
        else:
            blackplayer = players.NetworkPlayer('b', network_b, **xargs)
        super(NetworkPlayGame, self).__init__(whiteplayer, blackplayer, name_w, name_b,)


class SelfPlay(object):
    def __init__(
            self,
            network_w=None,
            network_b=None,
            white_name='net',
            black_name='net',
            random_switch=True,
            recoard_game=True,
            recoard_dir='data/distributed/',
            play_times=np.inf,
            distributed_dir='data/prepare_weight',
            **xargs
    ):
        self.network_w = network_w
        self.network_b = network_b
        self.white_name = white_name
        self.black_name = black_name
        self.random_switch = random_switch
        self.play_times = play_times
        self.recoard_game = recoard_game
        self.recoard_dir = recoard_dir
        self.xargs = xargs
        self.distributed_dir = distributed_dir

    def begin_of_game(self):
        pass

    def end_of_game(self):
        pass

    def play(self):
        num = 0
        while num < self.play_times:
            try:
                num += 1
                self.begin_of_game()
                for i in range(conf.SelfPlayConfig.self_play_games_one_time):
                    if self.random_switch and random.random() < 0.5:
                        self.network_w, self.network_b = self.network_b, self.network_w
                        self.white_name, self.black_name = self.black_name, self.white_name

                    network_play_game = NetworkPlayGame(self.network_w, self.network_b,self.white_name, self.black_name, **self.xargs)
                    winner, moves = network_play_game.play_till_end()

                    stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
                    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
                    cbfile = cbf.CBF(
                        black=self.black_name,
                        red=self.white_name,
                        date=date,
                        site='北京',
                        name='noname',
                        datemodify=date,
                        redteam=self.white_name,
                        blackteam=self.black_name,
                        round='第一轮'
                    )
                    cbfile.receive_moves(moves)

                    randstamp = random.randint(0, 1000)
                    cbffilename = '{}_{}_mcts-mcts_{}-{}_{}.cbf'.format(
                        stamp, randstamp, self.white_name, self.black_name, winner)
                    cbf_name = os.path.join(self.recoard_dir, cbffilename)
                    cbfile.dump(cbf_name)
            except Exception:
                logging.info('one game error')
                pass

        self.end_of_game()


class DistributedSelfPlayGames(SelfPlay):
    def __init__(self, gpu_num=0, **kwargs):
        self.gpu_num = gpu_num
        self.bot_player = CchessPlayer(self.gpu_num, conf.EvaluateConfig.cchess_playout)
        latest_model_name = get_latest_weight_path()
        self.icy_player1 = resnet.get_model(
            os.path.join(conf.ResourceConfig.model_dir, latest_model_name),
            create_uci_labels(),
            gpu_core=[self.gpu_num],
            filters=conf.TrainingConfig.network_filters,
            num_res_layers=conf.TrainingConfig.network_layers
        )
        self.icy_player2 = resnet.get_model(
            os.path.join(conf.ResourceConfig.model_dir, latest_model_name),
            create_uci_labels(),
            gpu_core=[self.gpu_num],
            filters=conf.TrainingConfig.network_filters,
            num_res_layers=conf.TrainingConfig.network_layers
        )
        super(DistributedSelfPlayGames, self).__init__(**kwargs)

    def begin_of_game(self):
        """
        when self playing, init network player using the latest weights
        """
        # get self_play model
        # if no nash_res.json then use latest weight
        # else use nash.npy to choose model

        # get main_model and battle_model
        nash_res_dirs = os.listdir(conf.ResourceConfig.nash_battle_local_dir)
        try:
            nash_res_dirs.sort()

            lastest_nash_res_file = os.path.join(
                conf.ResourceConfig.nash_battle_local_dir, nash_res_dirs[-1], conf.ResourceConfig.nash_res_json)
            with open(lastest_nash_res_file, 'r') as f:
                nash_res = json.load(f)
            [players, q] = nash_res['nash_res']
            main_model = players[q.index(max(q))]

            latest_nash_res_bot_file = os.path.join(
                conf.ResourceConfig.nash_battle_local_dir, nash_res_dirs[-1], conf.ResourceConfig.nash_res_bot_json)
            with open(latest_nash_res_bot_file, 'r') as f:
                nash_res_bot = json.load(f)
            [players_bot, q_bot] = nash_res_bot['nash_res_bot']
            sum_q_bot = sum(q_bot)
            q_new_bot = [_ / sum_q_bot for _ in q_bot]
            battle_model = np.random.choice(players_bot, p=q_new_bot)
        except BaseException:
            logging.error('nash file not found, using latest model and bot for battle')
            latest_model_name = get_latest_weight_path()
            main_model = latest_model_name
            battle_model = 'bot'

        logging.info('restore params main_model {} battle_model {}'.format(main_model, battle_model))
        # restore main_model
        (sess, graph), _ = self.icy_player1
        with graph.as_default():
            saver = tf.train.Saver(var_list=tf.global_variables())
            saver.restore(sess, os.path.join(conf.ResourceConfig.model_dir, main_model))
        self.network_w = self.icy_player1
        self.white_name = 'icy'

        if battle_model == 'bot':
            self.bot_player.reset()
            self.network_b = self.bot_player
            self.black_name = 'bot'
        else:
            (sess, graph), _ = self.icy_player2
            with graph.as_default():
                saver = tf.train.Saver(var_list=tf.global_variables())
                saver.restore(sess, os.path.join(conf.ResourceConfig.model_dir, battle_model))
            self.network_b = self.icy_player2
            self.black_name = 'icy'

    def end_of_game(self):
        self.bot_player.close()
        del self.bot_player


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="worker self play script")
    parser.add_argument("-gpu_num", type=str, default="0")
    args = parser.parse_args()

    worker_type = 'gpu'
    if args.gpu_num.lower() == 'none':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        worker_type = 'cpu'
        gpu_num = None
    else:
        gpu_num = int(args.gpu_num)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    self_play = DistributedSelfPlayGames(
        gpu_num=gpu_num,
        n_playout=conf.SelfPlayConfig.train_playout,
        recoard_dir=conf.ResourceConfig.distributed_datadir,
        c_puct=conf.TrainingConfig.c_puct,
        distributed_dir=conf.ResourceConfig.model_dir,
        dnoise=True,
        is_selfplay=True,
        play_times=10,
    )

    self_play.play()
