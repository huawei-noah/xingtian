import os
import random
import time

import tensorflow as tf
import logging

import numpy as np

from env.game_state import GameState
from agent import resnet, players
from env.cchess_env import create_uci_labels
from lib import cbf
from config import conf
from lib.utils import get_latest_weight_path


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s][%(message)s]",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )
running_time = 0
running_step = 0


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
    def __init__(self, white, black, verbose=True):
        self.white = white
        self.black = black
        self.verbose = verbose
        self.gamestate = GameState()
        self.total_time = 0
        self.steps = 0
    
    def play_till_end(self):
        global running_step
        global running_time

        winner = 'peace'
        moves = []
        peace_round = 0
        remain_piece = count_piece(self.gamestate.statestr)
        while True:
            start_time = time.time()
            if self.gamestate.move_number % 2 == 0:
                player_name = 'w'
                player = self.white
                opponent_player = self.black
            else:
                player_name = 'b'
                player = self.black
                opponent_player = self.white
            
            move, score = player.make_move(self.gamestate, allow_legacy=True)
            opponent_player.oppoent_make_move(move, allow_legacy=True)

            if move is None:
                winner = 'b' if player_name == 'w' else 'w'
                break
            moves.append(move)
            # if self.verbose:
            total_time = time.time() - start_time
            self.total_time += total_time
            self.steps += 1
            running_time += total_time
            running_step += 1
            logging.info('time average {}'.format(round(running_time / running_step, 2)))
            logging.info('move {} {} play {} score {} use {:.2f}s pr {} pid {}'.format(
                self.gamestate.move_number,
                player_name,
                move,
                score if player_name == 'w' else -score,
                total_time,
                peace_round,
                os.getpid())
            )
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
        return winner, moves


class NetworkPlayGame(Game):
    def __init__(self, network_w, network_b, **xargs):
        whiteplayer = players.NetworkPlayer('w', network_w, **xargs)
        blackplayer = players.NetworkPlayer('b', network_b, **xargs)
        super(NetworkPlayGame, self).__init__(whiteplayer, blackplayer)


class ContinousNetworkPlayGames(object):
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
        # self.distributed_server = distributed_server
        self.distributed_dir = distributed_dir
    
    def begin_of_game(self):
        pass
    
    def end_of_game(self, cbf_name, moves, cbfile, training_dt, epoch):
        pass
    
    def play(self, data_url=None, epoch=0, yundao_new_data_dir=None):
        num = 0
        while num < self.play_times:
            time_one_game_start = time.time()
            num += 1
            self.begin_of_game()
            if self.random_switch and random.random() < 0.5:
                self.network_w, self.network_b = self.network_b, self.network_w
                self.white_name, self.black_name = self.black_name, self.white_name
                
            network_play_game = NetworkPlayGame(self.network_w, self.network_b, **self.xargs)
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

            if data_url:
                output_game_file_path = os.path.join(data_url, cbffilename)
                cbfile.dump(output_game_file_path)

            if yundao_new_data_dir:
                import moxing as mox
                mox.file.copy(cbf_name, os.path.join(yundao_new_data_dir, cbffilename))

            training_dt = time.time() - time_one_game_start
            self.end_of_game(cbffilename, moves, cbfile, training_dt, epoch)


class DistributedSelfPlayGames(ContinousNetworkPlayGames):
    def __init__(self, gpu_num=0, auto_update=True, **kwargs):
        self.gpu_num = gpu_num
        self.auto_update = auto_update
        self.model_name_in_use = None  # for tracking latest weight
        super(DistributedSelfPlayGames, self).__init__(**kwargs)

    def begin_of_game(self):
        """
        when self playing, init network player using the latest weights
        """
        if not self.auto_update:
            return

        latest_model_name = get_latest_weight_path()
        model_path = os.path.join(self.distributed_dir, latest_model_name)
        if self.network_w is None or self.network_b is None:
            network = resnet.get_model(
                model_path,
                create_uci_labels(),
                gpu_core=[self.gpu_num],
                filters=conf.TrainingConfig.network_filters,
                num_res_layers=conf.TrainingConfig.network_layers
            )
            self.network_w = network
            self.network_b = network
            self.model_name_in_use = model_path
        else:
            if model_path != self.model_name_in_use:
                (sess, graph), ((X, training), (net_softmax, value_head)) = self.network_w
                with graph.as_default():
                    saver = tf.train.Saver(var_list=tf.global_variables())
                    saver.restore(sess, model_path)
                self.model_name_in_use = model_path

    def end_of_game(self, cbf_name, moves, cbfile, training_dt, epoch):
        trained_games = len(os.listdir(conf.ResourceConfig.distributed_datadir))
        logging.info('------------------epoch {}: trained {} games, this game used {}s'.format(
            epoch,
            trained_games,
            round(training_dt, 6),
        ))


class ValidationGames(ContinousNetworkPlayGames):
    pass
