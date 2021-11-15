"""
    Author: Yang Li, (modified by Kun Xiong)
    Logs: 04.29.2021 create the file to calculate nash equilibrium.
"""
import os
import copy

import logging
import numpy as np
import pandas as pd

from config import conf


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


class NashEq:
    def __init__(self):
        self.__name = 'cal_Nash_equilibrium'

    @staticmethod
    def cal_rate(battle_dir_path):
        """
        calculate the winner prob matrix
        Args:
            battle_dir_path 单个nash battle对局
            例如：
            --- model_1
                --- model_0 （battle_dir_path的位置）
                    last_vs_new_w.cbf  (其中 new指model_1， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_1， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_1， old指model_0)

        Return:
            player1_win_rate ： [0,1)
            player2_win_rate : [0,1) peace is treated as the player 2 winning
        """
        logging.info('cal rate {}'.format(battle_dir_path))

        chess_plays = os.listdir(battle_dir_path)
        player1, player2 = 'new', 'last'
        player1_score = 0
        player2_score = 0

        for one_play in chess_plays:
            one_play = one_play.replace('.cbf', '')
            winner = one_play.split('_')[-1]
            w_player = one_play.split('_')[-2].split('-')[0]
            if winner == 'peace':
                player2_score += 0.5
                player1_score += 0.5
            elif winner == 'w':
                if w_player == player1:
                    player1_score += 1
                else:
                    player2_score += 1
            elif winner == 'b':
                if w_player == player1:
                    player2_score += 1
                else:
                    player1_score += 1
        player1_win_rate = player1_score / len(chess_plays)
        player2_win_rate = player2_score / len(chess_plays)
        return player1_win_rate, player2_win_rate

    @staticmethod
    def cal_nash_eq(rate_tables):
        """
        next: calculate the nash equilibrium of column
        Args:
            rate_tables: tables
        """
        a_table = np.zeros_like(rate_tables, dtype='float')
        b_table = np.zeros_like(rate_tables, dtype='float')
        for i in range(rate_tables.shape[0]):
            for j in range(rate_tables.shape[1]):
                if i != j:
                    a_table[i][j] = (float(rate_tables[i][j])-0.5)*2
                    b_table[i][j] = -float(rate_tables[i][j])

        import nashpy as nash
        game = nash.Game(a_table, b_table)
        iterations = 10000
        np.random.seed(0)
        etha = 0.1
        epsilon_bar = 10 ** -1
        play_counts = tuple(game.stochastic_fictitious_play(iterations=iterations, etha=etha, epsilon_bar=epsilon_bar))
        prb_row = [
            row_play_counts / np.sum(row_play_counts) if np.sum(row_play_counts) != 0 else row_play_counts + 1 / len(
                row_play_counts) for (row_play_counts, col_play_counts), _ in play_counts][-1]
        prb_col = [
            col_play_counts / np.sum(col_play_counts) if np.sum(col_play_counts) != 0 else col_play_counts + 1 / len(
                col_play_counts) for (row_play_counts, col_play_counts), _ in play_counts][-1]

        return prb_row, prb_col

    def rpp(
            self,
            last_rate_table_path,
            model_pool_list,
            eliminate_model_name,
            latest_model_name,
            local_nash_battle_dir
    ):
        """
        This part is created by YangLi to calculate the relative population performance rpp
        """
        df = pd.read_csv(last_rate_table_path, encoding='utf-8')
        eliminate_model_index = -1
        last_rate_table = np.array(df)
        for i in range(last_rate_table.shape[0]):
            if last_rate_table[i, 0] == eliminate_model_name:
                eliminate_model_index = i
                break
        assert eliminate_model_index != -1, "can't find eliminate_model in last model rate table"
        last_rate_table = last_rate_table[:, 1:]
        for i in range(last_rate_table.shape[0]):
            for j in range(last_rate_table.shape[1]):
                if i != j:
                    last_rate_table[i][j] = (last_rate_table[i][j]-0.5)*2

        model_pool_list.append(eliminate_model_name)
        model_pool_list.sort()

        latest_vs_others_table = []
        player1 = latest_model_name
        player1_path = os.path.join(local_nash_battle_dir, player1)
        for player2 in model_pool_list:
            battle_dir_path = os.path.join(player1_path, player2)
            if os.path.isdir(battle_dir_path):
                # player1 is always the new player
                player1_win_rate, player2_win_rate = self.cal_rate(battle_dir_path)
                latest_vs_others_table.append(player1_win_rate)
            else:
                logging.info('{}'.format(battle_dir_path))
        # 21
        model_pool_size = last_rate_table.shape[0]
        rate_tables = [[0 for _ in range(model_pool_size)] for _ in range(model_pool_size)]
        for i in range(model_pool_size):
            for j in range(model_pool_size):
                if j < model_pool_size-1:
                    if j < eliminate_model_index:
                        rate_tables[i][j] = last_rate_table[i][j]
                    else:
                        rate_tables[i][j] = last_rate_table[i][j+1]
                else:
                    # first convert to the row vision,then to -1~1
                    rate_tables[i][j] = ((1-latest_vs_others_table[i])-0.5)*2
        rate_tables = np.array(rate_tables)
        rate_tables = - rate_tables.T  # row is the new population, column is the last population
        rate_table_pd = pd.DataFrame(data=rate_tables)
        rate_tables = np.array(rate_tables)
        row_nash_q, col_nash_q = self.cal_nash_eq(rate_tables)
        rpp = np.matmul(row_nash_q.T, rate_tables)
        rpp = np.matmul(rpp, col_nash_q)
        return row_nash_q, col_nash_q, rate_tables, rpp, rate_table_pd

    def build_rate_table(self, model_pool_list, latest_model_name, local_nash_battle_dir):
        """
        build and save win rate table and return the nash equilibrium
        Args:
            model_pool_list: nash 对战模型列表， 按时间排序，最新的在最后
            latest_model_name
            local_nash_battle_dir: nash对战文件夹列表
            一个文件夹以模型名称为名， 内置所有与此模型对战的对局
            例如：
            --- model_0
            --- model_1
                --- model_0
                    last_vs_new_w.cbf  (其中 new指model_1， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_1， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_1， old指model_0)
                    ...
            --- model_2
                --- model_0
                    last_vs_new_w.cbf  (其中 new指model_2， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_2， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_2， old指model_0)
                    ...
                --- model_1
                    last_vs_new_w.cbf  (其中 new指model_2， old指model_1)
                    last_vs_new_w.cbf  (其中 new指model_2， old指model_1)
                    last_vs_new_w.cbf  (其中 new指model_2， old指model_1)
                    ...
            --- model_3
                --- model_0
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_0)
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_0)
                    ...
                --- model_1
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_1)
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_1)
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_1)
                    ...
                --- model_2
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_2)
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_2)
                    last_vs_new_w.cbf  (其中 new指model_3， old指model_2)
                    ...
        Return:
            the prob distribution of each model at nash equilibrium
        """
        logging.info('----------------- cal nash')
        battel_models = copy.deepcopy(model_pool_list)
        battel_models.extend([latest_model_name])
        battel_models.sort(reverse=True)
        table = {}
        for model_index in range(len(battel_models) - 1):
            player1 = battel_models[model_index]
            player1_path = os.path.join(local_nash_battle_dir, player1)
            for player2 in battel_models[model_index+1:]:
                battle_dir_path = os.path.join(player1_path, player2)
                if os.path.isdir(battle_dir_path):
                    # player1 is always the new player
                    player1_win_rate, player2_win_rate = self.cal_rate(battle_dir_path)
                    table[player1 + '_vs_' + player2] = player1_win_rate
                    table[player2 + '_vs_' + player1] = player2_win_rate

        logging.info('win rate table: {}'.format(table))

        players = copy.deepcopy(battel_models)
        players.sort()
        rate_tables = [[0 for _ in range((len(players) + 1))] for _ in range((len(players) + 1))]
        for i in range(1, len(players) + 1):
            rate_tables[0][i] = players[i - 1]
            rate_tables[i][0] = players[i - 1]

        for row in range(len(players) + 1):
            for col in range(len(players) + 1):
                if 0 < row != col > 0:
                    name = rate_tables[row][0] + '_vs_' + rate_tables[0][col]
                    rate_tables[row][col] = table[name]

        rate_table_pd = pd.DataFrame(data=rate_tables)
        rate_tables = np.array(rate_tables)
        rate_tables = rate_tables[1:, 1:]
        q, _ = self.cal_nash_eq(rate_tables)

        nash_res = [players, q]

        # add bot nash
        battel_models.sort()
        players_with_bot = copy.deepcopy(battel_models)
        players_with_bot.append(conf.EvaluateConfig.bot_name)
        win_rate_table_with_bot = copy.deepcopy(table)
        for single_model in battel_models:
            player1 = single_model
            player2 = conf.EvaluateConfig.bot_name
            battle_dir_path = os.path.join(local_nash_battle_dir, player1, player2)
            if os.path.exists(battle_dir_path) and os.path.isdir(battle_dir_path):
                player1_win_rate, player2_win_rate = self.cal_rate(battle_dir_path)
                win_rate_table_with_bot[player1 + '_vs_' + player2] = player1_win_rate
                win_rate_table_with_bot[player2 + '_vs_' + player1] = player2_win_rate

        rate_table_bot_len = len(players_with_bot) + 1
        rate_tables_with_bot = [[0 for _ in range(rate_table_bot_len)] for _ in range(rate_table_bot_len)]
        for i in range(1, len(players_with_bot) + 1):
            rate_tables_with_bot[0][i] = players_with_bot[i - 1]
            rate_tables_with_bot[i][0] = players_with_bot[i - 1]

        for row in range(len(players_with_bot) + 1):
            for col in range(len(players_with_bot) + 1):
                if 0 < row != col > 0:
                    name = rate_tables_with_bot[row][0] + '_vs_' + rate_tables_with_bot[0][col]
                    rate_tables_with_bot[row][col] = win_rate_table_with_bot[name]

        rate_table_bot_pd = pd.DataFrame(data=rate_tables_with_bot)
        rate_tables_with_bot = np.array(rate_tables_with_bot)
        rate_tables_with_bot = rate_tables_with_bot[1:, 1:]
        q_with_bot, _ = self.cal_nash_eq(rate_tables_with_bot)
        nash_res_with_bot = [players_with_bot, q_with_bot]

        logging.info('cal nash eq res, \nnash_res {}, \nrate_table_pd {} \nrate_table_bot_pd {}'.format(
            nash_res, rate_table_pd, rate_table_bot_pd))

        return nash_res, rate_table_pd, nash_res_with_bot, rate_table_bot_pd
