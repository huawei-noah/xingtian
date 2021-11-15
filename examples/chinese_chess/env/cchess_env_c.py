"""
中国象棋环境
主要功能：
1. 获取当前可行动作
2. step
3. 判断游戏是否结束
4. 将 statestr 转换成 plane
"""
import copy
import os

import numpy as np
from enum import IntEnum
from ctypes import *


FULL_INIT_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
y_axis = '9876543210'
x_axis = 'abcdefghi'


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
library = cdll.LoadLibrary(os.path.join(DIR_PATH, "libDemo3.so"))


class CchessEnvC(object):
    def __init__(self):
        self.name = 'a chess env'

    @staticmethod
    def sim_do_action(action, state_str):
        library.sim_do_action.argtypes = [c_char_p, c_char_p, c_char_p]
        library.sim_do_action.restype = c_void_p

        input_action = create_string_buffer(action.encode('utf-8'))
        input_state = create_string_buffer(state_str.encode('utf-8'))

        res_pt = (c_char * 150)()
        library.sim_do_action(input_action, input_state, res_pt)
        return res_pt.value.decode('utf-8')

    @staticmethod
    def is_check_catch(state_str, next_player):
        library.is_check_catch.argtypes = [c_char_p, c_char_p]
        library.is_check_catch.restype = c_int8

        input_player = create_string_buffer(next_player.encode('utf-8'))
        input_state = create_string_buffer(state_str.encode('utf-8'))

        check_catch = library.is_check_catch(input_state, input_player)

        return check_catch

    @staticmethod
    def game_end(state_str, player):
        library.game_end.argtypes = [c_char_p, c_char_p, c_char_p]
        library.game_end.restype = c_int8

        input_player = create_string_buffer(player.encode('utf-8'))
        input_state = create_string_buffer(state_str.encode('utf-8'))

        res_pt = (c_char * 5)()
        over = library.game_end(input_state, input_player, res_pt)
        winner = res_pt.value.decode('utf-8')

        return over, winner

    @staticmethod
    def get_legal_action(state_str, player):
        library.get_legal_action.argtypes = [c_char_p, c_char_p, c_char_p]
        library.get_legal_action.restype = c_void_p

        input_player = create_string_buffer(player.encode('utf-8'))
        input_state = create_string_buffer(state_str.encode('utf-8'))

        res_pt = (c_char * 1500)()
        library.get_legal_action(input_state, input_player, res_pt)
        # print("out_word:", res_pt.value.decode('utf-8'))
        legal_actions = res_pt.value.decode('utf-8')
        legal_actions = legal_actions[1:-1].split('/')

        return legal_actions
