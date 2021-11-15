"""
中国象棋环境
主要功能：
1. 获取当前可行动作
2. step
3. 判断游戏是否结束
4. 将 statestr 转换成 plane
"""
import copy

import numpy as np
from enum import IntEnum
from env.cchess_env_c import CchessEnvC


FULL_INIT_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
y_axis = '9876543210'
x_axis = 'abcdefghi'


class ChessSide(IntEnum):
    RED = 0
    BLACK = 1

    @staticmethod
    def next_side(side):
        return {ChessSide.RED: ChessSide.BLACK, ChessSide.BLACK: ChessSide.RED}[side]


class Pos(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def abs_diff(self, other):
        return abs(self.x - other.x), abs(self.y - other.y)

    def middle(self, other):
        return Pos((self.x + other.x) / 2, (self.y + other.y) / 2)

    def __str__(self):
        return str(self.x) + ":" + str(self.y)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    def __ne__(self, other):
        return (self.x != other.x) or (self.y != other.y)

    def __call__(self):
        return self.x, self.y


class BaseChessBoard(object):
    def __init__(self, fen=None):
        self._board = None
        self.move_side = None
        self.clear()
        self.round = 0
        if fen:
            self.from_fen(fen)

    def clear(self):
        self._board = [[None for x in range(9)] for y in range(10)]
        self.move_side = ChessSide.RED

    def copy(self):
        return copy.deepcopy(self)

    def put_fench(self, fench, pos):
        if self._board[pos.y][pos.x] is not None:
            return False

        self._board[pos.y][pos.x] = fench

        return True

    @staticmethod
    def judge_side(fen_ch):
        return ChessSide.BLACK if fen_ch.islower() else ChessSide.RED

    def is_valid_move(self, pos_from, pos_to):
        """
        只进行最基本的走子规则检查，不对每个子的规则进行检查，以加快文件加载之类的速度
        """
        if not (0 <= pos_to.x <= 8):
            return False
        if not (0 <= pos_to.y <= 9):
            return False

        fench_from = self._board[pos_from.y][pos_from.x]
        if not fench_from:
            return False

        from_side = self.judge_side(fench_from)

        # move_side 不是None值才会进行走子颜色检查，这样处理某些特殊的存储格式时会处理比较迅速
        if self.move_side and (from_side != self.move_side):
            return False

        fench_to = self._board[pos_to.y][pos_to.x]
        if not fench_to:
            return True

        to_side = self.judge_side(fench_to)

        return from_side != to_side

    def _move_piece(self, pos_from, pos_to):

        fench = self._board[pos_from.y][pos_from.x]
        self._board[pos_to.y][pos_to.x] = fench
        self._board[pos_from.y][pos_from.x] = None

        return fench

    def move(self, pos_from, pos_to):
        pos_from.y = 9 - pos_from.y
        pos_to.y = 9 - pos_to.y
        if not self.is_valid_move(pos_from, pos_to):
            return None

        self._move_piece(pos_from, pos_to)

        return 'step_success'

    def from_fen(self, fen):

        num_set = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        ch_set = {'k', 'a', 'b', 'n', 'r', 'c', 'p'}

        self.clear()

        if not fen or fen == '':
            return

        fen = fen.strip()

        x = 0
        y = 9

        for i in range(0, len(fen)):
            ch = fen[i]

            if ch == ' ':
                break
            elif ch == '/':
                y -= 1
                x = 0
                if y < 0:
                    break
            elif ch in num_set:
                x += int(ch)
                if x > 8:
                    x = 8
            elif ch.lower() in ch_set:
                if x <= 8:
                    self.put_fench(ch, Pos(x, y))
                    x += 1
            else:
                return False

        fens = fen.split()

        self.move_side = None
        if (len(fens) >= 2) and (fens[1] == 'b'):
            self.move_side = ChessSide.BLACK
        else:
            self.move_side = ChessSide.RED

        if len(fens) >= 6:
            self.round = int(fens[5])
        else:
            self.round = 1

        return True

    def get_board_arr(self):
        return np.asarray(self._board[::-1])


def create_uci_labels():
    """创建所有合法走子UCI，size 2086"""
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    for l1 in range(9):
        for n1 in range(10):
            destinations = [(t, n1) for t in range(9)] + \
                           [(l1, t) for t in range(10)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(9) and n2 in range(10):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)

    for p in advisor_labels:
        labels_array.append(p)

    for p in bishop_labels:
        labels_array.append(p)

    return labels_array


def create_position_labels():
    """
    ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9',
    'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9',
    'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9',
    'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9',
    'e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9',
    'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
    'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9',
    'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',
    'i0', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[8 - l1] + numbers[n1]
            labels_array.append(move)
    return labels_array


def create_position_labels_reverse():
    """
    ['a9', 'a8', 'a7', 'a6', 'a5', 'a4', 'a3', 'a2', 'a1', 'a0',
    'b9', 'b8', 'b7', 'b6', 'b5', 'b4', 'b3', 'b2', 'b1', 'b0',
    'c9', 'c8', 'c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1', 'c0',
    'd9', 'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1', 'd0',
    'e9', 'e8', 'e7', 'e6', 'e5', 'e4', 'e3', 'e2', 'e1', 'e0',
    'f9', 'f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0',
    'g9', 'g8', 'g7', 'g6', 'g5', 'g4', 'g3', 'g2', 'g1', 'g0',
    'h9', 'h8', 'h7', 'h6', 'h5', 'h4', 'h3', 'h2', 'h1', 'h0',
    'i9', 'i8', 'i7', 'i6', 'i5', 'i4', 'i3', 'i2', 'i1', 'i0']
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[l1] + numbers[n1]
            labels_array.append(move)
    labels_array.reverse()
    return labels_array


class CchessEnv(object):
    """
    board_pos_name
    [['a0' 'b0' 'c0' 'd0' 'e0' 'f0' 'g0' 'h0' 'i0']
     ['a1' 'b1' 'c1' 'd1' 'e1' 'f1' 'g1' 'h1' 'i1']
     ['a2' 'b2' 'c2' 'd2' 'e2' 'f2' 'g2' 'h2' 'i2']
     ['a3' 'b3' 'c3' 'd3' 'e3' 'f3' 'g3' 'h3' 'i3']
     ['a4' 'b4' 'c4' 'd4' 'e4' 'f4' 'g4' 'h4' 'i4']
     ['a5' 'b5' 'c5' 'd5' 'e5' 'f5' 'g5' 'h5' 'i5']
     ['a6' 'b6' 'c6' 'd6' 'e6' 'f6' 'g6' 'h6' 'i6']
     ['a7' 'b7' 'c7' 'd7' 'e7' 'f7' 'g7' 'h7' 'i7']
     ['a8' 'b8' 'c8' 'd8' 'e8' 'f8' 'g8' 'h8' 'i8']
     ['a9' 'b9' 'c9' 'd9' 'e9' 'f9' 'g9' 'h9' 'i9']]
    """
    board_pos_name = np.array(create_position_labels()).reshape(9, 10).transpose()
    Ny = 10
    Nx = 9

    def __init__(self):
        self.name = 'a chess env'

    @staticmethod
    def expand_num(single_line):
        single_line = single_line.replace("2", "11")
        single_line = single_line.replace("3", "111")
        single_line = single_line.replace("4", "1111")
        single_line = single_line.replace("5", "11111")
        single_line = single_line.replace("6", "111111")
        single_line = single_line.replace("7", "1111111")
        single_line = single_line.replace("8", "11111111")
        single_line = single_line.replace("9", "111111111")
        return single_line

    @staticmethod
    def compress_num(single_line):
        single_line = single_line.replace("111111111", "9")
        single_line = single_line.replace("11111111", "8")
        single_line = single_line.replace("1111111", "7")
        single_line = single_line.replace("111111", "6")
        single_line = single_line.replace("11111", "5")
        single_line = single_line.replace("1111", "4")
        single_line = single_line.replace("111", "3")
        single_line = single_line.replace("11", "2")
        return single_line

    @staticmethod
    def sim_do_action(in_action, in_state):
        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        src = in_action[0:2]
        dst = in_action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        # change two line or one line
        board = in_state.split('/')
        if src_y != dst_y:
            board_src_y = board[src_y]
            board_dst_y = board[dst_y]
            board_src_y = list(CchessEnv.expand_num(board_src_y))
            board_dst_y = list(CchessEnv.expand_num(board_dst_y))
            board_dst_y[dst_x] = board_src_y[src_x]
            board_src_y[src_x] = '1'
            board[src_y] = CchessEnv.compress_num(''.join(board_src_y))
            board[dst_y] = CchessEnv.compress_num(''.join(board_dst_y))
        else:
            board_line = board[src_y]
            board_line = list(CchessEnv.expand_num(board_line))
            board_line[dst_x] = board_line[src_x]
            board_line[src_x] = '1'
            board[src_y] = CchessEnv.compress_num(''.join(board_line))
        board = "/".join(board)
        return board

    @staticmethod
    def replace_num(board):
        board = CchessEnv.expand_num(board)
        return board.split("/")

    @staticmethod
    def check_bounds(to_y, to_x):
        if to_y < 0 or to_x < 0:
            return False

        if to_y >= CchessEnv.Ny or to_x >= CchessEnv.Nx:
            return False

        return True

    @staticmethod
    def validate_move(c, upper=True):
        if c.isalpha():
            if upper is True:
                if c.islower():
                    return True
                else:
                    return False
            else:
                if c.isupper():
                    return True
                else:
                    return False
        else:
            return True

    @staticmethod
    def get_king_pos(state_str):
        ys_label = '9876543210'[::-1]
        xs_label = 'abcdefghi'
        board_king = state_str.replace("1", " ")
        board_king = board_king.replace("2", "  ")
        board_king = board_king.replace("3", "   ")
        board_king = board_king.replace("4", "    ")
        board_king = board_king.replace("5", "     ")
        board_king = board_king.replace("6", "      ")
        board_king = board_king.replace("7", "       ")
        board_king = board_king.replace("8", "        ")
        board_king = board_king.replace("9", "         ")
        board_king = board_king.split('/')

        k_big, k = '', ''
        for i in range(3):
            pos = board_king[i].find('K')
            if pos != -1:
                k_big = "{}{}".format(xs_label[pos], ys_label[i])
                break
        for i in range(-1, -4, -1):
            pos = board_king[i].find('k')
            if pos != -1:
                k = "{}{}".format(xs_label[pos], ys_label[i])
                break
        return k_big, k

    @staticmethod
    def is_check_catch(state_str, next_player):
        """
        :param state_str: String, input FEN state str
        :param next_player: String, 'w' or 'b'
        :return : Boolean, 下一步对手是否可以将军
        """
        moveset = CchessEnv.get_legal_moves(state_str, next_player)
        targetset = set([i[-2:] for i in moveset])

        wk, bk = CchessEnv.get_king_pos(state_str)
        targetkingdic = {'b': wk, 'w': bk}
        targ_king = targetkingdic[next_player]
        # TODO add long catch logic
        if targ_king in targetset:
            return True
        else:
            return False

    @staticmethod
    def game_end(state_str, current_player):
        """
        :param state_str: String, input FEN state str
        :param current_player: String, 'w' or 'b'

        :return : (Boolean, String), 游戏是否结束， 胜者
        """
        # TODO add long catch
        if state_str.find('k') == -1:
            return True, 'w'
        elif state_str.find('K') == -1:
            return True, 'b'
        wk, bk = CchessEnv.get_king_pos(state_str)
        target_king_dic = {'b': wk, 'w': bk}
        move_set = CchessEnv.get_legal_moves(state_str, current_player)
        dst_point = [i[-2:] for i in move_set]

        targetset = set(dst_point)

        targ_king = target_king_dic[current_player]
        if targ_king in targetset:
            return True, current_player
        return False, None

    @staticmethod
    def get_legal_moves(state, current_player):
        """
        :param state: string, input FEN state str
        :param current_player: string, 'w' or 'b' current player

        : return legal moves: List
        """

        moves = []
        k_x = None
        k_y = None

        K_x = None
        K_y = None

        face_to_face = False

        board_positions = np.array(CchessEnv.replace_num(state))
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):
                if board_positions[y][x].isalpha():
                    # 黑车
                    if board_positions[y][x] == 'r' and current_player == 'b':
                        # 左右
                        to_y = y
                        for to_x in range(x - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                        for to_x in range(x + 1, CchessEnv.Nx):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                        # 上下
                        to_x = x
                        for to_y in range(y - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                        for to_y in range(y + 1, CchessEnv.Ny):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].isupper():
                                    moves.append(m)
                                break

                            moves.append(m)

                    # 红车
                    elif board_positions[y][x] == 'R' and current_player == 'w':
                        to_y = y
                        for to_x in range(x - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                        for to_x in range(x + 1, CchessEnv.Nx):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                        to_x = x
                        for to_y in range(y - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                        for to_y in range(y + 1, CchessEnv.Ny):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if board_positions[to_y][to_x].isalpha():
                                if board_positions[to_y][to_x].islower():
                                    moves.append(m)
                                break

                            moves.append(m)

                    # 马
                    elif (board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b':
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                to_y = y + 2 * i
                                to_x = x + 1 * j
                                if CchessEnv.check_bounds(to_y, to_x) and \
                                        CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) and \
                                        board_positions[to_y - i][x].isalpha() is False:
                                    moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                                to_y = y + 1 * i
                                to_x = x + 2 * j
                                if CchessEnv.check_bounds(to_y, to_x) and \
                                        CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) and \
                                        board_positions[y][to_x - j].isalpha() is False:
                                    moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                    elif (board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'w':
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                to_y = y + 2 * i
                                to_x = x + 1 * j
                                if CchessEnv.check_bounds(to_y, to_x) and \
                                        CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) and \
                                        board_positions[to_y - i][x].isalpha() is False:
                                    moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                                to_y = y + 1 * i
                                to_x = x + 2 * j
                                if CchessEnv.check_bounds(to_y, to_x) and \
                                        CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) and \
                                        board_positions[y][to_x - j].isalpha() is False:
                                    moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                    # 象
                    elif (board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b':
                        for i in range(-2, 3, 4):
                            to_y = y + i
                            to_x = x + i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) and \
                                    to_y >= 5 and board_positions[y + i // 2][x + i // 2].isalpha() is False:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                            to_y = y + i
                            to_x = x - i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) and \
                                    to_y >= 5 and board_positions[y + i // 2][x - i // 2].isalpha() is False:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                    elif (board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'w':
                        for i in range(-2, 3, 4):
                            to_y = y + i
                            to_x = x + i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) and \
                                    to_y <= 4 and board_positions[y + i // 2][x + i // 2].isalpha() is False:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                            to_y = y + i
                            to_x = x - i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) and \
                                    to_y <= 4 and board_positions[y + i // 2][x - i // 2].isalpha() is False:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                    # 士
                    elif board_positions[y][x] == 'a' and current_player == 'b':
                        for i in range(-1, 3, 2):
                            to_y = y + i
                            to_x = x + i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) and \
                                    to_y >= 7 and 3 <= to_x <= 5:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                            to_y = y + i
                            to_x = x - i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) and \
                                    to_y >= 7 and 3 <= to_x <= 5:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                    elif board_positions[y][x] == 'A' and current_player == 'w':
                        for i in range(-1, 3, 2):
                            to_y = y + i
                            to_x = x + i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) \
                                    and to_y <= 2 and 3 <= to_x <= 5:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                            to_y = y + i
                            to_x = x - i
                            if CchessEnv.check_bounds(to_y, to_x) and \
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) \
                                    and to_y <= 2 and 3 <= to_x <= 5:
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])
                    # 将 帅
                    elif board_positions[y][x] == 'k':
                        k_x = x
                        k_y = y
                        if current_player == 'b':
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    to_y = y + i * sign
                                    to_x = x + j * sign

                                    if CchessEnv.check_bounds(to_y, to_x) and \
                                            CchessEnv.validate_move(board_positions[to_y][to_x], upper=False) \
                                            and to_y >= 7 and 3 <= to_x <= 5:
                                        moves.append(
                                            CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                                        )
                    elif board_positions[y][x] == 'K':
                        K_x = x
                        K_y = y
                        if current_player == 'w':
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    to_y = y + i * sign
                                    to_x = x + j * sign

                                    if CchessEnv.check_bounds(to_y, to_x) and \
                                            CchessEnv.validate_move(board_positions[to_y][to_x], upper=True) and \
                                            to_y <= 2 and 3 <= to_x <= 5:
                                        moves.append(
                                            CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                                        )
                    # 炮
                    elif board_positions[y][x] == 'c' and current_player == 'b':
                        to_y = y
                        hits = False  # 可不可以架炮
                        for to_x in range(x - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].isupper():
                                        moves.append(m)
                                    break

                        hits = False
                        for to_x in range(x + 1, CchessEnv.Nx):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].isupper():
                                        moves.append(m)
                                    break

                        to_x = x
                        hits = False
                        for to_y in range(y - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].isupper():
                                        moves.append(m)
                                    break

                        hits = False
                        for to_y in range(y + 1, CchessEnv.Ny):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].isupper():
                                        moves.append(m)
                                    break
                    elif board_positions[y][x] == 'C' and current_player == 'w':
                        to_y = y
                        hits = False
                        for to_x in range(x - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].islower():
                                        moves.append(m)
                                    break

                        hits = False
                        for to_x in range(x + 1, CchessEnv.Nx):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].islower():
                                        moves.append(m)
                                    break

                        to_x = x
                        hits = False
                        for to_y in range(y - 1, -1, -1):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].islower():
                                        moves.append(m)
                                    break

                        hits = False
                        for to_y in range(y + 1, CchessEnv.Ny):
                            m = CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x]
                            if hits is False:
                                if board_positions[to_y][to_x].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[to_y][to_x].isalpha():
                                    if board_positions[to_y][to_x].islower():
                                        moves.append(m)
                                    break
                    # 兵
                    elif board_positions[y][x] == 'p' and current_player == 'b':
                        to_y = y - 1
                        to_x = x

                        if (CchessEnv.check_bounds(to_y, to_x) and
                                CchessEnv.validate_move(board_positions[to_y][to_x], upper=False)):
                            moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                        if y < 5:
                            to_y = y
                            to_x = x + 1
                            if (CchessEnv.check_bounds(to_y, to_x) and
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=False)):
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                            to_x = x - 1
                            if (CchessEnv.check_bounds(to_y, to_x) and
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=False)):
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                    elif board_positions[y][x] == 'P' and current_player == 'w':
                        to_y = y + 1
                        to_x = x

                        if (CchessEnv.check_bounds(to_y, to_x) and
                                CchessEnv.validate_move(board_positions[to_y][to_x], upper=True)):
                            moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                        if y > 4:
                            to_y = y
                            to_x = x + 1
                            if (CchessEnv.check_bounds(to_y, to_x) and
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=True)):
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

                            to_x = x - 1
                            if (CchessEnv.check_bounds(to_y, to_x) and
                                    CchessEnv.validate_move(board_positions[to_y][to_x], upper=True)):
                                moves.append(CchessEnv.board_pos_name[y][x] + CchessEnv.board_pos_name[to_y][to_x])

        if K_x is not None and k_x is not None and K_x == k_x:
            face_to_face = True
            for i in range(K_y + 1, k_y, 1):
                if board_positions[i][K_x].isalpha():
                    face_to_face = False

        if face_to_face is True:
            if current_player == 'b':
                moves.append(CchessEnv.board_pos_name[k_y][k_x] + CchessEnv.board_pos_name[K_y][K_x])
            else:
                moves.append(CchessEnv.board_pos_name[K_y][K_x] + CchessEnv.board_pos_name[k_y][k_x])

        return moves


if __name__ == '__main__':
    import time
    state_str_test = "RNBA1ABNR/4K4/1C5C1/P1PP2P1P/9/9/p1pp2p1p/1c5c1/4k4/rnba1abnr"
    time_start = time.time()
    for try_time in range(10000):
        CchessEnv.get_legal_moves(state_str_test, "w")
    print("py get_legal_moves:", (time.time() - time_start)*1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnvC.get_legal_action(state_str_test, "w")
    print("C get_legal_moves:", (time.time() - time_start) * 1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnv.sim_do_action("a0a1", state_str_test)
    print("py sim_do_action:", (time.time() - time_start) * 1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnvC.sim_do_action("a0a1", state_str_test)
    print("c sim_do_action:", (time.time() - time_start) * 1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnv.is_check_catch(state_str_test, "w")
    print("py is_check_catch:", (time.time() - time_start)*1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnvC.is_check_catch(state_str_test, "w")
    print("c is_check_catch:", (time.time() - time_start) * 1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnv.game_end(state_str_test, "w")
    print("py game_end:", (time.time() - time_start)*1000, " ms")

    time_start = time.time()
    for try_time in range(10000):
        CchessEnvC.game_end(state_str_test, "w")
    print("c game_end:", (time.time() - time_start) * 1000, " ms")
