from env.cchess_env import CchessEnv
from env.cchess_env_c import CchessEnvC
from config.conf import SelfPlayConfig


class GameState:
    def __init__(self):
        self.statestr = 'RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr'
        self.currentplayer = 'w'
        self.pastdic = {}
        self.maxrepeat = 0
        self.lastmove = ""
        self.move_number = 0

    def copy_custom(self, state_input):
        self.statestr = state_input.statestr
        self.currentplayer = state_input.currentplayer
        self.pastdic = {}
        for key in state_input.pastdic.keys():
            self.pastdic.setdefault(key, [0, False, self.get_next_player()])
            self.pastdic[key][0] = state_input.pastdic[key][0]
            self.pastdic[key][1] = state_input.pastdic[key][1]
            self.pastdic[key][2] = state_input.pastdic[key][2]
        self.maxrepeat = state_input.maxrepeat
        self.lastmove = state_input.lastmove
        self.move_number = state_input.move_number

    def get_current_player(self):
        return self.currentplayer

    def get_next_player(self):
        return 'w' if self.currentplayer == 'b' else 'b'

    def is_check_catch(self):
        if SelfPlayConfig.py_env:
            return CchessEnv.is_check_catch(self.statestr, self.get_next_player())
        return CchessEnvC.is_check_catch(self.statestr, self.get_next_player())

    def game_end(self):
        if SelfPlayConfig.py_env:
            return CchessEnv.game_end(self.statestr, self.currentplayer)
        return CchessEnvC.game_end(self.statestr, self.currentplayer)

    def do_move(self, move):
        self.lastmove = move
        if SelfPlayConfig.py_env:
            self.statestr = CchessEnv.sim_do_action(move, self.statestr)
        else:
            self.statestr = CchessEnvC.sim_do_action(move, self.statestr)
        self.currentplayer = 'w' if self.currentplayer == 'b' else 'b'

        # times, longcatch/check
        self.pastdic.setdefault(self.statestr, [0, False, self.get_next_player()])
        self.pastdic[self.statestr][0] += 1
        self.pastdic[self.statestr][1] = self.is_check_catch()

        self.move_number += 1
        self.maxrepeat = self.pastdic[self.statestr][0]

    def should_cutoff(self):
        # the pastdic is empty when first move was made
        if self.move_number < 2:
            return False

        if self.pastdic[self.statestr][0] > 1 and self.pastdic[self.statestr][1]:
            return True
        else:
            return False

    def long_catch_or_looping(self):
        if self.pastdic[self.statestr][0] > 1 and self.pastdic[self.statestr][1]:
            # when checking long catch, current player has changed to next player
            return True, self.get_current_player()
        elif self.pastdic[self.statestr][0] > 3:
            return True, 'peace'
        else:
            return False, None
