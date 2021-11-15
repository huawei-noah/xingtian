import random
import asyncio
from asyncio.queues import Queue
import logging

import numpy as np

from config import conf
from lib.game_convert import boardarr2netinput
from env.cchess_env import create_uci_labels, BaseChessBoard
from env.cchess_env_c import CchessEnvC
from env.cchess_env import CchessEnv
from agent import mcts_async
from collections import namedtuple


try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except IOError:
    logging.info("uvloop not detected, ignoring")
    pass


labels = create_uci_labels()
uci_labels = create_uci_labels()
QueueItem = namedtuple("QueueItem", "feature future")


def flipped_uci_labels(param):
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in param]


class Player(object):
    def __init__(self, side):
        assert(side in ['w', 'b'])
        self.side = side
    
    def make_move(self, state):
        assert(state.currentplayer == self.side)
        pass
    
    def oppoent_make_move(self, single_move):
        pass


class NetworkPlayer(Player):
    def __init__(
            self,
            side,
            network,
            debugging=True,
            n_playout=800,
            search_threads=16,
            virtual_loss=0.02,
            policy_loop_arg=True,
            c_puct=5,
            dnoise=False,
            temp_round=conf.SelfPlayConfig.train_temp_round,
            can_surrender=False,
            surrender_threshold=-0.99,
            allow_legacy=False,
            repeat_noise=True,
            is_selfplay=False,
            play=False,
            ma_service=False
    ):
        super(NetworkPlayer, self).__init__(side)
        self.network = network
        self.debugging = debugging
        loop = None
        if ma_service:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            loop = asyncio.get_event_loop()
        self.queue = Queue(400, loop=loop)
        self.temp_round = temp_round
        self.can_surrender = can_surrender
        self.allow_legacy = allow_legacy
        self.surrender_threshold = surrender_threshold
        self.repeat_noise = repeat_noise
        self.mcts_policy = mcts_async.MCTS(
            self.policy_value_fn_queue,
            n_playout=n_playout,
            search_threads=search_threads,
            virtual_loss=virtual_loss,
            policy_loop_arg=policy_loop_arg,
            c_puct=c_puct,
            dnoise=dnoise,
            play=play,
        )
        self.is_selfplay = is_selfplay
    
    async def push_queue(self, features, loop):
        future = loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    async def prediction_worker(self, mcts_policy_async):
        (sess, graph), ((X, training), (net_softmax, value_head)) = self.network
        q = self.queue
        while mcts_policy_async.num_proceed < mcts_policy_async._n_playout:
            if q.empty():
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]
            features = np.concatenate([item.feature for item in item_list], axis=0)

            action_probs, value = sess.run([net_softmax, value_head], feed_dict={X: features, training: False})
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    async def policy_value_fn_queue(self, state, loop):
        bb = BaseChessBoard(state.statestr)
        statestr = bb.get_board_arr()
        net_x = np.transpose(boardarr2netinput(statestr, state.get_current_player()), [1, 2, 0])
        net_x = np.expand_dims(net_x, 0)
        future = await self.push_queue(net_x, loop)
        await future
        policyout, valout = future.result()
        policyout, valout = policyout, valout[0]
        if conf.SelfPlayConfig.py_env:
            legal_move = CchessEnv.get_legal_moves(state.statestr, state.get_current_player())
        else:
            legal_move = CchessEnvC.get_legal_action(state.statestr, state.get_current_player())
        legal_move = set(legal_move)
        legal_move_b = set(flipped_uci_labels(legal_move))

        action_probs = []
        if state.currentplayer == 'b':
            for single_move, prob in zip(uci_labels, policyout):
                if single_move in legal_move_b:
                    single_move = flipped_uci_labels([single_move])[0]
                    action_probs.append((single_move, prob))
        else:
            for single_move, prob in zip(uci_labels, policyout):
                if single_move in legal_move:
                    action_probs.append((single_move, prob))
        return action_probs, valout

    def get_random_policy(self, policies):
        sumnum = sum([i[1] for i in policies])
        randnum = random.random() * sumnum
        tmp = 0
        for val, pos in policies:
            tmp += pos
            if tmp > randnum:
                return val
    
    def make_move(self, state, actual_move=True, infer_mode=False, allow_legacy=False, no_act=None):
        assert(state.currentplayer == self.side)
        if state.move_number < self.temp_round or (self.repeat_noise and state.maxrepeat > 1):
            temp = 1
        else:
            temp = 1e-4
        if state.move_number >= self.temp_round and self.is_selfplay is True:
            can_apply_dnoise = True
        else:
            can_apply_dnoise = False
        if infer_mode:
            acts, act_probs, info = self.mcts_policy.get_move_probs(
                state,
                temp=temp,
                verbose=False,
                predict_workers=[self.prediction_worker(self.mcts_policy)],
                can_apply_dnoise=can_apply_dnoise,
                infer_mode=infer_mode,
                no_act=no_act
            )
        else:
            acts, act_probs = self.mcts_policy.get_move_probs(
                state,
                temp=temp,
                verbose=False,
                predict_workers=[self.prediction_worker(self.mcts_policy)],
                can_apply_dnoise=can_apply_dnoise,
                no_act=no_act
            )

        # icy resign
        if not acts:
            if infer_mode:
                return None, None, None
            return None, None

        policies, score = list(zip(acts, act_probs)), self.mcts_policy._root._Q
        score = -score
        # 1 means going to win, -1 means going to lose
        if score < self.surrender_threshold and self.can_surrender:
            return None, score
        single_move = self.get_random_policy(policies)
        if actual_move:
            state.do_move(single_move)
            self.mcts_policy.update_with_move(single_move, allow_legacy=allow_legacy)

        if infer_mode:
            return single_move, score, info
        return single_move, score
    
    def oppoent_make_move(self, single_move, allow_legacy=False):
        self.mcts_policy.update_with_move(single_move, allow_legacy=allow_legacy)
