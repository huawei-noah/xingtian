import random
import copy
import multiprocessing
from multiprocessing import Manager

import numpy as np
import logging

from lib.game_convert import convert_game_value
from env.cchess_env import create_uci_labels


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


class ElePreloader(object):
    def __init__(self, datalist, batch_size=2048, shuffle=True):
        self.batch_size = batch_size
        self.datalist = datalist
        self.pos = 0
        self.feature_list = {
            "red": ['A', 'B', 'C', 'K', 'N', 'P', 'R'],
            "black": ['a', 'b', 'c', 'k', 'n', 'p', 'r']
        }
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.labels = create_uci_labels()
        self.label2ind = dict(zip(self.labels, list(range(len(self.labels)))))

    def load(self, num_process):
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
        states = []
        pre_proc = int(len(self.datalist) / num_process)
        for i in range(num_process):
            if i == num_process-1:
                sub_datalist = self.datalist[pre_proc * i:]
            else:
                sub_datalist = self.datalist[pre_proc*i:pre_proc*(i+1)]
            p = multiprocessing.Process(target=self.worker, args=(sub_datalist, i, return_dict))
            jobs.append(p)
        for proc in jobs:
            proc.start()
        for proc in jobs:
            proc.join()
        for state in return_dict.values():
            states.append(state)
        # only shuffle the first axis
        states = np.concatenate(states, axis=0)
        np.random.shuffle(states)
        states = states[:self.batch_size]
        return np.concatenate(states[:, 0], axis=0), \
               np.concatenate(states[:, 1], axis=0), np.asarray(states[:, 2]), self.batch_size

    def worker(self, datalist, procnum, return_dict):
        states = []
        for one_file in datalist:
            try:
                try:
                    game_data = convert_game_value(one_file, self.feature_list, None)
                except Exception:
                    print('{} error exist'.format(one_file))
                    continue

                if game_data:
                    for x1, y1, val1 in game_data:
                        x1 = np.transpose(x1, [1, 2, 0])
                        x1 = np.expand_dims(x1, axis=0)
                        oney = np.zeros(len(self.labels))
                        oney[self.label2ind[''.join(y1)]] = 1
                        states.append([x1, [oney], val1])
            except FileNotFoundError:
                print(one_file)
                import traceback
                traceback.print_exc()
                continue
        return_dict[procnum] = states
