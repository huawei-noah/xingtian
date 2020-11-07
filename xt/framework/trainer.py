import os
import threading
import time
from copy import deepcopy
from multiprocessing import Process, Manager, Event, Queue
from multiprocessing.sharedctypes import RawArray
from operator import mul
from functools import reduce
import numpy as np

from xt.algorithm import alg_builder

os.environ["KERAS_BACKEND"] = "tensorflow"


def build_alg_with_trainer(alg_para, model_q, model_path, process_num):
    """Build an algorithm instance with multi-process trainer."""
    alg_para = deepcopy(alg_para)
    if process_num >= 2:
        shared_list_for_train = Manager().list()
        alg, subprocess_instance = start_multi_processes(
            alg_para, model_q, model_path, process_num, shared_list_for_train,
        )
    else:
        alg = alg_builder(**alg_para)
        subprocess_instance = None
        shared_list_for_train = None

    return alg, subprocess_instance, shared_list_for_train


def start_multi_processes(alg_para, model_q, model_path, process_num, train_list):
    """Start multi processes to train."""
    array_list = init_memory(process_num)
    event_dict = {}
    grad_q = Queue()
    for i in range(process_num):
        event_dict[i] = Event()
    weight_list = init_memory(1)

    grad_process = [
        Process(target=grad_communicate, args=(
            alg_para, i, array_list, weight_list, train_list, event_dict, grad_q))
        for i in range(process_num)
    ]

    for p in grad_process:
        p.start()
        time.sleep(0.1)

    alg = alg_builder(**alg_para)
    train_main_thread = threading.Thread(target=train_main,
                                         args=(alg, array_list, weight_list,
                                               process_num, model_q, model_path, event_dict, grad_q))
    train_main_thread.start()

    return alg, grad_process


def train_main(alg, array_list, weight_list, process_num, model_q, model_path, event_dict, grad_q):
    """Create main process."""
    train_count = 0
    save_count = 0

    weights = alg.actor.get_weights()

    memory_weight = np.frombuffer(weight_list[0])
    shape, length = get_shape_and_length(weights)
    grad_memory_list = []
    for j in range(len(array_list)):
        grad_memory_list.append(np.frombuffer(array_list[j]))
    print("start main process")

    while True:
        time.sleep(0.001)
        weights = alg.actor.get_weights()
        put_list_to_memory(weights, memory_weight, length, shape)

        for j in range(len(array_list)):
            event_dict[j].set()

        grads = []
        for _ in range(process_num):
            id_ = grad_q.get()
            grad_per_process = []
            get_list_from_memory(grad_per_process, grad_memory_list[id_], length, shape)
            grads.append(grad_per_process)

        grad = []
        for i in range(len(grads[0])):
            grad.append(sum([grads[_][i] for _ in range(process_num)]) / process_num)
        alg.apply_grad(grad)

        train_count += 1

        if train_count % 10 == 0:
            model_name = alg.save(model_path, save_count)
            full_model_name = [os.path.join(model_path, i) for i in model_name]

            model_q.put(full_model_name)
            save_count += 1


def grad_communicate(alg_para, id_, array_list, weight_list, train_list, event_dict, grad_q):
    """Create subprocess."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id_ % 4)
    print("start ", id_)

    alg = alg_builder(**alg_para)
    process_num = len(array_list)

    tmp = alg.actor.get_weights()
    shape, length = get_shape_and_length(tmp)

    memory_weight = np.frombuffer(weight_list[0])
    memory_grad = np.frombuffer(array_list[id_])
    trainrecord_ = TrainRecord(id_)
    while True:
        time.sleep(0.001)
        if len(train_list) >= 4 * process_num:
            for _ in range(4):
                data = train_list.pop(0)
                trainrecord_.record_reward(data)
                alg.prepare_data(data)

            event_dict[id_].wait()
            weight = []
            get_list_from_memory(weight, memory_weight, length, shape)
            alg.actor.set_weights(weight)
            event_dict[id_].clear()

            grad = alg.get_grad()
            put_list_to_memory(grad, memory_grad, length, shape)

            # grad_state[id_].value = 1
            grad_q.put(id_)


def init_memory(process_num):
    """Initilize shared memory."""
    array_list = []
    for i in range(process_num):
        array_list.append(RawArray('d', 100000000))
    return array_list


def get_arr_from_memory(memory, start, length, shape):
    memroy_array_part = memory[start: start + length]
    memroy_array_part.shape = shape
    return memroy_array_part


def put_arr_to_memory(memory, start, length, shape, array):
    memroy_array_part = get_arr_from_memory(memory, start, length, shape)
    memroy_array_part[:] = array


def put_list_to_memory(grad, memory_grad, shape_mul, shape):
    start = 0
    for i, grad_data in enumerate(grad):
        put_arr_to_memory(memory_grad, start, shape_mul[i], shape[i], grad_data)
        start += shape_mul[i]


def get_list_from_memory(list_to_fill, arr, shape_mul, shape):
    start = 0
    for i in range(len(shape_mul)):
        memroy_array_part = get_arr_from_memory(arr, start, shape_mul[i], shape[i])
        start += shape_mul[i]
        list_to_fill.append(memroy_array_part)


def get_shape_and_length(array):
    shape, length = [], []
    for _, array_data in enumerate(array):
        weight_shape = array_data.shape
        shape.append(weight_shape)
        weight_length = reduce(mul, weight_shape)
        length.append(weight_length)
    return shape, length


class TrainRecord(object):
    """Record and output reward."""

    def __init__(self, trainer_id):
        self.rewards = []
        self.actor_reward = dict()
        self.actual_step = 0
        self.id = trainer_id

    def record_reward(self, train_data):
        key = train_data[0]
        train_data = train_data[1]
        if type(train_data) is not list:
            train_data = [train_data]

        if key not in self.actor_reward:
            self.actor_reward[key] = 0.

        for data in train_data:
            reward = data[2]
            done = data[4]
            self.actual_step += 1
            self.actor_reward[key] += reward

            if done:
                self.rewards.append(self.actor_reward[key])
                self.actor_reward[key] = 0.
                if len(self.rewards) % 20 == 0:
                    print(self.id, "mean reward is ", np.mean(self.rewards[-50:]), "total step is ", self.actual_step)
