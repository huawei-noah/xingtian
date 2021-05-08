# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Broker setup the message tunnel between learner and explorer."""
import os
import tracemalloc
import threading
import time
import ast
from multiprocessing import Queue, Process, Manager

import psutil
import lz4.frame
import setproctitle
from pyarrow import deserialize, serialize
from absl import logging
import pprint
from collections import defaultdict
from xt.framework.explorer import Explorer
from xt.framework.evaluator import Evaluator
from xt.framework.broker_stats import BrokerStats
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.ipc.share_buffer import ShareBuf
from zeus.common.ipc.message import message, get_msg_info, set_msg_data, get_msg_data
from zeus.common.util.profile_stats import TimerRecorder, show_memory_stats
from zeus.common.util.printer import debug_within_interval
from zeus.common.util.default_xt import DebugConf
from zeus.common.util.get_xt_config import init_main_broker_debug_kwargs, get_pbt_set


class Controller(object):
    """Controller Manage Broker within Learner."""

    def __init__(self, node_config_list):
        self.node_config_list = node_config_list
        self.node_num = len(node_config_list)

        self.recv_broker = UniComm("CommByZmq", type="PULL")
        self.send_broker = [
            UniComm("CommByZmq", type="PUSH") for _i in range(self.node_num)
        ]

        self.port_info = {
            "recv": ast.literal_eval(self.recv_broker.info),
            "send": [ast.literal_eval(_s.info) for _s in self.send_broker]}

        port_info_h = pprint.pformat(self.port_info, indent=0, width=1, )
        logging.info("Init Broker server info:\n{}\n".format(port_info_h))

        self.recv_local_q = dict()  # UniComm("LocalMsg")
        self.send_local_q = dict()

        self.data_manager = Manager()
        self._data_store = dict()

        self._main_task = list()
        self.metric = TimerRecorder("Controller", maxlen=50, fields=("send", "recv"))
        self.stats = BrokerStats()

        if DebugConf.trace:
            tracemalloc.start()

    def start_data_transfer(self):
        """Start transfer data and other thread."""
        data_transfer_thread = threading.Thread(target=self.recv_broker_task)
        data_transfer_thread.setDaemon(True)
        data_transfer_thread.start()

        data_transfer_thread = threading.Thread(target=self.recv_local)
        data_transfer_thread.setDaemon(True)
        data_transfer_thread.start()

        # alloc_thread = threading.Thread(target=self.alloc_actor)
        # alloc_thread.setDaemon(True)
        # alloc_thread.start()

    @property
    def tasks(self):
        return self._main_task

    def recv_broker_task(self):
        """Receive remote train data in sync mode."""
        while True:
            ctr_info, recv_data = self.recv_broker.recv_bytes()
            _t0 = time.time()
            ctr_info = deserialize(ctr_info)
            compress_flag = ctr_info.get('compress_flag', False)
            if compress_flag:
                recv_data = lz4.frame.decompress(recv_data)
            recv_data = deserialize(recv_data)
            recv_data = {'ctr_info': ctr_info, 'data': recv_data}
            self.metric.append(recv=time.time() - _t0)

            cmd = get_msg_info(recv_data, "cmd")
            send_cmd = self.send_local_q.get(cmd)
            if send_cmd:
                send_cmd.send(recv_data)
            else:
                logging.warning("invalid cmd: {}, with date: {}".format(
                    cmd, recv_data))

            # report log
            self.metric.report_if_need(field_sets=("send", "recv"))

    def _yield_local_msg(self):
        """Yield local msg received."""
        while True:
            # polling with whole learner with no wait.
            for cmd, recv_q in self.recv_local_q.items():
                if 'predict' in cmd:
                    try:
                        recv_data = recv_q.get(block=False)
                    except:
                        recv_data = None
                else:
                    recv_data = recv_q.recv(block=False)
                if recv_data:
                    yield recv_data
                else:
                    time.sleep(0.002)

    def recv_local(self):
        """Receive local cmd."""
        # split the case between single receive queue and pbt
        if len(self.recv_local_q) == 1:
            single_stub, = self.recv_local_q.values()
        else:
            single_stub = None

        kwargs = init_main_broker_debug_kwargs()
        yield_func = self._yield_local_msg()

        while True:
            if single_stub:
                recv_data = single_stub.recv(block=True)
            else:
                recv_data = next(yield_func)

            cmd = get_msg_info(recv_data, "cmd")
            if cmd in ["close"]:
                self.close(recv_data)
                break

            if cmd in self.send_local_q.keys():
                # print(self.send_local_q.keys())
                self.send_local_q[cmd].send(recv_data)
                # logging.debug("recv: {} with cmd-{}".format(
                #     recv_data["data"], cmd))
            else:
                _t1 = time.time()
                broker_id = get_msg_info(recv_data, "broker_id")
                _cmd = get_msg_info(recv_data, "cmd")
                # logging.debug("ctr recv:{} with cmd:'{}' to broker_id: <{}>".format(
                #     type(recv_data["data"]), _cmd, broker_id))
                # self.metric.append(debug=time.time() - _t1)

                if broker_id == -1:
                    for broker_item, node_info in zip(self.send_broker, self.node_config_list):
                        broker_item.send(recv_data['ctr_info'], recv_data['data'])
                else:
                    self.send_broker[broker_id].send(recv_data['ctr_info'], recv_data['data'])
                self.metric.append(send=time.time() - _t1)
                debug_within_interval(**kwargs)

    def add_task(self, learner_obj):
        """Add learner task into Broker."""
        self._main_task.append(learner_obj)

    def register(self, cmd, direction, comm_q=None):
        """Register cmd vary with type.

        :param cmd: name to register.
        :type cmd: str
         :param direction: direction relate to broker.
        :type direction: str
        :return:  stub of the local queue with registered.
        :rtype: option
        """
        if not comm_q:
            comm_q = UniComm("LocalMsg")

        if direction == "send":
            self.send_local_q.update({cmd: comm_q})
            return self.send_local_q[cmd]
        elif direction == "recv":
            self.recv_local_q.update({cmd: comm_q})
            return self.recv_local_q[cmd]
        elif direction == "store":
            self._data_store.update({cmd: self.data_manager.dict()})
            return self._data_store[cmd]
        else:
            raise KeyError("invalid register key: {}".format(direction))

    def alloc_actor(self):
        while True:
            time.sleep(10)
            if not self.send_local_q.get("train"):
                continue

            train_list = self.send_local_q["train"].comm.data_list
            if len(train_list) > 200:
                self.send_alloc_msg("decrease")
            elif len(train_list) < 10:
                self.send_alloc_msg("increase")

    def send_alloc_msg(self, actor_status):
        alloc_cmd = {
            "ctr_info": {"cmd": actor_status, "actor_id": -1, "explorer_id": -1}
        }
        for q in self.send_broker:
            q.send(alloc_cmd['ctr_info'], alloc_cmd['data'])

    def close(self, close_cmd):
        for broker_item in self.send_broker:
            broker_item.send(close_cmd['ctr_info'], close_cmd['data'])

        # close ctx may mismatch the socket, use the os.exit last.
        # self.recv_broker.close()
        # for _send_stub in self.send_broker:
        #     _send_stub.close()

    def start(self):
        """Start all system."""
        setproctitle.setproctitle("xt_main")
        self.start_data_transfer()

    def tasks_loop(self):
        """
        Create the tasks_loop after ready the messy setup works.

        The foreground task of Controller.
        :return:
        """
        if not self._main_task:
            logging.fatal("Without learning process ready!")

        train_thread = [threading.Thread(target=task.main_loop) for task in self.tasks]
        for task in train_thread:
            task.start()
            self.stats.add_relation_task(task)

        # check broker stats.
        self.stats.loop()

        # wait to job end.
        for task in train_thread:
            task.join()

    def stop(self):
        """Stop all system."""
        close_cmd = message(None, cmd="close")
        for _learner_id, recv_q in self.recv_local_q.items():
            if 'predict' in _learner_id:
                continue
            else:
                recv_q.send(close_cmd)
        time.sleep(0.1)


class Broker(object):
    """Broker manage the Broker within Explorer of each node."""

    def __init__(self, ip_addr, broker_id, push_port, pull_port):
        self.broker_id = broker_id
        self.send_controller_q = UniComm(
            "CommByZmq", type="PUSH", addr=ip_addr, port=push_port
        )

        self.recv_controller_q = UniComm(
            "CommByZmq", type="PULL", addr=ip_addr, port=pull_port
        )

        self.recv_explorer_q_ready = False
        # record the information between explorer and learner
        #     {"learner_id": UniComm("ShareByPlasma")}
        # add {"default_eval": UniComm("ShareByPlasma")}
        self.explorer_share_qs = {"EVAL0": None}

        # {"recv_id": receive_count}  --> {("recv_id", "explorer_id"): count}
        self.explorer_stats = defaultdict(int)

        self.send_explorer_q = dict()
        self.explore_process = dict()
        self.processes_suspend = 0
        logging.info("init broker with id-{}".format(self.broker_id))
        self._metric = TimerRecorder("broker", maxlen=50, fields=("send",))
        # Note: need check it if add explorer dynamic
        # buf size vary with env_num&algorithm
        # ~4M, impala atari model
        self._buf = ShareBuf(live=0, size=400000000, max_keep=94, start=True)

    def start_data_transfer(self):
        """Start transfer data and other thread."""
        data_transfer_thread = threading.Thread(target=self.recv_controller_task)
        data_transfer_thread.start()

        data_transfer_thread = threading.Thread(target=self.recv_explorer)
        data_transfer_thread.start()

    def _setup_share_qs_firstly(self, config_info):
        """Setup only once time."""
        if self.recv_explorer_q_ready:
            return

        _use_pbt, pbt_size, env_num, _ = get_pbt_set(config_info)
        plasma_size = config_info.get("plasma_size", 100000000)

        for i in range(pbt_size):
            plasma_path = "/tmp/plasma{}T{}".format(os.getpid(), i)
            self.explorer_share_qs["T{}".format(i)] = UniComm("ShareByPlasma",
                                                              size=plasma_size,
                                                              path=plasma_path)

            # print("broker pid:", os.getpid())
            # self.explorer_share_qs["T{}".format(i)].comm.connect()

        # if pbt, eval process will share single server
        # else, share with T0
        if not _use_pbt:
            self.explorer_share_qs["EVAL0"] = self.explorer_share_qs["T0"]
        else:  # use pbt, server will been set within create_evaluator
            self.explorer_share_qs["EVAL0"] = None

        self.recv_explorer_q_ready = True

    def recv_controller_task(self):
        """Recv remote train data in sync mode."""
        while True:
            # recv, data will deserialize with pyarrow default
            # recv_data = self.recv_controller_q.recv()
            ctr_info, data = self.recv_controller_q.recv_bytes()
            recv_data = {'ctr_info': deserialize(ctr_info), 'data': deserialize(data)}

            cmd = get_msg_info(recv_data, "cmd")
            if cmd in ["close"]:
                self.close(recv_data)

            if cmd in ["create_explorer"]:
                config_set = recv_data["data"]
                # setup plasma only one times!
                self._setup_share_qs_firstly(config_set)

                config_set.update({"share_path": self._buf.get_path()})
                self.create_explorer(config_set)

                # update the buffer live attribute, explorer num as default.
                # self._buf.plus_one_live()
                continue
            if cmd in ["create_evaluator"]:
                # evaluator share single plasma.
                config_set = recv_data["data"]

                if not self.explorer_share_qs["EVAL0"]:
                    use_pbt, _, _, _ = get_pbt_set(config_set)
                    if use_pbt:
                        plasma_size = config_set.get("plasma_size", 100000000)
                        plasma_path = "/tmp/plasma{}EVAL0".format(os.getpid())
                        self.explorer_share_qs["EVAL0"] = UniComm("ShareByPlasma",
                                                                  size=plasma_size,
                                                                  path=plasma_path)

                config_set.update({"share_path": self._buf.get_path()})
                logging.debug("create evaluator with config:{}".format(config_set))

                self.create_evaluator(config_set)
                # self._buf.plus_one_live()
                continue

            if cmd in ["increase", "decrease"]:
                self.alloc(cmd)
                continue

            if cmd in ("eval",):  # fixme: merge into explore
                test_id = get_msg_info(recv_data, "test_id")
                self.send_explorer_q[test_id].put(recv_data)
                continue

            # last job, distribute weights/model_name from controller
            explorer_id = get_msg_info(recv_data, "explorer_id")
            if not isinstance(explorer_id, list):
                explorer_id = [explorer_id]

            _t0 = time.time()

            if cmd in ("explore", ):  # todo: could mv to first priority.
                # here, only handle explore weights
                buf_id = self._buf.put(data)
                # replace weight with id
                recv_data.update({"data": buf_id})
            # predict_reply
            # e.g, {'ctr_info': {'broker_id': 0, 'explorer_id': 4, 'agent_id': -1,
            # 'cmd': 'predict_reply'}, 'data': 0}

            for _eid in explorer_id:
                if _eid > -1:
                    self.send_explorer_q[_eid].put(recv_data)
                elif _eid > -2:  # -1 # whole explorer, contains evaluator!
                    for qid, send_q in self.send_explorer_q.items():
                        if isinstance(qid, str) and "test" in qid:
                            # logging.info("continue test: ", qid, send_q)
                            continue

                        send_q.put(recv_data)
                else:
                    raise KeyError("invalid explore id: {}".format(_eid))

            self._metric.append(send=time.time() - _t0)
            self._metric.report_if_need()

    @staticmethod
    def _handle_data(ctr_info, data, explorer_stub, broker_stub):
        object_id = ctr_info['object_id']
        ctr_info_data = ctr_info['ctr_info_data']
        broker_stub.send_bytes(ctr_info_data, data)
        explorer_stub.delete(object_id)

    def _step_explorer_msg(self, use_single_stub):
        """Yield local msg received."""

        if use_single_stub:
            # whole explorer share single plasma
            recv_id = "T0"
            single_stub = self.explorer_share_qs[recv_id]
        else:
            single_stub, recv_id = None, None

        while True:
            if use_single_stub:
                ctr_info, data = single_stub.recv_bytes(block=True)
                self._handle_data(ctr_info, data, single_stub, self.send_controller_q)
                yield recv_id, ctr_info
            else:
                # polling with whole learner with no wait.
                for recv_id, recv_q in self.explorer_share_qs.items():
                    if not recv_q:  # handle eval dummy q
                        continue

                    ctr_info, data = recv_q.recv_bytes(block=False)
                    if not ctr_info:  # this receive q without ready!
                        time.sleep(0.002)
                        continue

                    self._handle_data(ctr_info, data, recv_q, self.send_controller_q)
                    yield recv_id, ctr_info

    def recv_explorer(self):
        """Recv explorer cmd."""
        while not self.recv_explorer_q_ready:
            # wait explorer share buffer ready
            time.sleep(0.05)
        # whole explorer share single plasma
        use_single_flag = True if len(self.explorer_share_qs) == 2 else False
        yield_func = self._step_explorer_msg(use_single_flag)

        while True:
            recv_id, _info = next(yield_func)
            _id = stats_id(_info)
            self.explorer_stats[_id] += 1
            debug_within_interval(logs=dict(self.explorer_stats),
                                  interval=DebugConf.interval_s, human_able=True)

    def create_explorer(self, config_info):
        """Create explorer."""
        env_para = config_info.get("env_para")
        env_num = config_info.get("env_num")
        speedup = config_info.get("speedup", True)
        start_core = config_info.get("start_core", 1)
        env_id = env_para.get("env_id")  # used for explorer id.

        ref_learner_id = config_info.get("learner_postfix")

        send_explorer = Queue()
        explorer = Explorer(
            config_info,
            self.broker_id,
            recv_broker=send_explorer,
            send_broker=self.explorer_share_qs[ref_learner_id],
        )

        p = Process(target=explorer.start)
        p.start()

        cpu_count = psutil.cpu_count()
        if speedup and cpu_count > (env_num + start_core):
            _p = psutil.Process(p.pid)
            _p.cpu_affinity([start_core + env_id])

        self.send_explorer_q.update({env_id: send_explorer})
        self.explore_process.update({env_id: p})

    def create_evaluator(self, config_info):
        """Create evaluator."""
        test_id = config_info.get("test_id")
        send_evaluator = Queue()

        evaluator = Evaluator(
            config_info,
            self.broker_id,
            recv_broker=send_evaluator,
            send_broker=self.explorer_share_qs["EVAL0"],
        )
        p = Process(target=evaluator.start)
        p.start()

        speedup = config_info.get("speedup", False)
        start_core = config_info.get("start_core", 1)
        eval_num = config_info.get("benchmark", {}).get("eval", {}).get("evaluator_num", 1)
        env_num = config_info.get("env_num")

        core_set = env_num + start_core
        cpu_count = psutil.cpu_count()
        if speedup and cpu_count > (env_num + eval_num + start_core):
            _p = psutil.Process(p.pid)
            _p.cpu_affinity([core_set])
            core_set += 1

        self.send_explorer_q.update({test_id: send_evaluator})
        self.explore_process.update({test_id: p})

    def alloc(self, actor_status):
        """Monitor system and adjust resource."""
        p_id = [_p.pid for _, _p in self.explore_process.items()]
        p = [psutil.Process(_pid) for _pid in p_id]

        if actor_status == "decrease":
            if self.processes_suspend < len(p):
                p[self.processes_suspend].suspend()
                self.processes_suspend += 1
        elif actor_status == "increase":
            if self.processes_suspend >= 1:
                p[self.processes_suspend - 1].resume()
                self.processes_suspend -= 1
            else:
                pass
        elif actor_status == "reset":
            # resume all processes suspend
            for _, resume_p in enumerate(p):
                resume_p.resume()

    def close(self, close_cmd):
        """Close broker."""
        for _, send_q in self.send_explorer_q.items():
            send_q.put(close_cmd)
        time.sleep(2)

        for _, p in self.explore_process.items():
            if p.exitcode is None:
                p.terminate()

        # self.send_controller_q.close()
        # self.recv_controller_q.close()

        os.system("pkill plasma -g " + str(os.getpgid(0)))
        os._exit(0)

    def start(self):
        """Start all system."""
        setproctitle.setproctitle("xt_broker")
        self.start_data_transfer()


def stats_id(ctr_info):
    """Assemble the id for record stats information."""
    return "B{}E{}{}".format(ctr_info["broker_id"], ctr_info["explorer_id"],
                             ctr_info["cmd"])
