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
import threading
import time
from multiprocessing import Queue, Process

import psutil
import lz4.frame
from pyarrow import deserialize
from absl import logging

from xt.framework.explorer import Explorer
from xt.framework.evaluator import Evaluator
from zeus.common.ipc.comm_conf import CommConf, get_port
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.ipc.share_buffer import ShareBuf
from xt.framework.remoter import dist_model
from zeus.common.ipc.message import message, get_msg_info, set_msg_data, get_msg_data
from zeus.common.util.profile_stats import TimerRecorder
from collections import defaultdict, deque
import numpy as np


class BrokerMaster(object):
    """BrokerMaster Manage Broker within Learner."""

    def __init__(self, node_config_list, start_port=None):
        self.node_config_list = node_config_list
        self.node_num = len(node_config_list)
        comm_conf = None
        if not start_port:
            comm_conf = CommConf()
            start_port = comm_conf.get_start_port()
        self.start_port = start_port
        logging.info("master broker init on port: {}".format(start_port))
        self.comm_conf = comm_conf

        recv_port, send_port = get_port(start_port)
        self.recv_slave = UniComm("CommByZmq", type="PULL", port=recv_port)
        self.send_slave = [
            UniComm("CommByZmq", type="PUSH", port=send_port + i)
            for i in range(self.node_num)
        ]

        self.recv_local_q = UniComm("LocalMsg")
        self.send_local_q = dict()

        self.main_task = None
        self.metric = TimerRecorder("master", maxlen=50, fields=("send", "recv"))

    def start_data_transfer(self):
        """Start transfer data and other thread."""
        data_transfer_thread = threading.Thread(target=self.recv_broker_slave)
        data_transfer_thread.setDaemon(True)
        data_transfer_thread.start()

        data_transfer_thread = threading.Thread(target=self.recv_local)
        data_transfer_thread.setDaemon(True)
        data_transfer_thread.start()

        # alloc_thread = threading.Thread(target=self.alloc_actor)
        # alloc_thread.setDaemon(True)
        # alloc_thread.start()

    def recv_broker_slave(self):
        """Receive remote train data in sync mode."""
        while True:
            recv_data = self.recv_slave.recv_bytes()
            _t0 = time.time()
            recv_data = deserialize(lz4.frame.decompress(recv_data))
            self.metric.append(recv=time.time() - _t0)

            cmd = get_msg_info(recv_data, "cmd")
            if cmd in []:
                pass
            else:
                send_cmd = self.send_local_q.get(cmd)
                if send_cmd:
                    send_cmd.send(recv_data)

            # report log
            self.metric.report_if_need()

    def recv_local(self):
        """Receive local cmd."""
        while True:
            recv_data = self.recv_local_q.recv()
            cmd = get_msg_info(recv_data, "cmd")
            if cmd in ["close"]:
                self.close(recv_data)

            if cmd in [self.send_local_q.keys()]:
                self.send_local_q[cmd].send(recv_data)
                logging.debug("recv: {} with cmd-{}".format(type(recv_data["data"]), cmd))
            else:
                _t1 = time.time()
                broker_id = get_msg_info(recv_data, "broker_id")
                _cmd = get_msg_info(recv_data, "cmd")
                logging.debug("master recv:{} with cmd:'{}' to broker_id: <{}>".format(
                    type(recv_data["data"]), _cmd, broker_id))
                # self.metric.append(debug=time.time() - _t1)

                if broker_id == -1:
                    for slave, node_info in zip(self.send_slave, self.node_config_list):
                        slave.send(recv_data)
                else:
                    self.send_slave[broker_id].send(recv_data)
                self.metric.append(send=time.time() - _t1)

    def register(self, cmd):
        self.send_local_q.update({cmd: UniComm("LocalMsg")})
        return self.send_local_q[cmd]

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
        for q in self.send_slave:
            q.send(alloc_cmd)

    def close(self, close_cmd):
        for slave in self.send_slave:
            slave.send(close_cmd)

        time.sleep(1)
        try:
            self.comm_conf.release_start_port(self.start_port)
        except BaseException:
            pass

        os._exit(0)

    def start(self):
        """Start all system."""
        self.start_data_transfer()

    def main_loop(self):
        """
        Create the main_loop after ready the messy setup works.

        The foreground task of broker master.
        :return:
        """
        if not self.main_task:
            logging.fatal("learning process isn't ready!")
        self.main_task.main_loop()

    def stop(self):
        """Stop all system."""
        close_cmd = message(None, cmd="close")
        self.recv_local_q.send(close_cmd)


class BrokerSlave(object):
    """BrokerSlave manage the Broker within Explorer of each node."""

    def __init__(self, ip_addr, broker_id, start_port):
        self.broker_id = broker_id
        train_port, predict_port = get_port(start_port)

        self.send_master_q = UniComm(
            "CommByZmq", type="PUSH", addr=ip_addr, port=train_port
        )

        self.recv_master_q = UniComm(
            "CommByZmq", type="PULL", addr=ip_addr, port=predict_port + broker_id
        )

        self.recv_explorer_q = UniComm("ShareByPlasma")
        self.send_explorer_q = dict()
        self.explore_process = dict()
        self.processes_suspend = 0
        logging.info("init broker slave with id-{}".format(self.broker_id))
        self._metric = TimerRecorder("broker_slave", maxlen=50, fields=("send",))
        # Note: need check it if add explorer dynamic
        self._buf = ShareBuf(live=0, start=True)

    def start_data_transfer(self):
        """Start transfer data and other thread."""
        data_transfer_thread = threading.Thread(target=self.recv_master)
        data_transfer_thread.start()

        data_transfer_thread = threading.Thread(target=self.recv_explorer)
        data_transfer_thread.start()

    def recv_master(self):
        """Recv remote train data in sync mode."""
        while True:
            # recv, data will deserialize with pyarrow default
            # recv_data = self.recv_master_q.recv()
            recv_bytes = self.recv_master_q.recv_bytes()
            recv_data = deserialize(recv_bytes)

            cmd = get_msg_info(recv_data, "cmd")
            if cmd in ["close"]:
                self.close(recv_data)

            if cmd in ["create_explorer"]:
                config_set = recv_data["data"]
                config_set.update({"share_path": self._buf.get_path()})
                self.create_explorer(config_set)

                # update the buffer live attribute, explorer num as default.
                self._buf.plus_one_live()
                continue
            if cmd in ["create_evaluator"]:
                config_set = recv_data["data"]
                config_set.update({"share_path": self._buf.get_path()})
                logging.debug("create evaluator with config:{}".format(config_set))

                self.create_evaluator(config_set)
                self._buf.plus_one_live()
                continue

            if cmd in ["increase", "decrease"]:
                self.alloc(cmd)
                continue

            if cmd in ("eval",):  # fixme: merge into explore
                test_id = get_msg_info(recv_data, "test_id")
                self.send_explorer_q[test_id].put(recv_data)
                continue

            # last job, distribute weights/model_name from broker master
            explorer_id = get_msg_info(recv_data, "explorer_id")
            if not isinstance(explorer_id, list):
                explorer_id = [explorer_id]

            _t0 = time.time()
            if "explore" in cmd:
                # here, only handle explore weights
                buf_id = self._buf.put(recv_bytes)
                # replace weight with id
                recv_data.update({"data": buf_id})

            # predict_reply

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

    def recv_explorer(self):
        """Recv explorer cmd."""
        while True:
            data, object_info = self.recv_explorer_q.recv_bytes()
            object_id, data_type = object_info
            if data_type == "data":
                self.send_master_q.send_bytes(data)
            elif data_type == "buf_reduce":
                recv_data = deserialize(data)
                to_reduce_id = get_msg_data(recv_data)
                self._buf.reduce_once(to_reduce_id)
            else:
                raise KeyError("un-known data type: {}".format(data_type))
            self.recv_explorer_q.delete(object_id)

    def create_explorer(self, config_info):
        """Create explorer."""
        env_para = config_info.get("env_para")
        env_id = env_para.get("env_id")
        send_explorer = Queue()

        explorer = Explorer(
            config_info,
            self.broker_id,
            recv_broker=send_explorer,
            send_broker=self.recv_explorer_q,
        )

        p = Process(target=explorer.start)
        p.daemon = True
        p.start()

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
            send_broker=self.recv_explorer_q,
        )
        p = Process(target=evaluator.start)
        p.daemon = True
        p.start()

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
        time.sleep(5)

        for _, p in self.explore_process.items():
            if p.exitcode is None:
                p.terminate()

        os.system("pkill plasma -g " + str(os.getpgid(0)))
        os._exit(0)

    def start(self):
        """Start all system."""
        self.start_data_transfer()
