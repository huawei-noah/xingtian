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
import os
import signal
import threading
from copy import deepcopy
from absl import logging
import setproctitle
from zeus.common.ipc.share_buffer import ShareBuf
from xt.framework.agent_group import AgentGroup
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.ipc.message import message, get_msg_info, get_msg_data, set_msg_info
from zeus.common.util.logger import set_logging_format

set_logging_format()


class Explorer(object):
    """Create an explorer to explore environment to generate train data."""

    def __init__(self, config_info, broker_id, recv_broker, send_broker):
        self.env_para = deepcopy(config_info.get("env_para"))
        self.alg_para = deepcopy(config_info.get("alg_para"))
        self.agent_para = deepcopy(config_info.get("agent_para"))
        self.recv_broker = recv_broker
        self.send_broker = send_broker
        self.recv_agent = UniComm("LocalMsg")
        self.send_agent = UniComm("LocalMsg")
        self.explorer_id = self.env_para.get("env_id")
        self.broker_id = broker_id
        self.learner_postfix = config_info.get("learner_postfix")
        self.rl_agent = None
        self.report_stats_interval = max(config_info.get('env_num'), 7)

        self._buf_path = config_info["share_path"]
        self._buf = ShareBuf(live=10, path=self._buf_path)  # live para is dummy

        logging.info("init explorer with id: {}, buf_path: {}".format(self.explorer_id, self._buf_path))

    def start_explore(self):
        """Start explore process."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
        explored_times = 0

        try:
            self.rl_agent = AgentGroup(
                self.env_para,
                self.alg_para,
                self.agent_para,
                self.send_agent,
                self.recv_agent,
                self._buf
            )
            explore_time = self.agent_para.get("agent_config", {}).get("sync_model_interval", 1)
            logging.info("explorer-{} start with sync interval-{}".format(
                self.explorer_id, explore_time))

            while True:
                model_type = self.rl_agent.update_model()
                stats = self.rl_agent.explore(explore_time)

                explored_times += explore_time
                if explored_times % self.report_stats_interval == self.explorer_id \
                        or explored_times == explore_time:
                    stats_msg = message(stats, cmd="stats_msg",
                                        broker_id=self.broker_id,
                                        explorer_id=self.explorer_id)
                    self.recv_agent.send(stats_msg)
                    if self.explorer_id < 1:
                        logging.debug("EXP{} ran {} ts, restore {} ts, last type:{}".format(
                            self.explorer_id, explored_times, self.rl_agent.restore_count, model_type))

        except BaseException as ex:
            logging.exception(ex)
            os._exit(4)

    def start_data_transfer(self):
        """Start transfer data and other thread."""
        data_transfer_thread = threading.Thread(target=self.transfer_to_broker)
        data_transfer_thread.start()

        data_transfer_thread = threading.Thread(target=self.transfer_to_agent)
        data_transfer_thread.start()

    def transfer_to_agent(self):
        """Send train data to learner."""
        while True:
            data = self.recv_broker.get()
            cmd = get_msg_info(data, "cmd")
            if cmd == "close":
                logging.debug("enter explore close")
                self.close()
                continue

            data = get_msg_data(data)
            self.send_agent.send(data)

    def transfer_to_broker(self):
        """Send train data to learner."""
        while True:
            data = self.recv_agent.recv()
            info_cmd = get_msg_info(data, "cmd")

            new_cmd = info_cmd + self.learner_postfix
            set_msg_info(data, broker_id=self.broker_id,
                         explorer_id=self.explorer_id, cmd=new_cmd)

            self.send_broker.send(data)

    def start(self):
        """Start actor's thread and process."""
        setproctitle.setproctitle("xt_explorer")

        self.start_data_transfer()
        self.start_explore()

    def close(self):
        self.rl_agent.close()


def setup_explorer(controller_recv_stub, config_info, env_id):
    config = deepcopy(config_info)
    config["env_para"].update({"env_id": env_id})

    msg = message(config, cmd="create_explorer")
    controller_recv_stub.send(msg)
