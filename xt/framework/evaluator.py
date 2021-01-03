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
"""Evaluate worker."""

import sys
import setproctitle
from copy import deepcopy
from absl import logging
from xt.framework.agent_group import AgentGroup
from zeus.common.ipc.message import get_msg_data, message, get_msg_info
from zeus.common.util.printer import print_immediately


class Evaluator(object):
    """Setup an evaluator on local node as default."""

    def __init__(self, config_info, broker_id, recv_broker, send_broker):
        self.env_para = deepcopy(config_info.get("env_para"))
        self.alg_para = deepcopy(config_info.get("alg_para"))
        self.agent_para = deepcopy(config_info.get("agent_para"))
        self.test_id = config_info.get("test_id")
        self.bm_eval = config_info.get("benchmark", {}).get("eval", {})
        self.broker_id = broker_id
        self.recv_broker = recv_broker
        self.send_broker = send_broker

        self.best_reward = None

    def start(self):
        """Run evaluator."""
        setproctitle.setproctitle("xt_evaluator")

        _ags = AgentGroup(self.env_para, self.alg_para, self.agent_para, scene="evaluate")
        while True:
            recv_data = self.recv_broker.get()
            cmd = get_msg_info(recv_data, "cmd")
            logging.debug("evaluator get meg: {}".format(type(recv_data)))
            if cmd in ("close", ):
                break

            if cmd not in ["eval"]:
                print_immediately("eval get un-used data:{}".format(recv_data))
                continue

            # print_immediately("recv_data in evaluator: {}".format(
            #     [v.keys() for v in recv_data["data"].values()]))

            for train_count, weights in recv_data["data"].items():
                _ags.restore(weights, is_id=False)
                eval_data = _ags.evaluate(self.bm_eval.get("episodes_per_eval", 1))

                # return each rewards for each agent
                record_item = tuple([eval_data,
                                     {"train_count": train_count,
                                      "broker_id": self.broker_id,
                                      "test_id": self.test_id}])
                print_immediately("collect eval results: {}".format(record_item))
                record_item = message(
                    record_item,
                    cmd="eval_return",
                    broker_id=self.broker_id,
                    test_id=self.test_id,
                )
                self.send_broker.send(record_item)

            # fixme: save weights with evaluated performance.
