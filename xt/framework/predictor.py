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
"""Create Predictor."""
import os
from time import time
from copy import deepcopy
from xt.algorithm import alg_builder
import setproctitle
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.ipc.message import message, get_msg_data, set_msg_info, set_msg_data, get_msg_info
from zeus.common.util.profile_stats import PredictStats, TimerRecorder


class Predictor(object):
    """Predict Worker for async algorithm."""

    def __init__(self, predictor_id, config_info, request_q, reply_q, predictor_name):
        self.config_info = deepcopy(config_info)
        self.predictor_id = predictor_id
        self.request_q = request_q
        self.reply_q = reply_q

        self._report_period = 200

        self._stats = PredictStats()
        self.predictor_name = predictor_name

    def process(self):
        """Predict action."""
        while True:
            start_t0 = time()
            ctr_info, data = self.request_q.recv()
            recv_data = {'ctr_info': ctr_info, 'data': data}
            self._stats.obs_wait_time += time() - start_t0

            cmd = ctr_info.get('sub_cmd', 'predict')
            # if 'predict' in cmd:
            #     cmd = 'predict'
            # else:
            #     print("sync model")

            if cmd in self.process_fn.keys():
                proc_fn = self.process_fn[cmd]
                proc_fn(recv_data)
            else:
                raise KeyError("invalid cmd: {}".format(ctr_info['cmd']))

    def sync_weights(self, recv_data):
        model_weights = recv_data['data']
        self.alg.set_weights(model_weights)

    def predict(self, recv_data):
        start_t1 = time()
        state = get_msg_data(recv_data)
        broker_id = get_msg_info(recv_data, 'broker_id')
        explorer_id = get_msg_info(recv_data, 'explorer_id')
        action = self.alg.predict(state)
        self._stats.inference_time += time() - start_t1

        reply_data = message(action, cmd="predict_reply", broker_id=broker_id,
                             explorer_id=explorer_id)
        self.reply_q.put(reply_data)

        self._stats.iters += 1
        if self._stats.iters > self._report_period:
            _report = self._stats.get()
            reply_data = message(_report, cmd="stats_msg{}".format(self.predictor_name))
            self.reply_q.put(reply_data)

    def start(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
        alg_para = self.config_info.get('alg_para')
        setproctitle.setproctitle("xt_predictor")

        self.alg = alg_builder(**alg_para)

        self.process_fn = {'sync_weights': self.sync_weights,
                           'predict': self.predict}

        #start msg process
        self.process()
