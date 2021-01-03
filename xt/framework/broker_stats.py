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
"""Broker status cover learner status and logging to front desk."""

import time
from absl import logging


class BrokerStats(object):
    """Broker states."""
    def __init__(self, timeout=60*5):
        """Broker status for record whole tasks have been submit."""
        # {"task_name", statsRecorder()}
        self.tasks = dict()

        # {"task.name: msg_deliver}
        self.msg_delivers = dict()
        self.relation_task = list()

        self.timeout = timeout

        self._acc_sleep_time = 0
        self._noop_t = 0.1

    def _reset_acc_wait(self):
        self._acc_sleep_time = 0

    def _acc_wait_time(self):
        self._acc_sleep_time += self._noop_t
        if self._acc_sleep_time > self.timeout:
            return True
        return False

    def add_stats_recorder(self, task_name, recorder):
        """Add one stats recorder."""
        self.tasks.update({task_name: recorder})
        self.msg_delivers.update({task_name: recorder.msg_deliver})

    def add_relation_task(self, task):
        """Record task in broker."""
        self.relation_task.append(task)

    def _all_task_join(self):
        return all([not t.isAlive() for t in self.relation_task])

    def _yield_stats(self):
        """Yield a stats, when its data ready."""
        while True:
            # polling with whole learner with no wait.
            for task_name, recv_q in self.msg_delivers.items():
                recv_data = recv_q.recv(block=False)
                if recv_data:
                    # self._reset_acc_wait()
                    yield task_name, recv_data
                else:
                    time.sleep(self._noop_t)
                    # if self._acc_wait_time():
                    #     yield None, None

                    if self._all_task_join():
                        yield None, None

            time.sleep(0.5)

    def loop(self):
        """Run the Stats update loop."""
        while True:
            to_end = False
            task_name, stats_data = next(self._yield_stats())
            if (not task_name) and (not stats_data):
                logging.info("broker status timeout!")
                break
            if self._all_task_join():
                to_end = True
                # process last stats.

            recorder = self.tasks[task_name]
            recorder.process_stats(stats_data)

            if to_end:
                break
