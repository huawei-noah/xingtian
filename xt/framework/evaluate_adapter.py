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

import csv
import threading
from collections import deque, defaultdict
from copy import deepcopy
from queue import Queue, Empty
from absl import logging
import numpy as np
from zeus.common.ipc.message import message, get_msg_data
from zeus.common.util.default_xt import XtBenchmarkConf as XBConf


class TesterManager(object):
    """Manage evaluate data."""

    def __init__(self, config_info, controller, s3_result_path=None):
        self.config_info = config_info
        self.s3_path = s3_result_path
        self.processed_model_count = 0

        self.record_station_buf = dict()
        self.eval_rewards = deque([], maxlen=50)

        # todo: the registered queue need a attribute about the register's name
        self.recv_broker = controller.register("eval_return", "send")
        self.send_broker = controller.register("to_eval", "recv")

        self.eval_info_queue = Queue()

        eval_info = self.config_info.get("benchmark", {}).get("eval", dict())
        self.max_instance = eval_info.get("evaluator_num", 2)
        self.eval_interval = eval_info.get(
            "gap", XBConf.default_train_interval_per_eval)
        self.max_step_per_episode = eval_info.get(
            "max_step_per_episode", 18000)

        self.used_node = dict()
        self.avail_node = list(((i, "test{}".format(tid))
                                for i in range(controller.node_num)
                                for tid in range(self.max_instance)))

        self.last_eval_index = -9999

    def check_finish_stat(self, target_model_count):
        """Check finish status."""
        return self.processed_model_count >= target_model_count

    def if_eval(self, train_count):
        if train_count - self.last_eval_index < self.eval_interval:
            return False
        return True

    def to_eval(self, weights, train_count, actual_step,
                elapsed_time, train_reward, train_loss):
        # update current evaluate index
        self.last_eval_index = train_count

        train_info = {
            "train_index": train_count,
            "sample_step": actual_step,
            "elapsed_sec": elapsed_time,  # time.time() - abs_start,
            "train_reward": train_reward,
            "loss": np.nan,
        }
        if type(train_loss) in (float, np.float64, np.float32, np.float16, np.float):
            train_info.update({"loss": train_loss})

        self.append_eval_queue(train_count, train_info)

        self.put_test_model({train_count: weights})
        # return self.get_avail_node()

    def append_eval_queue(self, train_id, train_info):
        """Append current train info into eval queue."""
        self.record_station_buf.update({train_id: train_info})

    @property
    def eval_result_empty(self):
        return self.eval_info_queue.empty()

    @property
    def processed_count(self):
        return self.processed_model_count

    def fetch_eval_result(self):
        """Fetch eval results with no wait."""
        ret = list()
        while True:
            try:
                item = self.eval_info_queue.get_nowait()
                ret.append(item)
            except Empty as err:
                break
        return ret

    def _parse_eval_result_and_archive(self, eval_result):
        _model_index = eval_result[-1]["train_count"]  # model receive is a list
        # fixme: only a model been test in an agent.
        _agent_id = list(eval_result[0].keys())[0]

        # find the eval model info, and update the eval reward
        if _model_index in self.record_station_buf:
            eval_info_dict = self.record_station_buf.pop(_model_index)
        else:
            eval_info_dict = dict()
            # print("-->", eval_result)
            # return  # test.api will use print replace write db record
        eval_info_dict.update(
            {
                "agent_id": _agent_id,
                "eval_episode_reward": np.nanmean(eval_result[0][_agent_id]["epi_reward"]),
                "eval_step_reward": np.nanmean(eval_result[0][_agent_id]["step_reward"]),
                "eval_name": _model_index,
            }
        )
        # custom evaluate
        for _key in ("custom_criteria", "battle_won"):
            if _key not in eval_result[0][_agent_id].keys():
                continue
            eval_info_dict.update(
                {_key: np.nanmean(eval_result[0][_agent_id][_key])})

        self.eval_info_queue.put(eval_info_dict)

    def recv_result(self):
        """Recieve test result."""
        while True:
            recv_data = self.recv_broker.recv()
            result_data = get_msg_data(recv_data)
            self.processed_model_count += 1
            logging.debug("result_data: \n{}".format(result_data))

            _meta = result_data[1]
            self.used_node[(_meta["broker_id"], _meta["test_id"])] -= 1

            self._parse_eval_result_and_archive(result_data)

    def put_test_model(self, model_weights):
        """Send test model."""
        key = self.get_avail_node()
        ctr_info = {"cmd": "eval", "broker_id": key[0], "test_id": key[1]}

        eval_cmd = message(model_weights, **ctr_info)
        self.send_broker.send(eval_cmd)
        logging.debug("put evaluate model: {}".format(type(model_weights)))
        self.used_node[key] += 1

    def send_create_evaluator_msg(self, broker_id, test_id):
        """Create evaluator."""
        config = deepcopy(self.config_info)
        config.update({"test_id": test_id})

        create_cmd = message(config, cmd="create_evaluator", broker_id=broker_id)
        self.send_broker.send(create_cmd)

    def get_avail_node(self):
        """Get available test node."""
        if self.used_node:
            min_key = min(self.used_node, key=self.used_node.get)
            if self.used_node.get(min_key) < 1 or not self.avail_node:
                return min_key

        # create new evaluator
        new_key = self.avail_node.pop(0)
        self.send_create_evaluator_msg(*new_key)  # (broker_id, test_id)
        self.used_node.update({new_key: 0})
        logging.info("create evaluator: {}".format(new_key))
        return new_key

    def start(self):
        t = threading.Thread(target=self.recv_result)
        t.start()


class EvalResultSummary(object):
    """Analysis the result within divided test case."""
    def __init__(self, divide_time, target_ids, output_csv="./eval_results.csv"):
        self._divided_time = divide_time

        self._must_id = "train_id"
        if not isinstance(target_ids, list):
            self._ids = [target_ids]
        else:
            self._ids = target_ids

        if self._must_id not in self._ids:
            self._ids.append(self._must_id)

        self._buf = dict()

        # {"train_id@divided_id"}
        # -- >
        # {"train_id": {0, 1, 2, 3,...}}
        self._info = defaultdict(set)

        self._csv = output_csv
        self._csv_open = open(self._csv, "w+")
        self._record_writer = csv.DictWriter(self._csv_open, fieldnames=self._ids)
        self._record_writer.writeheader()

        self.processed_train_ids = list()

    def append(self, eval_results):
        """Append eval results to buf."""
        if not eval_results:
            return

        for _result in eval_results:
            self._buf[_result["eval_name"]] = _result

            train_id, eval_id = str(_result["eval_name"]).split("@")
            self._info[train_id].add(eval_id)

    def _train_id_end(self, train_id):
        return len(self._info[train_id]) == self._divided_time

    def _analysis(self, train_id):
        ret = {_k: list() for _k in self._ids}

        for _eval_id in self._info[train_id]:
            eval_name = "{}@{}".format(train_id, _eval_id)
            eval_result = self._buf.pop(eval_name)

            assert eval_name == eval_result["eval_name"], "{} vs {} mismatch".format(
                eval_name, eval_result["eval_name"]
            )

            for _k in ret:
                if _k not in eval_result:
                    continue

                ret[_k].append(eval_result[_k])

        ret_sum = {_rk: np.nanmean(ret[_rk]) if ret[_rk] else 0. for _rk in ret}
        ret_sum.update({"train_id": train_id})

        return ret_sum

    def check_and_archive(self):
        """Check and archive api."""
        deal_id = list()
        single_record = dict()
        for k in self._info:
            if not self._train_id_end(k):
                continue

            single_record = self._analysis(k)
            self._record_writer.writerow(single_record)
            deal_id.append(k)

        self._csv_open.flush()

        for k in deal_id:
            self._info.pop(k)

        self.processed_train_ids.extend(deal_id)
        return single_record

    @property
    def processed_ids(self):
        """Check have processed train ids."""
        return self.processed_train_ids

    def check_finish_stats(self, target_count):
        """Check finish stats."""
        # print(len(self.processed_ids))
        return len(self.processed_ids) >= target_count

    def close(self):
        """Close file."""
        self._csv_open.close()

    def __del__(self):
        self._csv_open.close()
