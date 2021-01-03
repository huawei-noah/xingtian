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
"""Build general utils for algorithm."""

from absl import logging
from collections import deque, defaultdict


def _clip_explorer_id(raw_dist_info, clip_set):
    if not clip_set:
        return raw_dist_info
    elif raw_dist_info["explorer_id"] == -1:
        raw_dist_info["explorer_id"] = clip_set
    else:
        clipped_id = list([_id for _id in raw_dist_info["explorer_id"] if _id in clip_set])
        raw_dist_info["explorer_id"] = clipped_id
    return raw_dist_info


class DefaultAlgDistPolicy(object):
    def __init__(self, actor_num, **kwargs):
        self.actor_num = actor_num
        # message {"broker_id": -1, "explorer_id": -1, "agent_id": -1}
        self.default_policy = {"broker_id": -1, "explorer_id": -1}

    def get_dist_info(self, model_index, explorer_set=None):
        return _clip_explorer_id(self.default_policy, explorer_set)

    def add_processed_ctr_info(self, ctr_info):
        pass


class DivideDistPolicy(DefaultAlgDistPolicy):
    def get_dist_info(self, model_index, explorer_set=None):
        if model_index > -1:
            self.default_policy.update(
                {"explorer_id": model_index % self.actor_num})
        return _clip_explorer_id(self.default_policy, explorer_set)


def _fetch_broker_info(ctr_relation_buf: defaultdict):
    """Fetch broker information."""
    ctr_list = list()
    default_policy = {"broker_id": -1, "explorer_id": -1}
    for _broker, _explorer in ctr_relation_buf.items():
        default_policy.update(
            {"broker_id": _broker, "explorer_id": list(_explorer)})
        ctr_list.append(default_policy.copy())

    return ctr_list


class FIFODistPolicy(DefaultAlgDistPolicy):
    """DESC: distribute to which had submitted explore data."""
    def __init__(self, actor_num, prepare_times, **kwargs):
        super(FIFODistPolicy, self).__init__(actor_num, **kwargs)
        self._processed_agent = deque()
        self.prepare_data_times = prepare_times

    def add_processed_ctr_info(self, ctr_info):
        self._processed_agent.append(ctr_info)

    def get_dist_info(self, model_index, explorer_set=None):
        if model_index < 0:
            return self.default_policy

        ctr_relation_buf = defaultdict(set)
        for _i in range(len(self._processed_agent)):
            try:
                _info = self._processed_agent.popleft()
                # key = (broker_id, explorer_id, agent_id)

                ctr_relation_buf[_info[0]].update((_info[1],))
            except IndexError:
                logging.ERROR("without data in FIFODistPolicy.deque, used last!")

        return _clip_explorer_id(_fetch_broker_info(ctr_relation_buf), explorer_set)


class EqualDistPolicy(DefaultAlgDistPolicy):
    """DESC: distribute to which had submitted explore data."""

    def __init__(self, actor_num, prepare_times, **kwargs):
        super(EqualDistPolicy, self).__init__(actor_num, **kwargs)
        self._processed_agent = defaultdict(int)
        self.prepare_data_times = prepare_times

    def add_processed_ctr_info(self, ctr_info):
        self._processed_agent[ctr_info] += 1

    def get_dist_info(self, model_index, explorer_set=None):
        if model_index < 0:
            return self.default_policy
        ctr_relation_buf = defaultdict(set)
        for _id, _val in self._processed_agent.items():
            if _val >= self.prepare_data_times:  # fixme: check threshold
                self._processed_agent[_id] -= self.prepare_data_times
                ctr_relation_buf[_id[0]].update((_id[1],))

        return _clip_explorer_id(_fetch_broker_info(ctr_relation_buf), explorer_set)
