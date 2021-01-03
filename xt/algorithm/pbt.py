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
"""Population based Training Algorithm, PBT for short."""

import math
import random
import copy
from absl import logging
from multiprocessing import Manager

from xt.algorithm import alg_builder


class PbtInfo(object):
    """Information for pbt.

    e.g,
    metric_val = {
        "learner0": {
                "episodic_reward_mean": 10,  # reward could update from msg_stats.
                "elapsed_step": 10000,
                "end": False,
                "checkpoint": False,
                "hyper_params": {
                    "hp_lr": 0.00001,
                }
        },
        "learner1": {
                "episodic_reward_mean": 10,
                "elapsed_step": 10000},
    }
    """

    manager = Manager()

    learner_ids = manager.list()
    metric = manager.dict()
    weights = manager.dict()

    def _update_metric(self, learner_id, values):
        """Update value."""
        self.metric.update({learner_id: values})

    def _update_weights(self, learner_id, weight):
        """Update weight."""
        self.weights.update({learner_id: weight})

    def update(self, learner_id, metric, weight, **kwargs):
        """Update pbt info."""
        self.metric.update({learner_id: metric})
        self.weights.update({learner_id: weight})


class PbtAid(object):
    """PBT aid will help to calculate the explore and exploit."""
    def __init__(self, learner_id, alg_para, config, metric_stub, weight_stub, **kwargs):
        self._lid = learner_id

        # raw alg_para, used to build new algorithm.
        self._alg_para = copy.deepcopy(alg_para)
        self._config = config

        self._interval = config.get("pbt_interval", 50000)
        self._metric_key = config.get("metric_key", "episodic_reward_mean")
        self._metric_type = config.get("metric_type", "max")
        self._last_ready_step = 0
        self.metric_stub = metric_stub
        self.weight_stub = weight_stub

        # history step of each organism, exclude current exploit
        self._previous_acc_episode = 0
        self._max_episode = config.get("complete_episode")
        self._resample_prob = config.get("resample_probability", 0.25)
        self._top_rate = config.get("top_rate", 0.2)
        self._perturb_delta = config.get("perturb_factor_delta", 0.2)

        # save the hyper_param, after each perturb
        self._hyperpara_mutations = config.get("hyperparameters_mutations", dict())
        self._mutation_key = "hyper_params"
        self._setup_metric()

    def _setup_metric(self):
        """Only support model config to mutation"""
        # metric first key, params on second.
        _vars = copy.deepcopy(self._alg_para["model_info"]["actor"]["model_config"])
        mutation_vars = {k: v for k, v in _vars.items() if k in self._hyperpara_mutations}

        raw_metric = {
            "episodic_reward_mean": -9999.,  # reward could update from msg_stats.
            "elapsed_step": 0,
            "end": False,
            "checkpoint": False,
            self._mutation_key: mutation_vars,
        }

        self.metric_stub.update({self._lid: raw_metric})
        self.weight_stub.update({self._lid: {}})

    def meet_stop(self, cur_episode_index):
        """Need stop the population."""
        if self._max_episode and \
                self._previous_acc_episode + cur_episode_index > self._max_episode:
            return True

        return False

    def update_self_metric(self, metric):
        """Update self info into population."""
        metric_handler = self.metric_stub[self._lid]
        metric_handler.update(metric)
        self.metric_stub[self._lid] = metric_handler

        # print(type(weight), weight.keys())
        # self.weight_stub[self._lid] = dict(weight)
        # print("after update: ", self.metric_stub[self._lid])

    def _update_self_weight(self, weight):
        weight_handler = self.weight_stub[self._lid]
        weight_handler.update(dict(weight))
        self.weight_stub[self._lid] = weight_handler
        self._set_checkpoint_bit()

    def _set_checkpoint_bit(self):
        metric_handler = self.metric_stub[self._lid]
        metric_handler.update({"checkpoint": True})
        self.metric_stub[self._lid] = metric_handler

    def _unset_checkpoint_bit(self):
        metric_handler = self.metric_stub[self._lid]
        metric_handler.update({"checkpoint": False})
        self.metric_stub[self._lid] = metric_handler

    def _ck_bit(self, learner_id):
        return self.metric_stub[learner_id]["checkpoint"]

    def _hyper_to_store(self, hyper_params):
        self.metric_stub[self._lid][self._mutation_key].update(hyper_params)

    @staticmethod
    def collect_metric(**kwargs):
        """Collect metric for learner."""
        return dict(**kwargs)

    def fetch_population_metric(self):
        """Fetch population newest info."""

        return self.metric_stub

    def _eval(self):
        """Fetch lerner stats, to update population metric."""
        pass

    def _get_metric(self, learner_id, key):
        return self.metric_stub[learner_id][key]

    def _get_weight(self, learner_id):
        return self.weight_stub[learner_id]

    def _sort_organism(self, metric):
        """Sort organism of the Population, get the top and bottom organism.name."""
        # note: without use end flag.
        sorted_p = sorted(metric.items(), key=lambda x: x[1][self._metric_key])
        # print("sorted_p: ", sorted_p)
        # target_id = [pid for pid, p_val in sorted_p if p_val["checkpoint"]]
        # target_id = sorted_p
        target_id = [pid for pid, p_val in sorted_p]
        # print("target p: ", target_id)

        if len(target_id) <= 1:
            return [], []

        # sorted_id = [p[0] for p in sorted_p]
        target_count = int(math.ceil(len(target_id) * self._top_rate))

        # too much target organism, clip it
        if target_count * 2 >= len(target_id):
            target_count = int(len(target_id) // 2)
        bottom_id, top_id = target_id[:target_count], target_id[-target_count:]

        # default as 'max', if 'min' type, exchange it.
        if self._metric_type == "min":
            bottom_id, top_id = top_id, bottom_id

        return bottom_id, top_id

    def _assign(self, to_alg_instance, hyper_parameter, weight):
        """Assign the hyper parameter&weight to algorithm."""
        pass

    @staticmethod
    def _explore_hyper_params(config, mutations, resample_prob, perturb_delta):
        """Get new config with explore config."""
        _params = config
        new_params = copy.deepcopy(_params)

        for k, val in mutations.items():
            # skip config keys without this set. e.g, alg skip model config
            if k not in new_params:
                continue

            random_v = random.random()
            # support list and distribution two type.
            if isinstance(val, list):
                if random_v < resample_prob or new_params[k] not in val:
                    new_params[k] = random.choice(val)
                elif random_v > 0.5:
                    # up one index
                    new_params[k] = val[max(0, val.index(_params[k])-1)]
                else:
                    # down one index
                    new_params[k] = val[min(len(val)-1, val.index(_params[k]) + 1)]

            else:  # distribute
                if random_v < resample_prob:
                    new_params[k] = val()
                elif random_v > 0.5:
                    new_params[k] = _params[k] * (1. + perturb_delta)
                else:
                    new_params[k] = _params[k] * (1. - perturb_delta)

                if isinstance(_params[k], int):
                    new_params[k] = int(new_params[k])

        return new_params

    def _exploit_hyper_params(self, hyper_param_from):
        # for _key in to_key:
        # model config
        to_update_params = self._alg_para["model_info"]["actor"]["model_config"]
        for _k in to_update_params:
            if _k not in hyper_param_from:
                continue
            to_update_params[_k] = hyper_param_from[_k]

        # to_update_params.update(hyper_param_from)
        # alg config
        to_update_params = self._alg_para["alg_config"]
        for _k in to_update_params:
            if _k not in hyper_param_from:
                continue
            to_update_params[_k] = hyper_param_from[_k]
        return self._alg_para

    def _flip_k(self, values):
        return {k: val for k, val in values.items() if k in self._hyperpara_mutations}

    def exploit_and_explore(self, top_id):
        """Exploit and explore."""
        weight_src = self._get_weight(top_id)
        hyper_para_src = self._get_metric(top_id, self._mutation_key)
        assert top_id is not self._lid, "self.id: {} vs the top are the same!"

        # exploit within algorithm&model config
        self._alg_para = self._exploit_hyper_params(copy.deepcopy(hyper_para_src))

        # explore model config
        to_mutation = self._alg_para["model_info"]["actor"]["model_config"]
        new_mutation = self._explore_hyper_params(
            to_mutation, self._hyperpara_mutations,
            self._resample_prob, self._perturb_delta)

        # model logs to record
        _old, _new = self._flip_k(to_mutation), self._flip_k(new_mutation)

        self._alg_para["model_info"]["actor"]["model_config"].update(new_mutation)

        # explore algorithm config
        to_mutation_alg = self._alg_para["alg_config"]
        new_mutation_alg = self._explore_hyper_params(
            to_mutation_alg, self._hyperpara_mutations,
            self._resample_prob, self._perturb_delta
        )
        # algorithm logs to record
        _old.update(self._flip_k(to_mutation_alg))
        _new.update(self._flip_k(new_mutation_alg))

        logging.info("[{}] @explore hyper from <{}>:\n{}\nto\n{}\n".format(
            self._lid, top_id, _old, _new))

        self._alg_para["alg_config"].update(new_mutation_alg)

        new_alg = alg_builder(**self._alg_para)
        # print("new_alg_para: ", self._alg_para)
        # print("new alg:", new_alg)
        # print("weights from ", weight_src.keys(), "\n\n", weight_src)
        new_alg.restore(model_weights=weight_src)

        # update population info
        self._hyper_to_store(new_mutation)

        return new_alg

    def _ready(self, t, cur_epi, metric):
        """Check ready, and record info, contains episode_num after each pbt.exploit"""
        if t - self._last_ready_step > self._interval:
            self._last_ready_step = t
            self._previous_acc_episode += cur_epi
            return True

        return False

    def step(self, cur_info, cur_alg):
        """Run a step of PBT."""
        history_step, cur_episode = cur_info["elapsed_step"], cur_info["elapsed_episode"]
        cur_metric = cur_info["episodic_reward_mean"]

        if self._ready(history_step, cur_episode, cur_metric):

            # update whole info into metric
            # save weights always
            # print("cur_info", cur_info, type(cur_info))
            self.update_self_metric(cur_info)
            p_metric = self.fetch_population_metric()
            # print(p_metric)  # , p_weight)
            bottom_ids, top_ids = self._sort_organism(p_metric)
            logging.info("self.lid-{} bottom_ids: {}, top_ids: {}".format(
                self._lid, bottom_ids, top_ids))
            if self._lid in bottom_ids and len(top_ids) > 0:
                # start exploit if need
                # print("raw alg: ", cur_alg)

                # check top organism with checkpoint
                top_ids = [_id for _id in top_ids if self._ck_bit(_id)]
                if not top_ids:  # top without checkpoint do nothing.
                    return None

                organism_to_exploit = random.choice(top_ids)
                new_alg = self.exploit_and_explore(organism_to_exploit)

                self._unset_checkpoint_bit()
                return new_alg
            elif self._lid in top_ids:
                # save top, with set checkpoint bit
                cur_weight = cur_alg.get_weights()
                self._update_self_weight(cur_weight)

            # print("{} get top learner: {} with p_weight \n{}".format(
            #     self._lid, target_organ, p_weight))
        return None

    def summary(self):
        """Get the best organism, and Summary the Population as well."""
        pass
