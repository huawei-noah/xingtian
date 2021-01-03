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
"""Agent group handle the agents' creating, managing and scheduling."""

import os
import sys
import pprint
from copy import deepcopy
from collections import defaultdict, OrderedDict
from functools import partial
from time import time

import numpy as np
from absl import logging

from xt.agent import agent_builder
from xt.algorithm import alg_builder
from xt.environment import env_builder
from zeus.common.ipc.message import message
from zeus.common.util.profile_stats import AgentGroupStats


class WorkerPool(object):
    def __init__(self, parallel_num=4):
        """
        Initialize the Worker Pool with concurrent.futures.

        Now, Using thread pool, could been extend to process fleetly.
        https://docs.python.org/3/library/concurrent.futures.html
        :param parallel_num:
        """
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=parallel_num)

    def do_same_job(self, func, input_list):
        """
        Parallel call func with each para of input_list.

        :param func:
        :param input_list:
        :return: output's index same to the input
        """
        task_submitted = []
        for data in input_list:
            task_submitted.append(self.executor.submit(func, *data))

        return [t.result() for t in task_submitted]

    def do_multi_job(self, func_list, input_list):
        task_submitted = []
        if not input_list:
            for func in func_list:
                task_submitted.append(self.executor.submit(func))
        else:
            for func, data in zip(func_list, input_list):
                task_submitted.append(self.executor.submit(func, *data))

        return [t.result() for t in task_submitted]


class EvaluateData(object):
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self._data_key = ("epi_reward", "step_reward")
        self.data = {_id: self.data_template(self._data_key) for _id in agent_ids}

    def clear(self):
        self.data = {_id: self.data_template(self._data_key) for _id in self.agent_ids}

    @staticmethod
    def data_template(keys):
        return {k: list() for k in keys}

    def append(self, rewards, criteria):
        """
        Append the rewards and criteria data within one evaluate.

        assume, the key in each criteria are s
        :param rewards:
        :param criteria:
        :return:
        """
        for val in rewards:
            agent_id = list(val.keys())[0]
            if agent_id not in self.data.keys():
                self.data.update({agent_id: self.data_template(self._data_key)})
            agent_data = self.data[agent_id]
            for _k in self._data_key:
                agent_data[_k].append(val[agent_id][_k])

        for dict_val in criteria:
            for _ag_id, val in dict_val.items():
                if _ag_id not in self.data.keys():
                    self.data.update({_ag_id: dict()})
                for k, v in val.items():
                    if k not in self.data[_ag_id].keys():
                        self.data[_ag_id].update({k: [v]})
                    else:
                        self.data[_ag_id][k].append(v)

    def get_records(self):
        return self.data


class AgentGroup(object):

    def __init__(self, env_para, alg_para, agent_para,
                 recv_explorer=None, send_explorer=None, buf_stub=None, **kwargs):
        # agent group set scene 'explore' as default.
        alg_para.update({"scene": kwargs.get("scene", "explore")})
        # fixme: parameter apportion
        if "alg_config" not in alg_para.keys():
            alg_para.update({"alg_config": dict()})
        _exp_params = pprint.pformat(
            {"env_para": env_para, "alg_para": alg_para, "agent_para": agent_para},
            indent=0,
            width=1,
        )
        if env_para.get("env_id", 0) < 1:
            logging.info("init agent group for: {}".format(alg_para.get("scene")))
        else:
            logging.debug("init agent group-{}".format(env_para.get("env_id")))

        # That agent belong with an AgentGroup will share the same environment.
        self.env_id = env_para.get("env_id", 0)
        self.restore_count = 0

        self.fill_env_para(env_para, agent_para)
        self.env = env_builder(**env_para)
        self.env_info = self.env.get_env_info()

        # fixme: check from env
        self.agent_num = agent_para.get("agent_num", 1)

        if self.env_info["api_type"] == "standalone":
            self.algs = [alg_builder(**alg_para) for _ in range(self.agent_num)]
            paras_to_init = [
                partial(
                    self.__para_template,
                    agent_para,
                    self.algs[i],
                    self.env,
                    recv_explorer,
                    send_explorer
                )()
                for i in range(self.agent_num)
            ]
        elif self.env_info["api_type"] == "unified":
            self.algs = [alg_builder(**alg_para)]
            paras_to_init = [
                partial(
                    self.__para_template,
                    agent_para,
                    self.algs[0],
                    self.env,
                    recv_explorer,
                    send_explorer
                )()
                for i in range(self.agent_num)
            ]
        else:
            raise ValueError("invalid 'api_type':{} from environment".format(self.env_info))

        # 1. without set weights map, share weights to all agents and set agent_id as index.
        # 2. multi agent, there may have name for each agent, use its original name.
        paras_to_init = self.__update_agent_id(paras_to_init)
        paras_to_init = self._update_env_num(paras_to_init, env_para.get("env_info"))

        # logging.debug("paras_to_init as: \n {}".format(paras_to_init))
        self.agents = [agent_builder(**para) for para in paras_to_init]
        logging.debug("makeup agents: {}".format(self.agents))
        self.step_per_episode = paras_to_init[0]["agent_config"].get("max_steps", 18000)

        # get newest weights map from the algorithm module.
        self.alg_weights_map = {}
        for ag_id in self.env_info["agent_ids"]:
            self.alg_weights_map[ag_id] = self.algs[0].update_weights_map(ag_id)
        for alg in self.algs:
            alg.weights_map = deepcopy(self.alg_weights_map)

        self.recv_explorer = recv_explorer
        self.send_explorer = send_explorer
        self.buf_stub = buf_stub

        self.trajectories = []

        self.bot = WorkerPool(parallel_num=self.agent_num)
        self.eval_data = EvaluateData(self.env_info["agent_ids"])
        self.ag_stats = AgentGroupStats(self.agent_num, self.env_info["api_type"])

    def _update_env_num(self, target_para, env_info):
        if not env_info:
            return target_para

        for i in range(self.agent_num):
            target_para[i].update(
                {"vector_env_size": env_info.get("vector_env_size", 1)})
        return target_para

    @staticmethod
    def __para_template(agent_para, alg, env, recv_explorer, send_explorer):
        # fixme: model info may vary with environment dynamical
        para_template = {
            "agent_name": agent_para.get("agent_name"),
            "alg": alg,
            "env": env,
            "agent_config": agent_para.get("agent_config", {}).copy(),
        }
        # makeup_async_configure
        para_template.update(
            {"recv_explorer": recv_explorer, "send_explorer": send_explorer}
        )

        return para_template

    def __update_agent_id(self, paras):
        if self.env_info["api_type"] == "standalone":
            for i in range(self.agent_num):
                paras[i]["agent_config"].update(
                    {"agent_id": i + self.env_id * self.agent_num})
        else:
            assert self.agent_num == len(self.env_info["agent_ids"]), \
                "agent num not match with environment's, {} vs {}".format(
                    self.agent_num, len(self.env_info["agent_ids"]))
            for i, _id in zip(range(self.agent_num), self.env_info["agent_ids"]):
                paras[i]["agent_config"]["agent_id"] = _id
        return paras

    def _infer_actions(self, inputs):
        job_list = [agent.infer_action for agent in self.agents]
        action_list = self.bot.do_multi_job(job_list, inputs)
        return action_list

    def _handle_env_feedbacks(self, env_feedback_list):
        job_list = [agent.handle_env_feedback for agent in self.agents]
        return self.bot.do_multi_job(job_list, env_feedback_list)

    def _post_processes(self):
        """Post processes after all agents done with one episode."""
        return self.agents[0].post_process(self.agents)

    def restore(self, weights, is_id=True):
        """
        Restore the weights for all the agents.

        {"agent_id": {"prefix": "actor", "name":"YOUR/PATH/TO/MODEL/FILE.h5"}}
        First, find the prefix,
        Second, find name of the model file.
        :param weights:
        :param is_id:
        :return:
        """
        self.restore_count += 1
        # fixme: remove model name file, and make sense to multi-agent.
        if is_id:
            weights = self.buf_stub.get(weights)
            model_weights = {"data": weights}
        else:
            model_weights = {"data": weights}
        # logging.info("model_weights: {}".format(model_weights))
        # logging.info("explorer-{} restore weights: {}".format(self.env_id, type(model_weights)))
        for alg in self.algs:
            # weights as dict data, deliver model by weighs
            # dict, would be useful to multi-agent.
            # bytes, as the weights_id
            # list, as to keras.get_weights
            if isinstance(weights, (dict, bytes, list)):
                alg.restore(model_weights=model_weights["data"])
                continue
            elif not model_weights["data"]:  # None, dummy model.
                # buffer may return weights with None
                logging.debug("Dummy model 'None' in dict, continue!")
                continue

            # 0, default, without weights map, agents will share the same weights
            if not self.alg_weights_map:
                logging.debug("without weights map, use the first weights as default")
                model_name = weights[0]
            # 1, use weight prefix
            elif self.alg_weights_map.get("prefix"):
                pass
            # 2, use model name
            else:
                pass

            assert model_name is not None, "No model weight".format(alg.alg_name)

            # restore model with agent.alg.restore()
            logging.debug("agent-{} trying to load model: {}".format(
                alg.alg_name, model_name))
            alg.restore(model_name)

    def clear_trajectories(self):
        self.trajectories = list()

    def get_trajectories(self):
        return self.trajectories

    @staticmethod
    def __check_all_done(done):
        pass

    def _run_one_unified_episode(self, use_explore, collect=True):
        for agent in self.agents:
            agent.clear_trajectory()

        self.env.reset()
        states = self.env.get_init_state()
        for _step in range(self.step_per_episode):
            for agent in self.agents:
                agent.clear_transition()

            states, transitions = self._do_one_unified_interaction(states, use_explore)

            if collect:
                feed_funcs = [agent.add_to_trajectory for agent in self.agents]
                feed_inputs = [[agent.transition_data] for agent in self.agents]
                self.bot.do_multi_job(feed_funcs, feed_inputs)

            if all([t["done"] for t in transitions]):
                logging.debug("end interaction on step-{}".format(_step))
                break
        else:
            logging.debug("end without done, but max step-{}".format(
                self.step_per_episode))

        states = self._decode_group_data(states)
        last_pred_vals = self._unified_infer(states)
        last_pred_vals = self._reorganize_pred_vals(last_pred_vals)

        feed_inputs = [[last_pred_val] for last_pred_val in last_pred_vals]
        feed_funcs = [agent.get_trajectory for agent in self.agents]

        return self.bot.do_multi_job(feed_funcs, feed_inputs)

    def _decode_group_data(self, data):
        # TODO: check with dynamic agent id
        return [data[agent.id] for agent in self.agents]

    def _unified_infer(self, states):
        pred_vals = self.algs[0].predict(states)
        if not isinstance(pred_vals, tuple):
            pred_vals = [pred_vals]
        return pred_vals

    def _reorganize_pred_vals(self, pred_vals):
        """
        DESC: Reorganize predict values

        predcit values are not organized in a single agent compatiable form,
        so they need to be reorganized.

        the following code does the same thing as:
        ```
        pred_vals_cand = [[] for _ in range(len(self.agents))]
        for i in range(len(self.agents)):
            for j in range(len(pred_vals)):
                pred_vals_cand[i].append(pred_vals[j][i])
        return pred_vals_cand
        ```
        """
        expand_func = partial(np.expand_dims, axis=-1)
        split_func = partial(np.vsplit, indices_or_sections=self.agent_num)
        squeeze_func = partial(np.squeeze, axis=-1)

        pred_vals = map(expand_func, pred_vals)
        pred_vals = map(split_func, pred_vals)
        pred_vals = map(squeeze_func, pred_vals)
        pred_vals = list(zip(*pred_vals))

        return pred_vals

    def _do_one_unified_interaction(self, states, use_explore):
        _start0 = time()
        states = self._decode_group_data(states)
        pred_vals = self._unified_infer(states)
        pred_vals = self._reorganize_pred_vals(pred_vals)

        feed_funcs = [agent.handel_predict_value for agent in self.agents]
        feed_inputs = list(zip(states, pred_vals))

        batch_action =  self.bot.do_multi_job(feed_funcs, feed_inputs)

        # agent.id keep pace with the id within the environment.
        action_package = {_ag.id: v for _ag, v in zip(self.agents, batch_action)}
        self.ag_stats.inference_time += time() - _start0

        _start1 = time()
        next_states, rewards, done, info = self.env.step(action_package)
        self.ag_stats.env_step_time += time() - _start1
        self.ag_stats.iters += 1

        feed_funcs = [agent.handle_env_feedback for agent in self.agents]
        feed_inputs = [
            (s, r, d, i, use_explore)
            for s, r, d, i in zip(*map(self._decode_group_data,
                                       [next_states, rewards, done, info]))
        ]

        transition_data_list = self.bot.do_multi_job(feed_funcs, feed_inputs)

        return next_states, transition_data_list

    def update_model(self):
        """Split update model and explore process. Return model type."""
        _start0 = time()
        model_name = self.agents[0].sync_model()  # fixme: async alg dummy
        self.ag_stats.wait_model_time = time() - _start0

        # fixme: unify model type
        # 1) don't restore model before data meet minimum data set. likes qmix.
        # 2) don't restore with special policy, likes IMPALA.
        if model_name:
            _start1 = time()
            self.restore(model_name)
            self.ag_stats.restore_model_time = time() - _start1
        return type(model_name)

    def explore(self, episode_count):
        """
        Explore the environment.

        agent_num impact on the api about run interaction with environment.
            == 1: use standalone api, `run_one_episode`
            >= 2 and env.api_type == "standalone": agent.run_one_episode
            >= 2 and env.api_type == "unified": agent.do_one_interaction.

        :param episode_count:
        :return:
        """
        # single agent, always use the `run_one_episode` api.
        # multi agent with `standalone` api_type, use the `run_one_episode` api.
        if self.env_info["api_type"] == "standalone":
            # (use_explore, collect)
            _paras = [
                (True, False if _ag.alg.async_flag else True) for _ag in self.agents
            ]
            job_funcs = [agent.run_one_episode for agent in self.agents]
            for _epi_index in range(episode_count):
                _start2 = time()
                self.env.reset()
                for agent in self.agents:
                    agent.reset()

                trajectory_list = self.bot.do_multi_job(job_funcs, _paras)
                for agent, trajectory in zip(self.agents, trajectory_list):
                    if not agent.alg.async_flag:
                        # self.trajectories.append(trajectory)
                        self.send_explorer.send(trajectory)

                self._post_processes()
                self.ag_stats.explore_time_in_epi = time() - _start2

                if _epi_index == episode_count - 1:
                    self.ag_stats.update_with_agent_stats(
                        [agent.get_perf_stats() for agent in self.agents]
                    )

        elif self.env_info["api_type"] == "unified":
            for _ in range(episode_count):
                _start2 = time()
                trajectories = self._run_one_unified_episode(
                    use_explore=True, collect=True)

                for _ag, trajectory in zip(self.agents, trajectories):
                    if not _ag.alg.async_flag:
                        # self.trajectories.append(trajectory)
                        self.send_explorer.send(trajectory)

                self._post_processes()
                self.ag_stats.explore_time_in_epi = time() - _start2
        else:
            pass

        self.clear_trajectories()
        return self.ag_stats.get()

    def evaluate(self, episode_count):
        """Evaluate agent."""
        self.eval_data.clear()

        if self.env_info["api_type"] == "standalone":
            # _paras[0]: False, evaluate without explore
            # _paras[1]: True, always collect the trajectory data, for custom evaluate
            _paras = [(False, True) for _ in range(self.agent_num)]

            interaction_jobs = [agent.run_one_episode for agent in self.agents]
            sum_rewards = [agent.sum_trajectory_reward for agent in self.agents]
            custom_jobs = [agent.calc_custom_evaluate for agent in self.agents]
            for _ in range(episode_count):
                # reset env & agent
                self.env.reset()
                for agent in self.agents:
                    agent.reset()

                self.bot.do_multi_job(interaction_jobs, _paras)

                reward_per_agent = self.bot.do_multi_job(sum_rewards, None)

                # call user's custom evaluate operations on the trajectory, or others
                custom_per_agent = self.bot.do_multi_job(custom_jobs, None)

                self.eval_data.append(reward_per_agent, custom_per_agent)

        elif self.env_info["api_type"] == "unified":
            sum_rewards = [agent.sum_trajectory_reward for agent in self.agents]
            custom_jobs = [agent.calc_custom_evaluate for agent in self.agents]

            for _ in range(episode_count):
                # reset env & agent
                self.env.reset()
                for agent in self.agents:
                    agent.reset()

                self._run_one_unified_episode(use_explore=False, collect=True)

                reward_per_agent = self.bot.do_multi_job(sum_rewards, None)
                # call user's custom evaluate operations on the trajectory, or others
                custom_per_agent = self.bot.do_multi_job(custom_jobs, None)

                self.eval_data.append(reward_per_agent, custom_per_agent)
        else:
            raise ValueError(
                "invalid 'api_type':{} from environment".format(self.env_info)
            )

        return self.eval_data.get_records()

    @staticmethod
    def fill_env_para(env_para, agent_para):
        env_para["env_info"]["agent_num"] = agent_para.get("agent_num", 1)

    def close(self):
        """Close  environment."""
        self.env.close()
