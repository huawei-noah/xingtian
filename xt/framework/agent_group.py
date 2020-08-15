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
"""Agent group handle the agents' creating, managing and scheduling.
"""
import os
import sys
import pprint
from time import time

from functools import partial
from absl import logging

from xt.framework.comm.message import message
from xt.agent import agent_builder
from xt.algorithm import alg_builder
from xt.environment import env_builder
from xt.util.profile_stats import AgentGroupStats


class WorkerPool(object):
    def __init__(self, parallel_num=4):
        """
        Init the Worker Pool with concurrent.futures.
        Now, Using thread pool, could been extend to process fleetly.
        https://docs.python.org/3/library/concurrent.futures.html
        :param parallel_num:
        """
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=parallel_num)

    def do_same_job(self, func, input_list):
        """
        parallel call func with each para of input_list
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
        self.data = {_id: {"reward": list()} for _id in agent_ids}

    def clear(self):
        self.data = {_id: {"reward": list()} for _id in self.agent_ids}

    def append(self, rewards, criteria):
        """
        append the rewards and criteria data within one evaluate.
        assume, the key in each criteria are s
        :param rewards:
        :param criteria:
        :return:
        """
        for val in rewards:
            agent_id = list(val.keys())[0]
            if agent_id not in self.data.keys():
                self.data.update({agent_id: {"reward": list()}})
            agent_data = self.data[agent_id]
            agent_data["reward"].append(val[agent_id])

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
    def __init__(
        self, env_para, alg_para, agent_para, recv_explorer=None, send_explorer=None
    ):
        # agent group set scene 'explore' as default.
        alg_para.update({"scene": "explore"})
        _exp_params = pprint.pformat(
            {"env_para": env_para, "alg_para": alg_para, "agent_para": agent_para},
            indent=0,
            width=1,
        )
        if env_para.get("env_id", 0) < 1:
            logging.info("init agent group with: \n" + _exp_params + "\n")
        else:
            logging.debug("init agent group-{}".format(env_para.get("env_id")))

        # That agent belong with an AgentGroup will share the same environment.
        self.env_id = env_para.get("env_id", 0)
        self.restore_count = 0

        self.fill_env_para(env_para, agent_para)
        self.env = env_builder(**env_para)
        self.env_info = self.env.get_env_info()

        self.agent_num = agent_para.get("agent_num", 1)  # fixme: check from env
        paras_to_init = [
            partial(
                self.__para_template,
                agent_para,
                alg_para,
                self.env,
                recv_explorer,
                send_explorer,
            )()
            for _ in range(self.agent_num)
        ]

        # get newest weights map from the algorithm module.
        self.alg_weights_map = paras_to_init[0]["alg"].weights_map
        # 1. without set weights map, share the weights for all agents.
        # and set the agent_id as index
        # 2. multi agent, there may have name for each agent, used it.
        paras_to_init = self.__update_agent_id(paras_to_init)
        paras_to_init = self._update_env_num(paras_to_init, env_para.get("env_info"))
        # logging.debug("paras_to_init as: \n {}".format(paras_to_init))
        self.agents = [agent_builder(**para) for para in paras_to_init]
        logging.debug("makeup agents: {}".format(self.agents))
        self.step_per_episode = paras_to_init[0]["agent_config"].get("max_steps", 18000)

        self.recv_explorer = recv_explorer
        self.send_explorer = send_explorer

        self.trajectories = []

        self.bot = WorkerPool(parallel_num=self.agent_num)
        self.eval_data = EvaluateData(self.env_info["agent_ids"])
        self.ag_stats = AgentGroupStats(self.agent_num, self.env_info["api_type"])

    def _update_env_num(self, target_para, env_info):
        if not env_info:
            return target_para

        for i in range(self.agent_num):
            target_para[i].update({"vector_env_size": env_info.get("vector_env_size", 1)})
        return target_para

    @staticmethod
    def __para_template(agent_para, alg_para, env, recv_explorer, send_explorer):
        if "alg_config" not in alg_para.keys():  # fixme: parameter apportion
            alg_para.update({"alg_config": dict()})
        # fixme: model info may vary with environment dynamical
        para_template = {
            "agent_name": agent_para.get("agent_name"),
            "alg": alg_builder(**alg_para),
            "env": env,
            "agent_config": agent_para.get("agent_config", {}).copy(),
        }
        # makeup_async_configure
        para_template.update(
            {"recv_explorer": recv_explorer, "send_explorer": send_explorer}
        )

        return para_template

    def __update_agent_id(self, paras):
        if not self.alg_weights_map:
            for i in range(self.agent_num):
                paras[i]["agent_config"].update({"agent_id": i+self.env_id*self.agent_num})
        else:
            assert self.agent_num == len(
                self.env_info["agent_ids"]
            ), "agent num not match with environment's, {} vs {}".format(
                self.agent_num, len(self.env_info["agent_ids"])
            )
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
        """post processes after all agents done with one episode."""

        return self.agents[0].post_process(self.agents)

    def restore(self, weights):
        """restore the weights for all the agents
            {"agent_id": {"prefix": "actor", "name":"YOUR/PATH/TO/MODEL/FILE.h5"}}
            firstly, find the prefix,
            second, find name of the model file.
        :param weights:
        :return:
        """
        self.restore_count += 1
        for _ag in self.agents:
            # weights as dict data, deliver model by weighs
            if isinstance(weights, dict):
                _ag.alg.restore(model_weights=weights)
                print("ag-{} restore weights t-{}".format(_ag.id, self.restore_count))

                continue

            # 0, default, without weights map, agents will share the same weights
            if not self.alg_weights_map:
                logging.debug("without weights map, use the first weights as default")
                model_name = weights[0]
            # 1, use weight prefix
            elif self.alg_weights_map[_ag.id].get("prefix"):
                weight_prefix = self.alg_weights_map[_ag.id].get("prefix")
                model_candid = [
                    _item
                    for _item in weights
                    if os.path.basename(_item).startswith(weight_prefix)
                ]
                model_name = model_candid[0] if len(model_candid) > 0 else None
            # 2, use model name
            else:
                model_name = self.alg_weights_map[_ag.id].get("name")

            assert model_name is not None, "NO model weight for: {}".format(_ag.id)

            # restore model with agent.alg.restore()
            logging.debug(
                "agent-{} trying to load model: {}".format(_ag.id, model_name)
            )
            _ag.alg.restore(model_name)

    def clear_trajectories(self):
        self.trajectories = list()

    def get_trajectories(self):
        return self.trajectories

    @staticmethod
    def __check_all_done(done):
        pass

    def _run_one_unified_episode(self, use_explore, collect=True):
        for _ag in self.agents:
            _ag.clear_trajectory()

        self.env.reset()
        states = self.env.get_init_state()
        for _step in range(self.step_per_episode):
            for _ag in self.agents:
                _ag.clear_transition()

            transitions = self._do_one_unified_interaction(
                states, self.agents, use_explore
            )

            states = {
                _ag.id: _transit["next_state"]
                for _ag, _transit in zip(self.agents, transitions)
            }
            if collect:
                for _ag in self.agents:
                    _ag.add_to_trajectory(_ag.transition_data)

            if all([t["done"] for t in transitions]):
                logging.debug("end interaction on step-{}".format(_step))
                break
        else:
            logging.debug(
                "end without done, but max step-{}".format(self.step_per_episode)
            )

        return [ag.get_trajectory() for ag in self.agents]

    def _decode_group_data(self, data):
        return [data[_ag.id] for _ag in self.agents]

    def _do_one_unified_interaction(self, states, agents, use_explore):
        infer_funcs = [agent.infer_action for agent in agents]
        # agent share weight, inference with anyOne states
        if not self.alg_weights_map:
            inputs = [sta for sta in states.values()]
        else:  # TODO: check with dynamic agent id
            inputs = [states[_agent.id] for _agent in agents]

        _start0 = time()
        inputs = [(val, use_explore) for val in inputs]
        batch_action = self.bot.do_multi_job(infer_funcs, inputs)
        self.ag_stats.inference_time = +time() - _start0

        # agent.id keep pace with the id within the environment.
        action_package = {_ag.id: v for _ag, v in zip(self.agents, batch_action)}

        _start1 = time()
        next_states, rewards, done, info = self.env.step(action_package)
        self.ag_stats.env_step_time += time() - _start1
        self.ag_stats.iters += 1

        feed_inputs = [
            (s, r, d, i, use_explore)
            for s, r, d, i in zip(
                # map(self._decode_group_data, [next_states, rewards, done, info])
                self._decode_group_data(next_states),
                self._decode_group_data(rewards),
                self._decode_group_data(done),
                self._decode_group_data(info),
            )
        ]
        feed_funcs = [agent.handle_env_feedback for agent in agents]
        transition_data_list = self.bot.do_multi_job(feed_funcs, feed_inputs)

        return transition_data_list

    def explore(self, episode_count):
        """
        agent_num impact on the api about run interaction with environment.
            == 1: use standalone api, `run_one_episode`
            >= 2 and env.api_type == "standalone": agent.run_one_episode
            >= 2 and env.api_type == "unified": agent.do_one_interaction.

        :param episode_count:
        :return:
        """
        _start0 = time()
        model_name = self.agents[0].sync_model()  # fixme: async alg dummy
        self.ag_stats.wait_model_time = time() - _start0

        logging.debug("get sync model: {}".format(model_name))

        if isinstance(model_name, dict) or \
                (isinstance(model_name, list) and "none" not in model_name):
            _start1 = time()
            self.restore(model_name)
            self.ag_stats.restore_model_time = time() - _start1

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
                for _ag, trajectory in zip(self.agents, trajectory_list):
                    if not _ag.alg.async_flag:
                        self.trajectories.append(trajectory)
                        self.send_explorer.send(trajectory)

                self._post_processes()
                self.ag_stats.explore_time_in_epi = time() - _start2

                if _epi_index == episode_count - 1:
                    self.ag_stats.update_with_agent_stats(
                        [_a.get_perf_stats() for _a in self.agents]
                    )

        elif self.env_info["api_type"] == "unified":
            for _ in range(episode_count):
                _start2 = time()
                trajectories = self._run_one_unified_episode(
                    use_explore=True, collect=True
                )

                for _ag, trajectory in zip(self.agents, trajectories):
                    if not _ag.alg.async_flag:
                        self.trajectories.append(trajectory)
                        self.send_explorer.send(trajectory)

                self._post_processes()
                self.ag_stats.explore_time_in_epi = time() - _start2

        else:
            raise ValueError(
                "invalid 'api_type':{} from environment".format(self.env_info)
            )

        stats_info = self.ag_stats.get()
        stats_msg = message(stats_info, cmd="stats_msg")
        self.send_explorer.send(stats_msg)

    def evaluate(self, episode_count):
        """ evaluate agent """
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
        """ close  environment """
        self.env.close()
