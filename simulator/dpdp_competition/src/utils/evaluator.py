# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# THE SOFTWARE

import sys

from src.utils.logging_engine import logger
from src.conf.configs import Configs


# 评价器
class Evaluator(object):
    # 多目标处理方式：距离增加量与超时量加权求和
    @staticmethod
    def calculate_total_score(history, route_map, vehicle_num: int):
        total_distance = Evaluator.calculate_total_distance(history.get_vehicle_position_history(), route_map)
        logger.info(f"Total distance: {total_distance: .3f}")
        total_over_time = Evaluator.calculate_total_over_time(history.get_order_item_status_history())
        logger.info(f"Sum over time: {total_over_time: .3f}")
        total_score = total_distance / vehicle_num + total_over_time * Configs.LAMDA / 3600
        logger.info(f"Total score: {total_score: .3f}")
        return total_score

    @staticmethod
    def calculate_total_distance(vehicle_id_to_node_list: dict, route_map):
        total_distance = 0
        if not vehicle_id_to_node_list:
            return total_distance

        for vehicle_id, nodes in vehicle_id_to_node_list.items():
            travel_factory_list = []
            for node in nodes:
                travel_factory_list.append(node['factory_id'])
            distance = calculate_traveling_distance_of_routes(travel_factory_list, route_map)
            total_distance += distance
            logger.info(f"Traveling Distance of Vehicle {vehicle_id} is {distance: .3f}, "
                        f"visited node list: {len(travel_factory_list)}")
        return total_distance

    @staticmethod
    def calculate_total_over_time(item_id_to_status_list: dict):
        total_over_time = 0

        order_id_to_item_id_list = {}
        item_id_to_complete_time = {}
        order_id_to_committed_completion_time = {}
        failure_flag = False
        for item_id, status_info_list in item_id_to_status_list.items():
            selected_status_info_list = [status_info for status_info in status_info_list
                                         if status_info["state"] == Configs.ORDER_STATUS_TO_CODE.get("COMPLETED")]

            if len(selected_status_info_list) == 0:
                logger.error(f"Item {item_id} has no history of completion status")
                failure_flag = True
                continue

            selected_status_info_list.sort(key=lambda x: x["update_time"])
            item_id_to_complete_time[item_id] = selected_status_info_list[0].get("update_time")

            order_id = selected_status_info_list[0].get("order_id")
            if order_id not in order_id_to_item_id_list:
                order_id_to_item_id_list[order_id] = []
            order_id_to_item_id_list[order_id].append(item_id)
            if order_id not in order_id_to_committed_completion_time:
                order_id_to_committed_completion_time[order_id] = selected_status_info_list[0].get(
                    "committed_completion_time")

        if failure_flag:
            return sys.maxsize

        for order_id, item_id_list in order_id_to_item_id_list.items():
            latest_arr_time = -1
            for item_id in item_id_list:
                if item_id in item_id_to_complete_time:
                    completion_time = item_id_to_complete_time.get(item_id)
                    if completion_time > latest_arr_time:
                        latest_arr_time = completion_time

            committed_completion_time = order_id_to_committed_completion_time.get(order_id)
            over_time = latest_arr_time - committed_completion_time
            if over_time > 0:
                total_over_time += over_time

        return total_over_time


def calculate_traveling_distance_of_routes(factory_id_list, route_map):
    travel_distance = 0
    if len(factory_id_list) <= 1:
        return travel_distance

    for index in range(len(factory_id_list) - 1):
        travel_distance += route_map.calculate_distance_between_factories(factory_id_list[index],
                                                                          factory_id_list[index + 1])
    return travel_distance
