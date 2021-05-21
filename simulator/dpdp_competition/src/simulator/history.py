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

from src.conf.configs import Configs


class History(object):
    def __init__(self):
        # 车辆为单位, history of vehicles
        self.__vehicle_id_to_node_list = {}
        # 订单为单位, history of order items
        self.__item_id_to_status_list = {}

    def add_vehicle_position_history(self, vehicle_id, update_time, cur_factory_id):
        if vehicle_id not in self.__vehicle_id_to_node_list:
            self.__vehicle_id_to_node_list[vehicle_id] = []

        if len(cur_factory_id) > 0:
            self.__vehicle_id_to_node_list[vehicle_id].append({"factory_id": cur_factory_id,
                                                               "update_time": update_time})

    def add_order_item_status_history(self, item_id, item_state, update_time: int, committed_completion_time, order_id):
        if item_id not in self.__item_id_to_status_list:
            self.__item_id_to_status_list[item_id] = []
        self.__item_id_to_status_list[item_id].append({"state": item_state,
                                                       "update_time": update_time,
                                                       "committed_completion_time": committed_completion_time,
                                                       "order_id": order_id})

    def get_vehicle_position_history(self):
        return self.__vehicle_id_to_node_list

    def get_order_item_status_history(self):
        return self.__item_id_to_status_list

    def add_history_of_vehicles(self, id_to_vehicle: dict, to_time=0):
        if to_time == 0:
            to_time = sys.maxsize

        for vehicle_id, vehicle in id_to_vehicle.items():
            if len(vehicle.cur_factory_id) > 0:
                if vehicle.leave_time_at_current_factory <= to_time:
                    self.add_vehicle_position_history(vehicle_id,
                                                      vehicle.leave_time_at_current_factory,
                                                      vehicle.cur_factory_id)

            if vehicle.destination is not None:
                if vehicle.destination.leave_time <= to_time:
                    self.add_vehicle_position_history(vehicle.id,
                                                      vehicle.destination.leave_time,
                                                      vehicle.destination.id)

            for node in vehicle.planned_route:
                if node.leave_time <= to_time:
                    self.add_vehicle_position_history(vehicle.id,
                                                      node.leave_time,
                                                      node.id)

    def add_history_of_order_items(self, id_to_vehicle: dict, to_time=0):
        if to_time == 0:
            to_time = sys.maxsize

        for vehicle_id, vehicle in id_to_vehicle.items():
            if vehicle.destination is not None:
                if vehicle.destination.arrive_time <= to_time:
                    update_time = vehicle.destination.arrive_time
                    for item in vehicle.destination.pickup_items:
                        self.add_order_item_status_history(item.id,
                                                           Configs.ORDER_STATUS_TO_CODE.get("ONGOING"),
                                                           update_time,
                                                           item.committed_completion_time,
                                                           item.order_id)
                    for item in vehicle.destination.delivery_items:
                        self.add_order_item_status_history(item.id,
                                                           Configs.ORDER_STATUS_TO_CODE.get("COMPLETED"),
                                                           update_time,
                                                           item.committed_completion_time,
                                                           item.order_id)

            for node in vehicle.planned_route:
                if node.arrive_time <= to_time:
                    update_time = node.arrive_time
                    for item in node.pickup_items:
                        self.add_order_item_status_history(item.id,
                                                           Configs.ORDER_STATUS_TO_CODE.get("ONGOING"),
                                                           update_time,
                                                           item.committed_completion_time,
                                                           item.order_id)
                    for item in node.delivery_items:
                        self.add_order_item_status_history(item.id,
                                                           Configs.ORDER_STATUS_TO_CODE.get("COMPLETED"),
                                                           update_time,
                                                           item.committed_completion_time,
                                                           item.order_id)
