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

import datetime
import os
import sys
import time

from src.common.dispatch_result import DispatchResult
from src.common.input_info import InputInfo
from src.conf.configs import Configs
from src.simulator.history import History
from src.simulator.vehicle_simulator import VehicleSimulator
from src.utils.checker import Checker
from src.utils.evaluator import Evaluator
from src.utils.json_tools import convert_input_info_to_json_files
from src.utils.json_tools import get_output_of_algorithm
from src.utils.json_tools import subprocess_function, get_algorithm_calling_command
from src.utils.logging_engine import logger
from src.utils.tools import get_item_dict_from_order_dict, get_order_items_to_be_dispatched_of_cur_time
from src.utils.tools import get_item_list_of_vehicles


class SimulateEnvironment(object):
    def __init__(self, initial_time: int, time_interval: int, id_to_order: dict, id_to_vehicle: dict,
                 id_to_factory: dict, route_map):
        """
        :param initial_time: unix timestamp, unit is second
        :param time_interval: unit is second
        :param id_to_order: 所有订单集合, total orders
        :param id_to_vehicle: 所有车辆, total vehicles
        :param id_to_factory: 工厂信息, total factories
        :param route_map: 路网信息
        """
        self.initial_time = initial_time
        self.time_interval = time_interval
        self.cur_time = initial_time
        self.pre_time = initial_time

        self.id_to_order = id_to_order
        # item list of total orders, used for order splitting
        self.id_to_order_item = get_item_dict_from_order_dict(id_to_order)
        self.id_to_vehicle = id_to_vehicle
        self.id_to_factory = id_to_factory
        self.route_map = route_map

        # 不同状态的订单, order with different status
        self.id_to_generated_order_item = {}  # state = 1
        self.id_to_ongoing_order_item = {}  # state = 2
        self.id_to_completed_order_item = {}  # state = 3

        # 车辆运行模拟器, simulate the vehicle status and order fulfillment in a given time interval [pre_time, cur_time]
        self.vehicle_simulator = VehicleSimulator(route_map, id_to_factory)

        # 每次派单结果的存档, save each dispatch result from the algorithm
        self.time_to_dispatch_result = {}

        # 历史记录保存, save the visited nodes of vehicles and different status of orders for evaluation
        self.history = self.__ini_history()

        # 目标函数值, objective
        self.total_score = sys.maxsize

    # 初始化历史记录
    def __ini_history(self):
        history = History()
        for vehicle_id, vehicle in self.id_to_vehicle.items():
            history.add_vehicle_position_history(vehicle_id, vehicle.gps_update_time, vehicle.cur_factory_id)
        for item_id, item in self.id_to_order_item.items():
            history.add_order_item_status_history(item.id, item.delivery_state, self.initial_time,
                                                  item.committed_completion_time, item.order_id)
        return history

    # 模拟器仿真环节
    def run(self):
        used_seconds = 0
        # 迭代
        while True:
            logger.info(f"{'*' * 50}")

            # 确定当前时间, 取算法执行时间和模拟器的切片时间的大值
            self.cur_time = self.pre_time + (used_seconds // self.time_interval + 1) * self.time_interval
            logger.info(f"cur time: {datetime.datetime.fromtimestamp(self.cur_time)}, "
                        f"pre time: {datetime.datetime.fromtimestamp(self.pre_time)}")

            # update the status of vehicles and orders in a given interval [self.pre_time, self.cur_time]
            updated_input_info = self.update_input()

            # 派单环节, 设计与算法交互
            used_seconds, dispatch_result = self.dispatch(updated_input_info)
            self.time_to_dispatch_result[self.cur_time] = dispatch_result

            # 校验, 车辆目的地不能改变
            if not Checker.check_dispatch_result(dispatch_result, self.id_to_vehicle, self.id_to_order):
                logger.error("Dispatch result is infeasible")
                return

            # 根据派单指令更新车辆
            self.deliver_control_command_to_vehicles(dispatch_result)

            # 判断是否完成所有订单的派发
            if self.complete_the_dispatch_of_all_orders():
                break

            self.pre_time = self.cur_time

            # 若订单已经超时, 但是算法依旧未分配, 模拟终止
            if self.ignore_allocating_timeout_orders(dispatch_result):
                break

        # 模拟完成车辆剩下的订单
        self.simulate_the_left_ongoing_orders_of_vehicles(self.id_to_vehicle)

        # 根据self.history 计算指标
        self.total_score = Evaluator.calculate_total_score(self.history, self.route_map, len(self.id_to_vehicle))

    # 数据更新
    def update_input(self):
        logger.info(f"Start to update the input of {datetime.datetime.fromtimestamp(self.cur_time)}")

        # 获取车辆的位置信息和订单状态
        # Get the updated status of vehicles and orders according to the simulator
        self.vehicle_simulator.run(self.id_to_vehicle, self.pre_time)
        self.vehicle_simulator.parse_simulation_result(self.id_to_vehicle, self.cur_time)
        # 增加历史记录, add history
        self.history.add_history_of_vehicles(self.id_to_vehicle, self.cur_time)
        self.history.add_history_of_order_items(self.id_to_vehicle, self.cur_time)

        # 更新订单状态
        self.update_status_of_orders(self.vehicle_simulator.completed_item_ids, self.vehicle_simulator.ongoing_item_ids)

        # 更新车辆状态
        self.update_status_of_vehicles(self.vehicle_simulator.vehicle_id_to_cur_position_info,
                                       self.vehicle_simulator.vehicle_id_to_destination,
                                       self.vehicle_simulator.vehicle_id_to_carrying_items)

        # 根据当前时间选择待分配订单的物料集合
        # Select the item collection of the orders to be allocated according to the current time
        self.id_to_generated_order_item = get_order_items_to_be_dispatched_of_cur_time(self.id_to_order_item,
                                                                                       self.cur_time)

        # 汇总车辆、订单和路网信息, 作为派单算法的输入
        # create the input of algorithm
        updated_input_info = InputInfo(self.id_to_generated_order_item, self.id_to_ongoing_order_item,
                                       self.id_to_vehicle, self.id_to_factory, self.route_map)
        logger.info(f"Get {len(self.id_to_generated_order_item)} unallocated order items, "
                    f"{len(self.id_to_ongoing_order_item)} ongoing order items, "
                    f"{len(self.id_to_completed_order_item)} completed order items")

        return updated_input_info

    # 更新订单状态
    def update_status_of_orders(self, completed_item_ids, ongoing_item_ids):
        for item_id in completed_item_ids:
            item = self.id_to_order_item.get(item_id)
            if item is not None:
                if item_id not in self.id_to_completed_order_item:
                    self.id_to_completed_order_item[item_id] = item
                    item.delivery_state = Configs.ORDER_STATUS_TO_CODE.get("COMPLETED")

        for item_id in ongoing_item_ids:
            item = self.id_to_order_item.get(item_id)
            if item is not None:
                if item_id not in self.id_to_ongoing_order_item:
                    self.id_to_ongoing_order_item[item_id] = item
                    item.delivery_state = Configs.ORDER_STATUS_TO_CODE.get("ONGOING")

        # remove expired items
        expired_item_id_list = []
        for item_id, item in self.id_to_ongoing_order_item.items():
            if item.delivery_state > Configs.ORDER_STATUS_TO_CODE.get("ONGOING"):
                expired_item_id_list.append(item_id)
        for item_id in expired_item_id_list:
            self.id_to_ongoing_order_item.pop(item_id)

    # 更新车辆状态
    def update_status_of_vehicles(self, vehicle_id_to_cur_position_info, vehicle_id_to_destination,
                                  vehicle_id_to_carry_items):
        for vehicle_id, vehicle in self.id_to_vehicle.items():
            if vehicle_id in vehicle_id_to_cur_position_info:
                cur_position_info = vehicle_id_to_cur_position_info.get(vehicle_id)
                vehicle.set_cur_position_info(cur_position_info.get("cur_factory_id"),
                                              cur_position_info.get("update_time"),
                                              cur_position_info.get("arrive_time_at_current_factory"),
                                              cur_position_info.get("leave_time_at_current_factory"))
            else:
                logger.error(f"Vehicle {vehicle_id} does not have updated position information")

            if vehicle_id in vehicle_id_to_destination:
                vehicle.destination = vehicle_id_to_destination.get(vehicle_id)
            else:
                logger.error(f"Vehicle {vehicle_id} does not have the destination information")

            if vehicle_id in vehicle_id_to_carry_items:
                vehicle.carrying_items = vehicle_id_to_carry_items.get(vehicle_id)
            else:
                logger.error(f"Vehicle {vehicle_id} does not have the information of carrying items")

            vehicle.planned_route = []

    # 派单环节
    def dispatch(self, input_info):
        # 1. Prepare the input json of the algorithm
        convert_input_info_to_json_files(input_info)

        # 2. Run the algorithm
        command = get_algorithm_calling_command()
        time_start_algorithm = time.time()
        used_seconds, message = subprocess_function(command)

        # 3. parse the output json of the algorithm
        if Configs.ALGORITHM_SUCCESS_FLAG in message:
            if (time_start_algorithm < os.stat(Configs.algorithm_output_destination_path).st_mtime < time.time()
                    and time_start_algorithm < os.stat(
                        Configs.algorithm_output_planned_route_path).st_mtime < time.time()):
                vehicle_id_to_destination, vehicle_id_to_planned_route = get_output_of_algorithm(self.id_to_order_item)
                dispatch_result = DispatchResult(vehicle_id_to_destination, vehicle_id_to_planned_route)
                return used_seconds, dispatch_result
            else:
                logger.error('output.json is not the newest')
                sys.exit(-1)
        else:
            logger.error(message)
            raise ValueError('未寻获算法输出成功标识SUCCESS')

    # 判断是否完成所有订单的派发
    def complete_the_dispatch_of_all_orders(self):
        for item in self.id_to_order_item.values():
            if item.delivery_state <= 1:
                logger.info(f"{datetime.datetime.fromtimestamp(self.cur_time)}, Item {item.id}: "
                            f"state = {item.delivery_state} < 2, we can not finish the simulation")
                return False
        logger.info(f"{datetime.datetime.fromtimestamp(self.cur_time)}, the status of all items is greater than 1, "
                    f"we could finish the simulation")
        return True

    # 把车辆身上的剩余订单模拟掉
    def simulate_the_left_ongoing_orders_of_vehicles(self, id_to_vehicle: dict):
        self.vehicle_simulator.run(id_to_vehicle, self.cur_time)
        self.history.add_history_of_vehicles(self.id_to_vehicle)
        self.history.add_history_of_order_items(self.id_to_vehicle)

    def deliver_control_command_to_vehicles(self, dispatch_result):
        vehicle_id_to_destination = dispatch_result.vehicle_id_to_destination
        vehicle_id_to_planned_route = dispatch_result.vehicle_id_to_planned_route

        for vehicle_id, vehicle in self.id_to_vehicle.items():
            if vehicle_id not in vehicle_id_to_destination:
                logger.error(f"algorithm does not output the destination of vehicle {vehicle_id}")
                continue
            if vehicle_id not in vehicle_id_to_planned_route:
                logger.error(f"algorithm does not output the planned route of vehicle {vehicle_id}")
                continue
            vehicle.destination = vehicle_id_to_destination.get(vehicle_id)
            if vehicle.destination is not None:
                vehicle.destination.update_service_time()
            vehicle.planned_route = vehicle_id_to_planned_route.get(vehicle_id)
            for node in vehicle.planned_route:
                if node is not None:
                    node.update_service_time()

    # 检查当前是否有订单已经超时却依旧未分配
    def ignore_allocating_timeout_orders(self, dispatch_result):
        vehicle_id_to_item_list = get_item_list_of_vehicles(dispatch_result, self.id_to_vehicle)
        total_item_ids_in_dispatch_result = []
        for vehicle_id, item_list in vehicle_id_to_item_list.items():
            for item in item_list:
                if item.id not in total_item_ids_in_dispatch_result:
                    total_item_ids_in_dispatch_result.append(item.id)

        for item_id, item in self.id_to_generated_order_item.items():
            if item_id not in total_item_ids_in_dispatch_result:
                if item.committed_completion_time < self.cur_time:
                    logger.error(f"{datetime.datetime.fromtimestamp(self.cur_time)}, Item {item_id} has timed out, "
                                 f"however it is still ignored in the dispatch result")
                    return True
        return False
