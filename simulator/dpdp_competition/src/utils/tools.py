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

import copy

from src.conf.configs import Configs


def calculate_load_time(total_big_board):
    # 单位是秒
    return total_big_board / Configs.LOAD_SPEED * 60


def calculate_unload_time(total_big_board):
    # 单位是秒
    return total_big_board / Configs.UNLOAD_SPEED * 60


# 订单转化
def get_item_dict_from_order_dict(id_to_order: dict):
    id_to_order_item = {}
    for order_id, order in id_to_order.items():
        for item in order.item_list:
            if item.id not in id_to_order_item:
                id_to_order_item[item.id] = item
    return id_to_order_item


# 获取当前待分配的订单
def get_order_items_to_be_dispatched_of_cur_time(id_to_order_item: dict, cur_time: int):
    """
    :param id_to_order_item: 所有订单物料, total order items
    :param cur_time: unix timestamp, unit is second
    :return: 返回当前时间新生成和delivery_state=1("GENERATED")的所有订单物料
    """
    # 获取当前时间之前还未到装货环节的订单, 依旧可以再分配
    id_to_generated_order_item = {item.id: item for item in id_to_order_item.values()
                                  if item.delivery_state == Configs.ORDER_STATUS_TO_CODE.get("GENERATED")}

    # 获取当前之间之前新生成的订单
    id_to_generated_order_item.update(__get_newly_generated_items(id_to_order_item, cur_time))

    return id_to_generated_order_item


def __get_newly_generated_items(id_to_order_item: dict, cur_time: int):
    id_to_item = {}
    for item_id, item in id_to_order_item.items():
        if item.creation_time <= cur_time:
            if item.delivery_state == Configs.ORDER_STATUS_TO_CODE.get("INITIALIZATION"):
                # 修改订单状态
                item.delivery_state = Configs.ORDER_STATUS_TO_CODE.get("GENERATED")
                id_to_item[item_id] = item
    return id_to_item


# 获取各车辆分配的物料集合
def get_item_list_of_vehicles(dispatch_result, id_to_vehicle: dict):
    vehicle_id_to_item_list = {}

    vehicle_id_to_destination = dispatch_result.vehicle_id_to_destination
    vehicle_id_to_planned_route = dispatch_result.vehicle_id_to_planned_route

    for vehicle_id, vehicle in id_to_vehicle.items():
        item_list = []
        item_id_list = []
        carrying_items = copy.deepcopy(vehicle.carrying_items)
        while not carrying_items.is_empty():
            item = carrying_items.pop()
            if item.id not in item_id_list:
                item_id_list.append(item.id)
                item_list.append(item)

        if vehicle_id in vehicle_id_to_destination:
            destination = vehicle_id_to_destination.get(vehicle_id)
            if destination is not None:
                pickup_items = destination.pickup_items
                for item in pickup_items:
                    if item.id not in item_id_list:
                        item_id_list.append(item.id)
                        item_list.append(item)

        if vehicle_id in vehicle_id_to_planned_route:
            for node in vehicle_id_to_planned_route.get(vehicle_id):
                pickup_items = node.pickup_items
                for item in pickup_items:
                    if item.id not in item_id_list:
                        item_id_list.append(item.id)
                        item_list.append(item)

        vehicle_id_to_item_list[vehicle_id] = item_list

    return vehicle_id_to_item_list
