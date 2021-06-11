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

from src.common.node import Node
from src.common.route import Map
from src.conf.configs import Configs
from src.utils.input_utils import get_factory_info, get_route_map
from src.utils.json_tools import convert_nodes_to_json
from src.utils.json_tools import get_vehicle_instance_dict, get_order_item_dict
from src.utils.json_tools import read_json_from_file, write_json_to_file
from src.utils.logging_engine import logger


# naive dispatching method
def dispatch_orders_to_vehicles(id_to_unallocated_order_item: dict, id_to_vehicle: dict, id_to_factory: dict):
    """
    :param id_to_unallocated_order_item: item_id ——> OrderItem object(state: "GENERATED")
    :param id_to_vehicle: vehicle_id ——> Vehicle object
    :param id_to_factory: factory_id ——> factory object
    """
    vehicle_id_to_destination = {}
    vehicle_id_to_planned_route = {}

    # dealing with the carrying items of vehicles (处理车辆身上已经装载的货物)
    for vehicle_id, vehicle in id_to_vehicle.items():
        unloading_sequence_of_items = vehicle.get_unloading_sequence()
        vehicle_id_to_planned_route[vehicle_id] = []
        if len(unloading_sequence_of_items) > 0:
            delivery_item_list = []
            factory_id = unloading_sequence_of_items[0].delivery_factory_id
            for item in unloading_sequence_of_items:
                if item.delivery_factory_id == factory_id:
                    delivery_item_list.append(item)
                else:
                    factory = id_to_factory.get(factory_id)
                    node = Node(factory_id, factory.lng, factory.lat, [], copy.copy(delivery_item_list))
                    vehicle_id_to_planned_route[vehicle_id].append(node)
                    delivery_item_list = [item]
                    factory_id = item.delivery_factory_id
            if len(delivery_item_list) > 0:
                factory = id_to_factory.get(factory_id)
                node = Node(factory_id, factory.lng, factory.lat, [], copy.copy(delivery_item_list))
                vehicle_id_to_planned_route[vehicle_id].append(node)

    # for the empty vehicle, it has been allocated to the order, but have not yet arrived at the pickup factory
    pre_matching_item_ids = []
    for vehicle_id, vehicle in id_to_vehicle.items():
        if vehicle.carrying_items.is_empty() and vehicle.destination is not None:
            pickup_items = vehicle.destination.pickup_items
            pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(pickup_items, id_to_factory)
            vehicle_id_to_planned_route[vehicle_id].append(pickup_node)
            vehicle_id_to_planned_route[vehicle_id].append(delivery_node)
            pre_matching_item_ids.extend([item.id for item in pickup_items])

    # dispatch unallocated orders to vehicles
    capacity = __get_capacity_of_vehicle(id_to_vehicle)

    order_id_to_items = {}
    for item_id, item in id_to_unallocated_order_item.items():
        if item_id in pre_matching_item_ids:
            continue
        order_id = item.order_id
        if order_id not in order_id_to_items:
            order_id_to_items[order_id] = []
        order_id_to_items[order_id].append(item)

    vehicle_index = 0
    vehicles = [vehicle for vehicle in id_to_vehicle.values()]
    for order_id, items in order_id_to_items.items():
        demand = __calculate_demand(items)
        if demand > capacity:
            cur_demand = 0
            tmp_items = []
            for item in items:
                if cur_demand + item.demand > capacity:
                    pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(tmp_items, id_to_factory)
                    if pickup_node is None or delivery_node is None:
                        continue
                    vehicle = vehicles[vehicle_index]
                    vehicle_id_to_planned_route[vehicle.id].append(pickup_node)
                    vehicle_id_to_planned_route[vehicle.id].append(delivery_node)

                    vehicle_index = (vehicle_index + 1) % len(vehicles)
                    tmp_items = []
                    cur_demand = 0

                tmp_items.append(item)
                cur_demand += item.demand

            if len(tmp_items) > 0:
                pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(tmp_items, id_to_factory)
                if pickup_node is None or delivery_node is None:
                    continue
                vehicle = vehicles[vehicle_index]
                vehicle_id_to_planned_route[vehicle.id].append(pickup_node)
                vehicle_id_to_planned_route[vehicle.id].append(delivery_node)
        else:
            pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(items, id_to_factory)
            if pickup_node is None or delivery_node is None:
                continue
            vehicle = vehicles[vehicle_index]
            vehicle_id_to_planned_route[vehicle.id].append(pickup_node)
            vehicle_id_to_planned_route[vehicle.id].append(delivery_node)

        vehicle_index = (vehicle_index + 1) % len(vehicles)

    # create the output of the algorithm
    for vehicle_id, vehicle in id_to_vehicle.items():
        origin_planned_route = vehicle_id_to_planned_route.get(vehicle_id)
        # Combine adjacent-duplicated nodes.
        __combine_duplicated_nodes(origin_planned_route)

        destination = None
        planned_route = []
        # determine the destination
        if vehicle.destination is not None:
            if len(origin_planned_route) == 0:
                logger.error(f"Planned route of vehicle {vehicle_id} is wrong")
            else:
                destination = origin_planned_route[0]
                destination.arrive_time = vehicle.destination.arrive_time
                planned_route = [origin_planned_route[i] for i in range(1, len(origin_planned_route))]
        elif len(origin_planned_route) > 0:
            destination = origin_planned_route[0]
            planned_route = [origin_planned_route[i] for i in range(1, len(origin_planned_route))]

        vehicle_id_to_destination[vehicle_id] = destination
        vehicle_id_to_planned_route[vehicle_id] = planned_route

    return vehicle_id_to_destination, vehicle_id_to_planned_route


def __calculate_demand(item_list: list):
    demand = 0
    for item in item_list:
        demand += item.demand
    return demand


def __get_capacity_of_vehicle(id_to_vehicle: dict):
    for vehicle_id, vehicle in id_to_vehicle.items():
        return vehicle.board_capacity


def __create_pickup_and_delivery_nodes_of_items(items: list, id_to_factory: dict):
    pickup_factory_id = __get_pickup_factory_id(items)
    delivery_factory_id = __get_delivery_factory_id(items)
    if len(pickup_factory_id) == 0 or len(delivery_factory_id) == 0:
        return None, None

    pickup_factory = id_to_factory.get(pickup_factory_id)
    delivery_factory = id_to_factory.get(delivery_factory_id)
    pickup_node = Node(pickup_factory.id, pickup_factory.lng, pickup_factory.lat, copy.copy(items), [])

    delivery_items = []
    last_index = len(items) - 1
    for i in range(len(items)):
        delivery_items.append(items[last_index - i])
    delivery_node = Node(delivery_factory.id, delivery_factory.lng, delivery_factory.lat, [], copy.copy(delivery_items))
    return pickup_node, delivery_node


def __get_pickup_factory_id(items):
    if len(items) == 0:
        logger.error("Length of items is 0")
        return ""

    factory_id = items[0].pickup_factory_id
    for item in items:
        if item.pickup_factory_id != factory_id:
            logger.error("The pickup factory of these items is not the same")
            return ""

    return factory_id


def __get_delivery_factory_id(items):
    if len(items) == 0:
        logger.error("Length of items is 0")
        return ""

    factory_id = items[0].delivery_factory_id
    for item in items:
        if item.delivery_factory_id != factory_id:
            logger.error("The delivery factory of these items is not the same")
            return ""

    return factory_id


# 合并相邻重复节点 Combine adjacent-duplicated nodes.
def __combine_duplicated_nodes(nodes):
    n = 0
    while n < len(nodes)-1:
        if nodes[n].id == nodes[n+1].id:
            nodes[n].pickup_items.extend(nodes.pop(n+1).pickup_items)
        n += 1


"""

Main body
# Note
# This is the demo to show the main flowchart of the algorithm

"""


def scheduling():
    # read the input json, you can design your own classes
    id_to_factory, id_to_unallocated_order_item, id_to_ongoing_order_item, id_to_vehicle = __read_input_json()

    # dispatching algorithm
    vehicle_id_to_destination, vehicle_id_to_planned_route = dispatch_orders_to_vehicles(
        id_to_unallocated_order_item,
        id_to_vehicle,
        id_to_factory)

    # output the dispatch result
    __output_json(vehicle_id_to_destination, vehicle_id_to_planned_route)


def __read_input_json():
    # read the factory info
    id_to_factory = get_factory_info(Configs.factory_info_file_path)

    # read the route map
    # code_to_route = get_route_map(Configs.route_info_file_path)
    # route_map = Map(code_to_route)

    # read the input json, you can design your own classes
    unallocated_order_items = read_json_from_file(Configs.algorithm_unallocated_order_items_input_path)
    id_to_unallocated_order_item = get_order_item_dict(unallocated_order_items, 'OrderItem')

    ongoing_order_items = read_json_from_file(Configs.algorithm_ongoing_order_items_input_path)
    id_to_ongoing_order_item = get_order_item_dict(ongoing_order_items, 'OrderItem')

    id_to_order_item = {**id_to_unallocated_order_item, **id_to_ongoing_order_item}

    vehicle_infos = read_json_from_file(Configs.algorithm_vehicle_input_info_path)
    id_to_vehicle = get_vehicle_instance_dict(vehicle_infos, id_to_order_item, id_to_factory)

    return id_to_factory, id_to_unallocated_order_item, id_to_ongoing_order_item, id_to_vehicle


def __output_json(vehicle_id_to_destination, vehicle_id_to_planned_route):
    write_json_to_file(Configs.algorithm_output_destination_path, convert_nodes_to_json(vehicle_id_to_destination))
    write_json_to_file(Configs.algorithm_output_planned_route_path, convert_nodes_to_json(vehicle_id_to_planned_route))
