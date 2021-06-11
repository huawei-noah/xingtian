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

from src.utils.logging_engine import logger
from src.utils.tools import get_item_list_of_vehicles


class Checker(object):
    @staticmethod
    def check_dispatch_result(dispatch_result, id_to_vehicle: dict, id_to_order: dict):
        vehicle_id_to_destination = dispatch_result.vehicle_id_to_destination
        vehicle_id_to_planned_route = dispatch_result.vehicle_id_to_planned_route

        # 检查是否所有的车辆都有返回值
        if len(vehicle_id_to_destination) != len(id_to_vehicle):
            logger.error(f"Num of returned destination {len(vehicle_id_to_destination)} "
                         f"is not equal to vehicle number {len(id_to_vehicle)}")
            return False

        if len(vehicle_id_to_planned_route) != len(id_to_vehicle):
            logger.error(f"Num of returned planned route {len(vehicle_id_to_planned_route)} "
                         f"is not equal to vehicle number {len(id_to_vehicle)}")
            return False

        # 逐个检查各车辆路径
        for vehicle_id, vehicle in id_to_vehicle.items():
            if vehicle_id not in vehicle_id_to_destination:
                logger.error(f"Destination information of Vehicle {vehicle_id} is not in the returned result")
                return False

            # check destination
            destination_in_result = vehicle_id_to_destination.get(vehicle_id)
            if not Checker.__is_returned_destination_valid(destination_in_result, vehicle):
                logger.error(f"Returned destination of Vehicle {vehicle_id} is not valid")
                return False

            # check routes
            if vehicle_id not in vehicle_id_to_planned_route:
                logger.error(f"Planned route of Vehicle {vehicle_id} is not in the returned result")
                return False

            route = []
            if destination_in_result is not None:
                route.append(destination_in_result)
            route.extend(vehicle_id_to_planned_route.get(vehicle_id))

            if len(route) > 0:
                # check capacity
                if not Checker.__meet_capacity_constraint(route, copy.deepcopy(vehicle.carrying_items),
                                                          vehicle.board_capacity):
                    logger.error(f"Vehicle {vehicle_id} violates the capacity constraint")
                    return False

                # check LIFO
                if not Checker.__meet_loading_and_unloading_constraint(route, copy.deepcopy(vehicle.carrying_items)):
                    logger.error(f"Vehicle {vehicle_id} violates the LIFO constraint")
                    return False

                # check adjacent-duplicated nodes
                Checker.__contain_duplicated_nodes(vehicle_id, route)

                # check duplicated item id
                if Checker.__contain_duplicate_items(route, copy.deepcopy(vehicle.carrying_items)):
                    return False

                if not Checker.__do_pickup_and_delivery_items_match_the_node(route):
                    return False

        # check order splitting
        if not Checker.__meet_order_splitting_constraint(dispatch_result, id_to_vehicle, id_to_order):
            return False

        return True

    @staticmethod
    def __is_returned_destination_valid(returned_destination, vehicle):
        origin_destination = vehicle.destination
        if origin_destination is not None:
            if returned_destination is None:
                logger.error(f"Vehicle {vehicle.id}, returned destination is None, "
                             f"however the origin destination is not None.")
                return False
            else:
                if origin_destination.id != returned_destination.id:
                    logger.error(f"Vehicle {vehicle.id}, returned destination id is {returned_destination.id}, "
                                 f"however the origin destination id is {origin_destination.id}.")
                    return False

                if origin_destination.arrive_time != returned_destination.arrive_time:
                    logger.error(f"Vehicle {vehicle.id}, arrive time of returned destination is "
                                 f"{returned_destination.arrive_time}, "
                                 f"however the arrive time of origin destination is "
                                 f"{origin_destination.arrive_time}.")
                    return False
        elif len(vehicle.cur_factory_id) == 0 and returned_destination is None:
            logger.error(f"Currently, Vehicle {vehicle.id} is not in the factory(cur_factory_id==''), "
                         f"however, returned destination is also None, we cannot locate the vehicle.")
            return False

        return True

    # 载重约束
    @staticmethod
    def __meet_capacity_constraint(route: list, carrying_items, capacity):
        left_capacity = capacity

        # Stack
        while not carrying_items.is_empty():
            item = carrying_items.pop()
            left_capacity -= item.demand
            if left_capacity < 0:
                logger.error(f"left capacity {left_capacity} < 0")
                return False

        for node in route:
            delivery_items = node.delivery_items
            pickup_items = node.pickup_items
            for item in delivery_items:
                left_capacity += item.demand
                if left_capacity > capacity:
                    logger.error(f"left capacity {left_capacity} > capacity {capacity}")
                    return False

            for item in pickup_items:
                left_capacity -= item.demand
                if left_capacity < 0:
                    logger.error(f"left capacity {left_capacity} < 0")
                    return False
        return True

    # LIFO约束
    @staticmethod
    def __meet_loading_and_unloading_constraint(route: list, carrying_items):
        for node in route:
            delivery_items = node.delivery_items
            pickup_items = node.pickup_items
            for item in delivery_items:
                unload_item = carrying_items.pop()
                if unload_item is None or unload_item.id != item.id:
                    return False

            for item in pickup_items:
                carrying_items.push(item)

        return carrying_items.is_empty()

    # 检查相邻的节点是否重复，并警告，鼓励把相邻重复节点进行合并
    @staticmethod
    def __contain_duplicated_nodes(vehicle_id, route):
        for n in range(len(route) - 1):
            if route[n].id == route[n + 1].id:
                logger.warning(f"{vehicle_id} has adjacent-duplicated nodes which are encouraged to be combined in one.")

    @staticmethod
    def __contain_duplicate_items(route, carrying_items):
        item_id_list = []
        while not carrying_items.is_empty():
            item = carrying_items.pop()
            if item.id not in item_id_list:
                item_id_list.append(item.id)
            else:
                logger.error(f"Item {item.id}: duplicate item id")
                return True

        for node in route:
            pickup_items = node.pickup_items
            for item in pickup_items:
                if item.id not in item_id_list:
                    item_id_list.append(item.id)
                else:
                    logger.error(f"Item {item.id}: duplicate item id")
                    return True
        return False

    @staticmethod
    def __do_pickup_and_delivery_items_match_the_node(route: list):
        for node in route:
            factory_id = node.id
            pickup_items = node.pickup_items
            for item in pickup_items:
                if item.pickup_factory_id != factory_id:
                    logger.error(f"Pickup factory of item {item.id} is {item.pickup_factory_id}, "
                                 f"however you allocate the vehicle to pickup this item in {factory_id}")
                    return False
            delivery_items = node.delivery_items
            for item in delivery_items:
                if item.delivery_factory_id != factory_id:
                    logger.error(f"Delivery factory of item {item.id} is {item.delivery_factory_id}, "
                                 f"however you allocate the vehicle to delivery this item in {factory_id}")
                    return False
        return True

    @staticmethod
    def __meet_order_splitting_constraint(dispatch_result, id_to_vehicle: dict, id_to_order: dict):
        vehicle_id_to_destination = dispatch_result.vehicle_id_to_destination
        vehicle_id_to_planned_route = dispatch_result.vehicle_id_to_planned_route

        vehicle_id_to_item_list = get_item_list_of_vehicles(dispatch_result, id_to_vehicle)

        capacity = 0
        split_order_id_list = Checker.__find_split_orders_from_vehicles(vehicle_id_to_item_list)

        for vehicle_id, vehicle in id_to_vehicle.items():
            logger.debug(f"Find split orders of vehicle {vehicle_id}")

            capacity = vehicle.board_capacity
            carrying_items = copy.deepcopy(vehicle.carrying_items)
            route = []
            if vehicle_id in vehicle_id_to_destination and vehicle_id_to_destination.get(vehicle_id) is not None:
                route.append(vehicle_id_to_destination.get(vehicle_id))
            if vehicle_id in vehicle_id_to_planned_route:
                route.extend(vehicle_id_to_planned_route.get(vehicle_id))
            split_order_id_list.extend(Checker.__find_split_orders_in_vehicle_routes(route, carrying_items))

        for order_id in split_order_id_list:
            if order_id in id_to_order:
                order = id_to_order.get(order_id)
                if order.demand <= capacity:
                    logger.error(f"order {order.id} demand: {order.demand} <= {capacity}, we can not split this order.")
                    return False
        return True

    @staticmethod
    def __find_split_orders_from_vehicles(vehicle_id_to_item_list: dict):
        order_id_to_vehicle_ids = {}
        for vehicle_id, item_list in vehicle_id_to_item_list.items():
            order_id_list = []
            for item in item_list:
                order_id = item.order_id
                if order_id not in order_id_list:
                    order_id_list.append(order_id)
            logger.debug(f"Vehicle {vehicle_id} contains {len(order_id_list)} orders, {len(item_list)} order items")

            for order_id in order_id_list:
                if order_id not in order_id_to_vehicle_ids:
                    order_id_to_vehicle_ids[order_id] = []
                order_id_to_vehicle_ids[order_id].append(vehicle_id)

        split_order_ids = []
        for order_id, vehicle_ids in order_id_to_vehicle_ids.items():
            if len(vehicle_ids) > 1:
                split_order_ids.append(order_id)
        logger.debug(f"Find {len(split_order_ids)} split orders from vehicles")
        return split_order_ids

    @staticmethod
    def __find_split_orders_in_vehicle_routes(route, carrying_items):
        order_id_list = []
        split_order_ids = []
        while not carrying_items.is_empty():
            item = carrying_items.pop()
            order_id = item.order_id
            if order_id not in order_id_list:
                order_id_list.append(order_id)

        for node in route:
            tmp_order_id_list = []
            pickup_items = node.pickup_items
            for item in pickup_items:
                if item.order_id not in tmp_order_id_list:
                    tmp_order_id_list.append(item.order_id)
            for order_id in tmp_order_id_list:
                if order_id not in order_id_list:
                    order_id_list.append(order_id)
                else:
                    split_order_ids.append(order_id)
        logger.debug(f"find {len(split_order_ids)} split orders")
        return split_order_ids
