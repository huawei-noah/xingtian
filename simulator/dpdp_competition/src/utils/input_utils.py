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
import time

import pandas as pd

from src.common.factory import Factory
from src.common.order import Order, OrderItem
from src.common.route import Map
from src.common.route import RouteInfo
from src.common.vehicle import Vehicle
from src.conf.configs import Configs
from src.utils.logging_engine import logger


def get_initial_data(data_file_path: str, vehicle_info_file_path: str, route_info_file_path: str,
                     factory_info_file_path: str, initial_time: int):
    """
    获取模拟器的输入数据, get the input of simulator
    :param data_file_path: 订单数据文件路径, path of the file containing information of orders
    :param vehicle_info_file_path: 车辆数据文件路径,  path of the file containing information of vehicles
    :param route_info_file_path: 地图数据文件路径,  path of the file containing information of route map
    :param factory_info_file_path: 工厂数据文件路径,  path of the file containing information of factories
    :param initial_time: unix timestamp
    :return: id_to_order: dict, id_to_vehicle: dict, route_map, id_to_factory: dict
    """
    # 获取工厂信息, get the factories
    id_to_factory = get_factory_info(factory_info_file_path)
    logger.info(f"Get {len(id_to_factory)} factories")

    # 获取地图信息, get the route map containing distance and time matrix between factories
    code_to_route = get_route_map(route_info_file_path)
    logger.info(f"Get {len(code_to_route)} routes")
    route_map = Map(code_to_route)

    # 车辆基本信息, get the vehicles
    id_to_vehicle = get_vehicle_info(vehicle_info_file_path)
    logger.info(f"Get {len(id_to_vehicle)} vehicles")

    # 获取订单信息, get the orders
    id_to_order = get_order_info(data_file_path, initial_time)
    logger.info(f"Get {len(id_to_order)} orders")

    return id_to_order, id_to_vehicle, route_map, id_to_factory


def get_order_info(file_path: str, ini_time: int):
    order_df = pd.read_csv(file_path, dtype={'order_id': object})

    id_to_order = {}
    for index, row in order_df.iterrows():
        order_id = str(row['order_id'])
        q_standard = int(row['q_standard'])
        q_small = int(row['q_small'])
        q_box = int(row['q_box'])
        components = {Configs.STANDARD_PALLET_LABEL: q_standard,
                      Configs.SMALL_PALLET_LABEL: q_small,
                      Configs.BOX_LABEL: q_box}
        demand = float(row['demand'])
        load_time = int(row['load_time'])
        unload_time = int(row['unload_time'])
        pickup_id = str(row['pickup_id'])
        delivery_id = str(row['delivery_id'])

        ini_datetime = datetime.datetime.fromtimestamp(ini_time)
        creation_datetime = datetime.datetime.combine(
            ini_datetime.date(), datetime.datetime.strptime(row['creation_time'], '%H:%M:%S').time())
        creation_time = time.mktime(creation_datetime.timetuple())
        committed_completion_datetime = datetime.datetime.combine(
            ini_datetime.date(), datetime.datetime.strptime(row['committed_completion_time'], '%H:%M:%S').time())
        committed_completion_time = time.mktime(committed_completion_datetime.timetuple())
        if committed_completion_time < creation_time:
            committed_completion_time += Configs.A_DAY_TIME_SECONDS

        order = Order(order_id, components, demand, int(creation_time), int(committed_completion_time),
                      load_time, unload_time, delivery_id, pickup_id)
        item_list = get_item_list(order)
        order.item_list = item_list

        if order_id not in id_to_order:
            id_to_order[order_id] = order
    return id_to_order


def get_item_list(order):
    """
    get the items of order
    Item is the smallest unit of the order. Suppose an order contains 2 standard pallets, 1 small pallet and 1 box.
    The smallest units are 1 standard pallet or 1 small pallet or 1 box. The length of the item list is 2+1+1=4.
    :param order: object
    :return: item_list: list
    """
    item_list = []
    seq = 1
    for demand_label in Configs.PALLET_TYPE_LABELS:
        demand = Configs.LABEL_TO_DEMAND_UNIT.get(demand_label)
        load_time = int(demand / Configs.LOAD_SPEED * 60)
        unload_time = int(demand / Configs.UNLOAD_SPEED * 60)
        num = order.components.get(demand_label, 0)
        for i in range(num):
            item_id = f"{order.id}-{seq}"
            item_list.append(
                OrderItem(item_id, demand_label, order.id, demand, order.pickup_factory_id, order.delivery_factory_id,
                          order.creation_time, order.committed_completion_time, load_time, unload_time, order.delivery_state))
            seq += 1
    return item_list


def get_factory_info(file_path: str):
    df = pd.read_csv(file_path)
    id_to_factory = {}
    for index, row in df.iterrows():
        factory_id = str(row['factory_id'])
        lng = float(row['longitude'])
        lat = float(row['latitude'])
        dock_num = int(row['port_num'])
        factory = Factory(factory_id, lng, lat, dock_num)
        if factory_id not in id_to_factory:
            id_to_factory[factory_id] = factory
    return id_to_factory


def get_route_map(file_path: str):
    route_df = pd.read_csv(file_path)
    code_to_route = {}
    for index, row in route_df.iterrows():
        route_code = str(row['route_code'])
        start_factory_id = str(row['start_factory_id'])
        end_factory_id = str(row['end_factory_id'])
        distance = float(row['distance'])
        transport_time = int(row['time'])
        route = RouteInfo(route_code, start_factory_id, end_factory_id, distance, transport_time)
        if route_code not in code_to_route:
            code_to_route[route_code] = route
    return code_to_route


def get_vehicle_info(file_path: str):
    vehicle_df = pd.read_csv(file_path)
    id_to_vehicle = {}
    for index, row in vehicle_df.iterrows():
        car_num = str(row['car_num'])
        capacity = int(row['capacity'])
        operation_time = int(row['operation_time'])
        gps_id = str(row['gps_id'])
        vehicle = Vehicle(car_num, capacity, gps_id, operation_time)
        if car_num not in id_to_vehicle:
            id_to_vehicle[car_num] = vehicle
    return id_to_vehicle
