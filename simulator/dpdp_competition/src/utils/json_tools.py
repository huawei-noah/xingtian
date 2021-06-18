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

import json
import os
import platform
import subprocess
import sys
import time
from importlib import import_module

from src.common.node import Node
from src.common.vehicle import Vehicle
from src.conf.configs import Configs
from src.utils.logging_engine import logger

COMMON_CLASS = {'Vehicle': 'src.common.vehicle',
                'Order': 'src.common.order',
                'OrderItem': 'src.common.order',
                'Node': 'src.common.node',
                'Stack': 'src.common.stack',
                'Factory': 'src.common.factory'
                }


# 通过类的名称导入common类的数据结构
def import_common_class(class_name):
    module = import_module(COMMON_CLASS.get(class_name))
    return getattr(module, class_name)


""" Schedule the algorithm"""


def get_algorithm_calling_command():
    files = os.listdir(Configs.root_folder_path)
    for file in files:
        # 调度算法的入口文件必须以main_algorithm开头
        if file.startswith(Configs.ALGORITHM_ENTRY_FILE_NAME):
            end_name = file.split('.')[-1]
            algorithm_language = Configs.ALGORITHM_LANGUAGE_MAP.get(end_name)
            if algorithm_language == 'python':
                return 'python {}'.format(file)
            elif algorithm_language == 'java':
                return 'java {}'.format(file.split('.')[0])
            # c和c++调用方式一样，但系统不同调用方式有异
            elif algorithm_language == 'c':
                system = platform.system()
                if system == 'Windows':
                    return file
                elif system == 'Linux':
                    os.system(f'chmod 777 {file}')
                    return './{}'.format(file)
    logger.error('Can not find main_algorithm file.')
    sys.exit(-1)


# 开启进程，调用算法
def subprocess_function(cmd):
    # 开启子进程，并连接到它们的输入/输出/错误管道，获取返回值
    sub_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    try:
        start_time = time.time()
        # 设置超时
        sub_process.wait(Configs.MAX_RUNTIME_OF_ALGORITHM)
        end_time = time.time()
        # 返回算法运行时间和算法返回值
        return end_time - start_time, sub_process.stdout.read().decode()
    except Exception as e:
        logger.error(e)
        sys.exit(-1)


""" IO"""


def read_json_from_file(file_name):
    with open(file_name, 'r') as fd:
        data = fd.read()
    return json.loads(data)


def write_json_to_file(file_name, data):
    with open(file_name, 'w') as fd:
        fd.write(json.dumps(data, indent=4))


""" create the input of the algorithm (output json of simulation)"""


# 输出input_info数据到input.json
def convert_input_info_to_json_files(input_info):
    vehicle_info_list = __get_vehicle_info_list(input_info.id_to_vehicle)
    write_json_to_file(Configs.algorithm_vehicle_input_info_path, vehicle_info_list)

    unallocated_order_items = convert_dict_to_list(input_info.id_to_unallocated_order_item)
    write_json_to_file(Configs.algorithm_unallocated_order_items_input_path, unallocated_order_items)

    ongoing_order_items = convert_dict_to_list(input_info.id_to_ongoing_order_item)
    write_json_to_file(Configs.algorithm_ongoing_order_items_input_path, ongoing_order_items)


def __get_vehicle_info_list(id_to_vehicle: dict):
    vehicle_info_list = []
    for vehicle_id, vehicle in id_to_vehicle.items():
        vehicle_info_list.append(__convert_vehicle_to_dict(vehicle))
    return vehicle_info_list


def __convert_vehicle_to_dict(vehicle):
    carrying_items = vehicle.get_loading_sequence()

    vehicle_property = {
        "id": vehicle.id,
        "operation_time": vehicle.operation_time,
        "capacity": vehicle.board_capacity,
        "gps_id": vehicle.gps_id,
        "update_time": vehicle.gps_update_time,
        "cur_factory_id": vehicle.cur_factory_id,
        "arrive_time_at_current_factory": vehicle.arrive_time_at_current_factory,
        "leave_time_at_current_factory": vehicle.leave_time_at_current_factory,
        "carrying_items": [item.id for item in carrying_items],
        "destination": __convert_destination_to_dict(vehicle.destination)
    }
    return vehicle_property


def __convert_destination_to_dict(destination):
    if destination is None:
        return None

    destination_property = {
        'factory_id': destination.id,
        'delivery_item_list': [item.id for item in destination.delivery_items],
        'pickup_item_list': [item.id for item in destination.pickup_items],
        'arrive_time': destination.arrive_time,
        'leave_time': destination.leave_time}
    return destination_property


# 实例的属性转数组
def convert_dict_to_list(_dict):
    _list = []
    for key, value in _dict.items():
        if hasattr(value, '__dict__'):
            d = value.__dict__
            _list.append({key: d[key] for key in d if "__" not in key})
    return _list


""" Read the input of the algorithm (read the output json of simulator)"""


def get_vehicle_instance_dict(vehicle_infos: list, id_to_order_item: dict, id_to_factory: dict):
    id_to_vehicle = {}
    for vehicle_info in vehicle_infos:
        vehicle_id = vehicle_info.get("id")
        operation_time = vehicle_info.get("operation_time")
        capacity = vehicle_info.get("capacity")
        gps_id = vehicle_info.get("gps_id")
        carrying_item_id_list = vehicle_info.get("carrying_items")
        carrying_items = [id_to_order_item.get(item_id) for item_id in carrying_item_id_list
                          if item_id in carrying_item_id_list]
        destination = vehicle_info.get("destination")
        if destination is not None:
            destination = __get_destination(destination, id_to_factory, id_to_order_item)

        cur_factory_id = vehicle_info.get("cur_factory_id")
        arrive_time_at_current_factory = vehicle_info.get("arrive_time_at_current_factory")
        leave_time_at_current_factory = vehicle_info.get("leave_time_at_current_factory")
        update_time = vehicle_info.get("update_time")

        logger.debug(f"Get vehicle {vehicle_id} instance from json, "
                     f"item id list = {len(carrying_item_id_list)},"
                     f"item list = {len(carrying_items)}")
        if vehicle_id not in id_to_vehicle:
            vehicle = Vehicle(vehicle_id, capacity, gps_id, operation_time, carrying_items)
            vehicle.destination = destination
            vehicle.set_cur_position_info(cur_factory_id, update_time,
                                          arrive_time_at_current_factory, leave_time_at_current_factory)
            id_to_vehicle[vehicle_id] = vehicle
    return id_to_vehicle


def __get_destination(_dict, id_to_factory: dict, id_to_order_item: dict):
    factory_id = _dict.get("factory_id")
    factory = id_to_factory.get(factory_id)
    delivery_item_ids = _dict.get("delivery_item_list")
    delivery_items = [id_to_order_item.get(item_id) for item_id in delivery_item_ids]
    pickup_item_ids = _dict.get("pickup_item_list")
    pickup_items = [id_to_order_item.get(item_id) for item_id in pickup_item_ids]
    arr_time = _dict.get("arrive_time")
    leave_time = _dict.get("leave_time")
    return Node(factory_id, factory.lng, factory.lat, pickup_items, delivery_items, arr_time, leave_time)


def get_order_item_dict(_item_list, class_name):
    items = convert_dicts_list_to_instances_list(_item_list, class_name)
    id_to_order_item = {}
    for item in items:
        if item.id not in id_to_order_item:
            id_to_order_item[item.id] = item
    return id_to_order_item


# 字典列表转换为实例列表
def convert_dicts_list_to_instances_list(_dicts_list, class_name):
    instances_list = []
    # 通过类名取导入类
    for _dict in _dicts_list:
        common_class = import_common_class(class_name)
        instance = common_class.__new__(common_class)
        instance.__dict__ = _dict
        instances_list.append(instance)
    return instances_list


""" Output the result of the algorithm in json files"""


# 把Node实例属性转为可用于实例化的参数字典
def convert_nodes_to_json(vehicle_id_to_nodes):
    result_dict = {}
    for key, value in vehicle_id_to_nodes.items():
        if value is None:
            result_dict[key] = None
            continue

        # 字典的情况
        if hasattr(value, '__dict__'):
            result_dict[key] = convert_node_to_json(value)
        # 列表的情况
        elif not value:
            result_dict[key] = []
        elif value and isinstance(value, list) and hasattr(value[0], '__dict__'):
            result_dict[key] = [convert_node_to_json(node) for node in value]
    return result_dict


# 把Node实例的属性转为可用于后续实例化的参数
def convert_node_to_json(node):
    node_property = {'factory_id': node.id, 'lng': node.lng, 'lat': node.lat,
                     'delivery_item_list': [item.id for item in node.delivery_items],
                     'pickup_item_list': [item.id for item in node.pickup_items],
                     'arrive_time': node.arrive_time,
                     'leave_time': node.leave_time}
    return node_property


""" Read the output json files of the algorithm"""


# 从output.json获取数据，并进行数据结构转换
def get_output_of_algorithm(id_to_order_item: dict):
    vehicle_id_to_destination_from_json = read_json_from_file(Configs.algorithm_output_destination_path)
    vehicle_id_to_destination = __convert_json_to_nodes(vehicle_id_to_destination_from_json, id_to_order_item)
    vehicle_id_to_planned_route_from_json = read_json_from_file(Configs.algorithm_output_planned_route_path)
    vehicle_id_to_planned_route = __convert_json_to_nodes(vehicle_id_to_planned_route_from_json, id_to_order_item)
    return vehicle_id_to_destination, vehicle_id_to_planned_route


def __convert_json_to_nodes(vehicle_id_to_nodes_from_json: dict, id_to_order_item: dict):
    result_dict = {}

    node_class = import_common_class('Node')
    for key, value in vehicle_id_to_nodes_from_json.items():
        if value is None:
            result_dict[key] = None
            continue
        if isinstance(value, dict):
            __get_order_item(value, id_to_order_item)
            # 传入参数创建新的Node实例
            result_dict[key] = Node(**value)
        elif not value:
            result_dict[key] = []
        elif isinstance(value, list):
            node_list = []
            for node in value:
                __get_order_item(node, id_to_order_item)
                node_list.append(node_class(**node))
            result_dict[key] = node_list
    return result_dict


# 根据order_item的id找到实例
def __get_order_item(node, id_to_order_item):
    node['delivery_item_list'] = [id_to_order_item.get(item_id) for item_id in node['delivery_item_list']]
    node['pickup_item_list'] = [id_to_order_item.get(item_id) for item_id in node['pickup_item_list']]
