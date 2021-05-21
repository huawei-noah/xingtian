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
import random
import time
import traceback

from src.conf.configs import Configs
from src.simulator.simulate_environment import SimulateEnvironment
from src.utils.input_utils import get_initial_data
from src.utils.logging_engine import logger


def __initialize(factory_info_file_name: str, route_info_file_name: str, instance_folder: str):
    """
    模拟器初始化, Initialize the simulator
    :param factory_info_file_name: 工厂数据文件名, name of the file containing information of factories
    :param route_info_file_name: 地图数据文件名, name of the file containing information of route map
    :param instance_folder: 测试例对应的文件夹, folder name of the instance
    :return: SimulateEnvironment
    """
    route_info_file_path = os.path.join(Configs.benchmark_folder_path, route_info_file_name)
    factory_info_file_path = os.path.join(Configs.benchmark_folder_path, factory_info_file_name)
    instance_folder_path = os.path.join(Configs.benchmark_folder_path, instance_folder)

    # 车辆数据文件名, name of the file containing information of vehicles
    vehicle_info_file_path = ""
    # 订单数据文件名, name of the file containing information of orders
    data_file_path = ""
    for file_name in os.listdir(instance_folder_path):
        if file_name.startswith("vehicle"):
            vehicle_info_file_path = os.path.join(instance_folder_path, file_name)
        else:
            data_file_path = os.path.join(instance_folder_path, file_name)

    # 初始化时间, initial the start time of simulator
    now = datetime.datetime.now()
    initial_datetime = datetime.datetime(now.year, now.month, now.day)
    initial_time = int(time.mktime(initial_datetime.timetuple()))
    time_interval = Configs.ALG_RUN_FREQUENCY * 60
    logger.info(f"Start time of the simulator: {initial_datetime}, time interval: {time_interval: .2f}")

    try:
        # 获取初始化数据, get the input
        id_to_order, id_to_vehicle, route_map, id_to_factory = get_initial_data(data_file_path,
                                                                                vehicle_info_file_path,
                                                                                route_info_file_path,
                                                                                factory_info_file_path,
                                                                                initial_time)
        # 初始化车辆位置, set the initial position of vehicles
        __initial_position_of_vehicles(id_to_factory, id_to_vehicle, initial_time)

        # return the instance of the object SimulateEnvironment
        return SimulateEnvironment(initial_time, time_interval, id_to_order, id_to_vehicle, id_to_factory, route_map)
    except Exception as exception:
        logger.error("Failed to read initial data")
        logger.error(f"Error: {exception}, {traceback.format_exc()}")
        return None


def __initial_position_of_vehicles(id_to_factory: dict, id_to_vehicle: dict, ini_time: int):
    factory_id_list = [*id_to_factory]
    random.seed(Configs.RANDOM_SEED)
    for vehicle_id, vehicle in id_to_vehicle.items():
        index = random.randint(0, len(factory_id_list) - 1)
        factory_id = factory_id_list[index]
        vehicle.set_cur_position_info(factory_id, ini_time, ini_time, ini_time)
        logger.info(f"Initial position of {vehicle_id} is {factory_id}")


def simulate(factory_info_file: str, route_info_file: str, instance: str):
    simulate_env = __initialize(factory_info_file, route_info_file, instance)
    if simulate_env is not None:
        # 模拟器仿真过程
        simulate_env.run()
    return simulate_env.total_score
