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
import simpy

from src.conf.configs import Configs
from src.utils.logging_engine import logger


class VehicleSimulator(object):
    def __init__(self, route_map, id_to_factory):
        self.env = simpy.Environment()
        self.factory_id_to_dock_resource = {}

        self.route_map = route_map
        self.id_to_factory = id_to_factory

        self.ongoing_item_ids = []
        self.completed_item_ids = []
        self.vehicle_id_to_destination = {}
        self.vehicle_id_to_cur_position_info = {}
        self.vehicle_id_to_carrying_items = {}

    def __ini_dock_resources_of_factories(self, id_to_factory: dict):
        self.factory_id_to_dock_resource = {}
        for factory_id, factory in id_to_factory.items():
            self.factory_id_to_dock_resource[factory_id] = simpy.Resource(self.env, capacity=factory.dock_num)

    def run(self, id_to_vehicle: dict, from_time: int):
        """
        :param id_to_vehicle:  total vehicles
        :param from_time: unit is second, start time of the simulator
        """
        # 初始化仿真环境, initial the simulation environment
        self.env = simpy.rt.RealtimeEnvironment(initial_time=from_time, factor=0.000000000001, strict=False)

        # 初始化各工厂的货口资源, initial the port resource of each factory
        self.__ini_dock_resources_of_factories(self.id_to_factory)

        # sort_vehicles
        sorted_vehicles = self.__sort_vehicles(id_to_vehicle, from_time)

        # Each vehicle starts to visit its route
        for vehicle in sorted_vehicles:
            self.env.process(self.work(vehicle))
        self.env.run()

    # Visiting process of each vehicle
    def work(self, vehicle):
        cur_factory_id = vehicle.cur_factory_id
        # 当前在工厂: 1. 还处于装卸过程(占用货口资源); 2. 停车状态(不占用货口资源)
        if len(cur_factory_id) > 0:
            # 1.还处于装卸过程(占用货口资源);
            if vehicle.leave_time_at_current_factory > self.env.now:
                resource = self.factory_id_to_dock_resource.get(cur_factory_id)
                with resource.request() as req:
                    yield req
                    yield self.env.timeout(vehicle.leave_time_at_current_factory - self.env.now)

            # 2. 停车状态(不占用货口资源)
            else:
                vehicle.leave_time_at_current_factory = self.env.now

        if vehicle.destination is None:
            if len(cur_factory_id) == 0:
                logger.error(f"Vehicle {vehicle.id}: both the current factory and the destination are None!!!")
            return

        if len(cur_factory_id) > 0:
            next_factory_id = vehicle.destination.id
            transport_time = self.route_map.calculate_transport_time_between_factories(cur_factory_id, next_factory_id)
            yield self.env.timeout(transport_time)
        else:
            # driving towards destination
            arr_time = vehicle.destination.arrive_time
            if arr_time >= self.env.now:
                yield self.env.timeout(arr_time - self.env.now)
            else:
                logger.error(f"Vehicle {vehicle.id} is driving toward the destination, "
                             f"however current time {datetime.datetime.fromtimestamp(self.env.now)} is greater than "
                             f"the arrival time {datetime.datetime.fromtimestamp(arr_time)} of destination!!!")

        vehicle.destination.arrive_time = self.env.now
        service_time = vehicle.destination.service_time
        cur_factory_id = vehicle.destination.id
        resource = self.factory_id_to_dock_resource.get(cur_factory_id)
        with resource.request() as req:
            yield req
            yield self.env.timeout(service_time + Configs.DOCK_APPROACHING_TIME)
        vehicle.destination.leave_time = self.env.now

        # driving towards the left nodes
        for node in vehicle.planned_route:
            next_factory_id = node.id

            # 计算运输时间
            transport_time = self.route_map.calculate_transport_time_between_factories(cur_factory_id, next_factory_id)
            yield self.env.timeout(transport_time)

            # 计算服务时间
            arr_time = self.env.now
            service_time = node.service_time
            resource = self.factory_id_to_dock_resource.get(next_factory_id)
            with resource.request() as req:
                yield req
                yield self.env.timeout(service_time + Configs.DOCK_APPROACHING_TIME)
            leave_time = self.env.now

            node.arrive_time = arr_time
            node.leave_time = leave_time
            cur_factory_id = next_factory_id

    @staticmethod
    def __sort_vehicles(id_to_vehicle: dict, start_time: int):
        factory_id_to_vehicles = {}
        for vehicle_id, vehicle in id_to_vehicle.items():
            if len(vehicle.cur_factory_id) > 0 and vehicle.leave_time_at_current_factory > start_time:
                factory_id = vehicle.cur_factory_id
                if factory_id not in factory_id_to_vehicles:
                    factory_id_to_vehicles[factory_id] = []
                factory_id_to_vehicles[factory_id].append(vehicle)

        sorted_vehicle_ids = []
        for factory_id, vehicles in factory_id_to_vehicles.items():
            tmp_dt = [(vehicle.id, vehicle.leave_time_at_current_factory) for vehicle in vehicles]
            tmp_dt.sort(key=lambda x: x[1])
            for dt in tmp_dt:
                sorted_vehicle_ids.append(dt[0])

        for vehicle_id in id_to_vehicle.keys():
            if vehicle_id not in sorted_vehicle_ids:
                sorted_vehicle_ids.append(vehicle_id)

        sorted_vehicles = [id_to_vehicle.get(vehicle_id) for vehicle_id in sorted_vehicle_ids]
        return sorted_vehicles

    # 解析输出, 加快照
    def parse_simulation_result(self, id_to_vehicle: dict, to_time: int):
        self.ongoing_item_ids = []
        self.completed_item_ids = []
        self.vehicle_id_to_destination = {}
        self.vehicle_id_to_cur_position_info = {}
        self.vehicle_id_to_carrying_items = {}

        self.get_position_info_of_vehicles(id_to_vehicle, to_time)
        self.get_destination_of_vehicles(id_to_vehicle, to_time)
        self.get_loading_and_unloading_result_of_vehicles(id_to_vehicle, to_time)

    def get_position_info_of_vehicles(self, id_to_vehicle: dict, to_time: int):
        for vehicle_id, vehicle in id_to_vehicle.items():
            cur_factory_id = ""
            arrive_time_at_current_factory = 0
            leave_time_at_current_factory = 0

            # still in current position
            if len(vehicle.cur_factory_id) > 0:
                if vehicle.leave_time_at_current_factory >= to_time or vehicle.destination is None:
                    cur_factory_id = vehicle.cur_factory_id
                    arrive_time_at_current_factory = vehicle.arrive_time_at_current_factory
                    leave_time_at_current_factory = max(vehicle.leave_time_at_current_factory, to_time)

            # in the following node
            if len(cur_factory_id) == 0:
                if vehicle.destination is None:
                    logger.error(f"Vehicle {vehicle_id}, the current position {vehicle.position_info.cur_factory_id}, "
                                 f"the destination is None")
                    continue

                if vehicle.destination.arrive_time > to_time:
                    cur_factory_id = ""
                elif vehicle.destination.leave_time > to_time:
                    cur_factory_id = vehicle.destination.id
                    arrive_time_at_current_factory = vehicle.destination.arrive_time
                    leave_time_at_current_factory = vehicle.destination.leave_time
                else:
                    if len(vehicle.planned_route) == 0:
                        cur_factory_id = vehicle.destination.id
                        arrive_time_at_current_factory = vehicle.destination.arrive_time
                        leave_time_at_current_factory = vehicle.destination.leave_time
                    else:
                        cur_factory_id = ""
                        for node in vehicle.planned_route:
                            if node.arrive_time <= to_time <= node.leave_time:
                                cur_factory_id = node.id
                                arrive_time_at_current_factory = node.arrive_time
                                leave_time_at_current_factory = node.leave_time
                                break

                        # Considering boundary conditions
                        if len(cur_factory_id) == 0 and vehicle.planned_route[-1].leave_time < to_time:
                            cur_factory_id = vehicle.planned_route[-1].id
                            arrive_time_at_current_factory = vehicle.planned_route[-1].arrive_time
                            leave_time_at_current_factory = vehicle.planned_route[-1].leave_time

            self.vehicle_id_to_cur_position_info[vehicle_id] = {"cur_factory_id": cur_factory_id,
                                                                "arrive_time_at_current_factory": arrive_time_at_current_factory,
                                                                "leave_time_at_current_factory": leave_time_at_current_factory,
                                                                "update_time": to_time}

    def get_destination_of_vehicles(self, id_to_vehicle: dict, to_time: int):
        for vehicle_id, vehicle in id_to_vehicle.items():
            if vehicle.destination is None:
                self.vehicle_id_to_destination[vehicle_id] = None
                continue

            if vehicle.destination.arrive_time > to_time:
                self.vehicle_id_to_destination[vehicle_id] = vehicle.destination
            else:
                destination = None
                for node in vehicle.planned_route:
                    if node.arrive_time > to_time:
                        destination = node
                        break
                self.vehicle_id_to_destination[vehicle_id] = destination

    def get_loading_and_unloading_result_of_vehicles(self, id_to_vehicle: dict, to_time: int):
        for vehicle_id, vehicle in id_to_vehicle.items():
            carrying_items = vehicle.carrying_items

            if vehicle.destination is None:
                self.vehicle_id_to_carrying_items[vehicle_id] = carrying_items
                continue

            if vehicle.destination.arrive_time <= to_time:
                self.loading_and_unloading(vehicle.destination, carrying_items,
                                           self.completed_item_ids, self.ongoing_item_ids)

            for node in vehicle.planned_route:
                arr_time = node.arrive_time
                leave_time = node.leave_time
                if arr_time <= to_time:
                    self.loading_and_unloading(node, carrying_items, self.completed_item_ids, self.ongoing_item_ids)
                if leave_time > to_time:
                    break

            self.vehicle_id_to_carrying_items[vehicle_id] = carrying_items

    @staticmethod
    def loading_and_unloading(node, carrying_items, completed_item_ids, ongoing_item_ids):
        delivery_items = node.delivery_items
        pickup_items = node.pickup_items
        for item in delivery_items:
            carrying_items.pop()
            completed_item_ids.append(item.id)
        for item in pickup_items:
            carrying_items.push(item)
            ongoing_item_ids.append(item.id)
