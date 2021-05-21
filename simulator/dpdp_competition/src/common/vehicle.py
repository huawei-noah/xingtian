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

from src.common.stack import Stack


class Vehicle(object):
    def __init__(self, car_num: str, capacity: int, gps_id: str, operation_time: int, carrying_items=None):
        """
        :param car_num: 车牌号, id of the vehicle
        :param capacity: unit is standard pallet, e.g. 15
        :param gps_id: gps的设备编号
        :param operation_time: namely the work shift of the vehicle, unit is hour, e.g. 24
        :param carrying_items: in the order of loading, 车辆当前载的物料列表，顺序为装货顺序
        """
        self.id = car_num
        self.operation_time = operation_time  # Unit: hour, 12h or 24h
        self.board_capacity = capacity
        self.gps_id = gps_id

        # 车辆当前载的货物, Carrying items of the vehicle consider the Last-in-first-out constraint
        if carrying_items is None:
            carrying_items = []
        self.__carrying_items = self.ini_carrying_items(carrying_items)

        """
        Sequence of visiting factories: 
            current_factory (If the vehicle is in transit, this value is "".) 
            ——> destination (next node to be visited, can not be changed after confirmation)
            ——> planned_route (node list, can be changed)  
        """
        self.gps_update_time = 0  # update time of current position, unix timestamp, unit is second

        # cur_factory_id: 车辆当前所在的工厂, 如果当前不在任何工厂则为""
        # the factory id where the vehicle is currently located. If the vehicle currently is not in any factory, the value is ""
        self.cur_factory_id = ""
        # 车辆到达和离开当前所在工厂的时间
        self.arrive_time_at_current_factory = 0
        self.leave_time_at_current_factory = 0

        # 车辆当前的目的地, 目的地一旦确定不可更改, 直到车辆到达
        # We cannot change the current destination of the vehicle until the vehicle has arrived
        self.destination = None

        # 车辆的规划路径, route is node list, not containing the destination
        self.planned_route = []

    @property
    def carrying_items(self):
        return self.__carrying_items

    @carrying_items.setter
    def carrying_items(self, carrying_items):
        self.__carrying_items = carrying_items

    @staticmethod
    def ini_carrying_items(carrying_items: list):
        stack = Stack()
        for item in carrying_items:
            stack.push(item)
        return stack

    def add_item(self, item):
        self.__carrying_items.push(item)

    def unload_item(self):
        if self.__carrying_items.is_empty():
            return None
        return self.__carrying_items.pop()

    def get_unloading_sequence(self):
        unloading_sequence = []
        copy_carrying_items = copy.deepcopy(self.__carrying_items)
        while not copy_carrying_items.is_empty():
            unloading_sequence.append(copy_carrying_items.pop())
        return unloading_sequence

    def get_loading_sequence(self):
        unloading_sequence = self.get_unloading_sequence()
        loading_sequence = []
        for i in range(len(unloading_sequence)):
            index = len(unloading_sequence) - 1 - i
            loading_sequence.append(unloading_sequence[index])
        return loading_sequence

    def set_cur_position_info(self, cur_factory_id, update_time: int, arrive_time_at_current_factory=0, leave_time_at_current_factory=0):
        self.cur_factory_id = cur_factory_id
        self.gps_update_time = update_time
        if len(self.cur_factory_id) > 0:
            self.arrive_time_at_current_factory = arrive_time_at_current_factory
            self.leave_time_at_current_factory = leave_time_at_current_factory
        else:
            self.arrive_time_at_current_factory = 0
            self.leave_time_at_current_factory = 0

    def __str__(self):
        return "[{}:{}]".format(self.__class__.__name__, self.gather_attrs())

    def gather_attrs(self):
        return ",".join("{}={}".format(k, getattr(self, k)) for k in self.__dict__.keys())
