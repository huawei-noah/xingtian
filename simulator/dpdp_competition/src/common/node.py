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


class Node(object):
    def __init__(self, factory_id: str, lng: float, lat: float, pickup_item_list: list, delivery_item_list: list,
                 arrive_time=0, leave_time=0):
        """
        车辆途经的节点
        :param factory_id: 对应的工厂id
        :param lng: 经度
        :param lat: 纬度
        """
        self.__id = factory_id
        self.__lng = lng
        self.__lat = lat

        # 送货清单, list of delivery items
        self.__delivery_items = delivery_item_list
        self.__unloading_time = self.calculate_unloading_time()

        # 取货清单, list of pickup items
        self.__pickup_items = pickup_item_list
        self.__loading_time = self.calculate_loading_time()

        """
        unloading and loading sequence of this node
        1. unloading items (Given the length of delivery_items is n)
        delivery_items[0], delivery_items[1], ..., delivery_items[n-1]
        2. loading items (Given the length of pickup_items is n)
        pickup_items[0], pickup_items[1], ..., pickup_items[n-1]
        """

        self.__service_time = self.__loading_time + self.__unloading_time

        self.arrive_time = arrive_time
        self.leave_time = leave_time

    def calculate_loading_time(self):
        loading_time = 0
        for item in self.__pickup_items:
            loading_time += item.load_time
        return loading_time

    def calculate_unloading_time(self):
        unloading_time = 0
        for item in self.__delivery_items:
            unloading_time += item.unload_time
        return unloading_time

    def update_service_time(self):
        self.__loading_time = self.calculate_loading_time()
        self.__unloading_time = self.calculate_unloading_time()
        self.__service_time = self.__unloading_time + self.__loading_time

    @property
    def id(self):
        return self.__id

    @property
    def lng(self):
        return self.__lng

    @property
    def lat(self):
        return self.__lat

    @property
    def service_time(self):
        return self.__service_time

    @property
    def pickup_items(self):
        return self.__pickup_items

    @pickup_items.setter
    def pickup_items(self, pickup_item_list):
        self.__pickup_items = pickup_item_list

    @property
    def delivery_items(self):
        return self.__delivery_items

    @delivery_items.setter
    def delivery_items(self, delivery_item_list):
        self.__delivery_items = delivery_item_list
