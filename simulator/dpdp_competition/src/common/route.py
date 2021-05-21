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

import sys

from src.utils.logging_engine import logger


class RouteInfo(object):
    def __init__(self, route_code: str, start_factory_id: str, end_factory_id: str, distance: float, time: float):
        """
        :param route_code: 路线编码
        :param start_factory_id: 起始的工厂编号
        :param end_factory_id: 结束的工厂编号
        :param distance: unit is km
        :param time: unit is second
        """
        self.route_code = str(route_code)
        self.start_factory_id = str(start_factory_id)
        self.end_factory_id = str(end_factory_id)
        self.distance = float(distance)
        self.time = int(time)


class Map(object):
    def __init__(self, code_to_route):
        self.__code_to_route = code_to_route
        # distance between factories, unit is km
        self.__factory_id_pair_to_distance = self.__get_distance_matrix_between_factories()
        # time between factories, unit is second
        self.__factory_id_pair_to_time = self.__get_time_matrix_between_factories()

    def __get_distance_matrix_between_factories(self):
        if len(self.__code_to_route) == 0:
            return

        dist_mat = {}
        for route_code, route in self.__code_to_route.items():
            if (route.start_factory_id, route.end_factory_id) not in dist_mat:
                dist_mat[(route.start_factory_id, route.end_factory_id)] = route.distance
        return dist_mat

    def __get_time_matrix_between_factories(self):
        if len(self.__code_to_route) == 0:
            return

        time_mat = {}
        for route_code, route in self.__code_to_route.items():
            if (route.start_factory_id, route.end_factory_id) not in time_mat:
                time_mat[(route.start_factory_id, route.end_factory_id)] = route.time
        return time_mat

    def calculate_distance_between_factories(self, src_factory_id, dest_factory_id):
        if src_factory_id == dest_factory_id:
            return 0

        if (src_factory_id, dest_factory_id) in self.__factory_id_pair_to_distance:
            return self.__factory_id_pair_to_distance.get((src_factory_id, dest_factory_id))
        else:
            logger.error(f"({src_factory_id}, {dest_factory_id}) is not in distance matrix")
            return sys.maxsize

    def calculate_transport_time_between_factories(self, src_factory_id, dest_factory_id):
        if src_factory_id == dest_factory_id:
            return 0
        if (src_factory_id, dest_factory_id) in self.__factory_id_pair_to_time:
            return self.__factory_id_pair_to_time.get((src_factory_id, dest_factory_id))
        else:
            logger.error(f"({src_factory_id}, {dest_factory_id}) is not in time matrix")
            return sys.maxsize
