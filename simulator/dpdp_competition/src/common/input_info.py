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


class InputInfo(object):
    def __init__(self, id_to_unallocated_order_item: dict, id_to_ongoing_order_item: dict, id_to_vehicle: dict,
                 id_to_factory: dict, route_map):
        """
        Input of the algorithm
        汇总所有的输入信息, 包括车辆信息(车辆当前的位置信息&装载货物), 订单信息(待分配和进行中), 路网信息
        :param id_to_unallocated_order_item: item_id ——> OrderItem object(state: "GENERATED")
        :param id_to_ongoing_order_item: item_id ——> OrderItem object(state: "ONGOING")
        :param id_to_vehicle: vehicle_id ——> Vehicle object
        :param id_to_factory: factory_id ——> factory object
        :param route_map: distance and time matrix between factories
        """
        self.id_to_unallocated_order_item = id_to_unallocated_order_item
        self.id_to_ongoing_order_item = id_to_ongoing_order_item
        self.id_to_vehicle = id_to_vehicle
        self.id_to_factory = id_to_factory
        self.route_map = route_map
