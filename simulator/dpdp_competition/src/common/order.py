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

from src.conf.configs import Configs
from src.utils.logging_engine import logger


class Order(object):
    def __init__(self, order_id: str, components: dict, demand: float,
                 creation_time: int, committed_completion_time: int, loading_time: int, unloading_time: int,
                 delivery_factory_id: str, pickup_factory_id: str, delivery_state=0):
        """
        :param order_id: 订单编号
        :param components: 订单包含的物料, components of the order, e.g. {'PALLET': 1, 'HALF_PALLET': 1, 'BOX': 2}
        :param demand: 货物总量, total cargo number, unit is standard pallet
        :param creation_time: unix timestamp, unit is second
        :param committed_completion_time: unix timestamp, unit is second
        :param loading_time: unit is second
        :param unloading_time: unit is second
        :param delivery_factory_id: 送货地
        :param pickup_factory_id: 取货地
        :param delivery_state: state of the order, 0: 初始(initialization), 1: 已生成(generated), 2: 进行中(ongoing), 3: 完成(Completed)
        """
        self.id = order_id
        self.creation_time = creation_time
        self.committed_completion_time = committed_completion_time
        self.load_time = loading_time
        self.unload_time = unloading_time
        self.pickup_factory_id = pickup_factory_id
        self.delivery_factory_id = delivery_factory_id
        self.components = components
        self.demand = demand
        self.item_list = []
        self.delivery_state = int(delivery_state)
        logger.debug(f"{order_id}, creation time: {datetime.datetime.fromtimestamp(creation_time)}, "
                     f"committed completion time: {datetime.datetime.fromtimestamp(committed_completion_time)}, "
                     f"components: {self.components.get(Configs.STANDARD_PALLET_LABEL, 0)} standard pallets, "
                     f"{self.components.get(Configs.SMALL_PALLET_LABEL, 0)} small pallets, "
                     f"{self.components.get(Configs.BOX_LABEL, 0)} boxes, "
                     f"pickup factory id: {delivery_factory_id}, delivery factory id: {pickup_factory_id}")

    def update_state(self):
        INI_STATE = 100
        min_state = INI_STATE
        for item in self.item_list:
            if item.delivery_state < min_state:
                min_state = item.delivery_state
        if min_state < INI_STATE:
            self.delivery_state = min_state


class OrderItem(object):
    def __init__(self, item_id: str, item_type: str, order_id: str, demand: float,
                 pickup_factory_id: str, delivery_factory_id: str, creation_time: int, committed_completion_time: int,
                 loading_time: int, unloading_time: int, delivery_state=0):
        """
        订单内的物料, Item of the order
        :param item_id: 物料编号
        :param item_type: 物料对应托盘类型, e.g. "PALLET", "HALF_PALLET", "BOX"
        :param order_id: 订单编号
        :param demand: unit is standard pallet
        :param creation_time: unix timestamp, unit is second
        :param committed_completion_time: unix timestamp, unit is second
        :param pickup_factory_id: 取货地
        :param delivery_factory_id: 送货地
        :param loading_time: 装货时间, unit is second
        :param unloading_time: 卸货时间, unit is second
        :param delivery_state: state of the item, 0: 初始(initialization), 1: 已生成(generated), 2: 进行中(ongoing), 3: 完成(Completed)
        """
        self.id = item_id
        self.type = item_type
        self.order_id = order_id
        self.demand = demand
        self.pickup_factory_id = pickup_factory_id
        self.delivery_factory_id = delivery_factory_id
        self.creation_time = creation_time
        self.committed_completion_time = committed_completion_time
        self.load_time = loading_time
        self.unload_time = unloading_time
        self.delivery_state = delivery_state
