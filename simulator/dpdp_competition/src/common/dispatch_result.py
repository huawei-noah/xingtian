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


class DispatchResult(object):
    """ Output of the dispatching algorithm"""

    def __init__(self, vehicle_id_to_destination: dict, vehicle_id_to_planned_route: dict):
        """
        :param vehicle_id_to_destination: key is the vehicle id, value is the destination(next node) of the vehicle
        :param vehicle_id_to_planned_route: key is the vehicle id, value is the planned route(except the destination) of vehicle, namely node list

        """
        # Have to return the destination of each vehicle. If the vehicle has no destination (stand-by, e.t.c), the destination is None
        self.vehicle_id_to_destination = vehicle_id_to_destination
        # Have to return the planned route of each vehicle. Set the value [] when the vehicle has no planned route
        self.vehicle_id_to_planned_route = vehicle_id_to_planned_route
