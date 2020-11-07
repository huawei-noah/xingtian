# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Metric of super resolution task."""
from mindspore.nn.metrics import Metric
from zeus.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.METRIC)
class SRMetric(Metric):
    """Calculate IoU between output and target."""

    __metric_name__ = 'SRMetric'

    def __init__(self, method='psnr', to_y=True, scale=2, max_rgb=1):
        self.method = method
        self.to_y = to_y
        self.scale = scale
        self.max_rgb = max_rgb

    def update(self, *inputs):
        """Update the metric."""
        if len(inputs) != 2:
            raise ValueError('SRMetric need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self.result = self.compute_metric(y_pred, y)

    def eval(self):
        """Get the metric."""
        return self.result

    def clear(self):
        """Reset the metric."""
        self.result = 0

    def compute_metric(self, output, target):
        """Compute sr metric."""
        # TODO
        return 0

    @property
    def objective(self):
        """Define reward mode, default is max."""
        return 'MAX'

    def __call__(self, output, target, *args, **kwargs):
        """Forward and calculate accuracy."""
        return self
