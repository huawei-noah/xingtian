# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Valid Filter."""
import logging
import zeus
from zeus.common import ClassFactory, ClassType
from .filter_terminate_base import FilterTerminateBase


@ClassFactory.register(ClassType.QUOTA)
class ValidFilter(FilterTerminateBase):
    """Valid Filter class."""

    def __init__(self):
        super(ValidFilter, self).__init__()
        self.dataloader = None
        if zeus.is_ms_backend():
            return

    def is_filtered(self, desc=None):
        """Filter function of latency."""
        if zeus.is_ms_backend():
            return False
        try:
            if not self.dataloader:
                dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
                self.dataset = dataset_cls()
                from zeus.datasets import Adapter
                self.dataloader = Adapter(self.dataset).loader
            model, count_input = self.get_model_input(desc)
            model(count_input)
            return False
        except Exception:
            encoding = desc['backbone']['encoding']
            logging.info('Invalid encoding: {}'.format(encoding))
            return True
