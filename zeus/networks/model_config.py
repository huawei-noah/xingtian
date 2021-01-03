# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Conf for Pipeline."""
import os
import glob
from zeus.common import ConfigSerializable
from zeus.common import FileOps, Config
from zeus.common.general import TaskConfig


class ModelConfig(ConfigSerializable):
    """Default Model config for Pipeline."""

    type = None
    model_desc = None
    model_desc_file = None
    pretrained_model_file = None
    models_folder = None
    num_classes = None

    @classmethod
    def from_json(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        t_cls = super(ModelConfig, cls).from_json(data, skip_check)
        if data.get("models_folder") and not data.get('model_desc'):
            folder = data.models_folder.replace("{local_base_path}",
                                                os.path.join(TaskConfig.local_base_path, TaskConfig.task_id))
            pattern = FileOps.join_path(folder, "desc_*.json")
            desc_file = glob.glob(pattern)[0]
            t_cls.model_desc = Config(desc_file)
        return t_cls
