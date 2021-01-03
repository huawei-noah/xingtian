# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Prune operators."""
import numpy as np


class PruneConv2DFilter(object):
    """Prune Conv2D."""

    def __init__(self, layer):
        self.layer = layer

    def filter(self, end_mask_code, start_mask_code=None):
        """Apply mask to weight."""
        if start_mask_code:
            self.filter_in_channels(start_mask_code)
        if end_mask_code:
            self.filter_out_channels(end_mask_code)

    def _make_mask(self, mask_code):
        """Make Mask by mask code."""
        if sum(mask_code) == 0:
            mask_code[0] = 1
        mask_code = np.array(mask_code)
        idx = np.squeeze(np.argwhere(mask_code)).tolist()
        idx = [idx] if not isinstance(idx, list) else idx
        return idx

    def filter_out_channels(self, mask_code):
        """Mask out channels."""
        filter_idx = self._make_mask(mask_code)
        self.layer.weight.data = self.layer.weight.data[filter_idx, :, :, :]
        # change the out_channels value
        self.layer.out_channels = sum(mask_code)
        if self.layer.bias is not None:
            self.layer.bias.data = self.layer.bias.data[filter_idx]

    def filter_in_channels(self, mask_code):
        """Mask in channels."""
        filter_idx = self._make_mask(mask_code)
        self.layer.weight.data = self.layer.weight.data[:, filter_idx, :, :]
        # change the in_channels value
        self.layer.in_channels = sum(mask_code)


class PruneBatchNormFilter(object):
    """Prune BatchNorm."""

    def __init__(self, layer):
        self.layer = layer

    def filter(self, mask_code):
        """Apply mask to batchNorm."""
        if sum(mask_code) == 0:
            mask_code[0] = 1
        mask_code = np.asarray(mask_code)
        idx = np.squeeze(np.argwhere(mask_code)).tolist()
        idx = [idx] if not isinstance(idx, list) else idx
        self.layer.weight.data = self.layer.weight.data[idx]
        self.layer.bias.data = self.layer.bias.data[idx]
        self.layer.running_mean = self.layer.running_mean[idx]
        self.layer.running_var = self.layer.running_var[idx]
        # change num_features value
        self.layer.num_features = sum(mask_code)


class PruneLinearFilter(object):
    """Prune Linear."""

    def __init__(self, layer):
        self.layer = layer

    def filter(self, mask_code):
        """Apply mask to linear."""
        if sum(mask_code) == 0:
            mask_code[0] = 1
        mask_code = np.asarray(mask_code)
        idx_in = np.squeeze(np.argwhere(mask_code)).tolist()
        idx_in = [idx_in] if not isinstance(idx_in, list) else idx_in
        self.layer.weight.data = self.layer.weight.data[:, idx_in]
        # fineTune out_feature value
        out_size = self.layer.weight.data.shape[0]
        if self.layer.out_features == out_size:
            return
        idx_out = list(np.random.permutation(out_size)[:self.layer.out_features])
        self.layer.weight.data = self.layer.weight.data[idx_out, :]
        if self.layer.bias is not None:
            self.layer.bias.data = self.layer.bias.data[idx_out]
