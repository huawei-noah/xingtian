# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Convert class to desc."""
from copy import deepcopy
import hashlib
import json
from inspect import signature as sig
from collections import OrderedDict
from zeus.common.config import Config
from zeus.common import ClassFactory, ClassType, SearchSpaceType


class Serializable(object):
    """Base class of Serializable."""

    def __new__(cls, *args, **kwargs):
        """Record params."""
        desc = {}
        params_sig = sig(cls.__init__).parameters
        param_names = list(params_sig.keys())
        if len(param_names) > len(args):
            # not dynamic parameter for connections
            for idx, arg in enumerate(args):
                arg_name = param_names[idx + 1]
                desc[arg_name] = arg
        if kwargs:
            desc.update(kwargs)
        instance = super(Serializable, cls).__new__(cls)
        instance._deep_level = 0
        instance._target_level = None
        instance.desc = Config(desc)
        return instance

    def to_desc(self, level=None):
        """Convert Module to desc dict."""
        raise NotImplementedError

    @classmethod
    def from_desc(cls, desc):
        """Create Model from desc."""
        raise NotImplementedError

    @property
    def md5(self):
        """MD5 value of network description."""
        return self.get_md5(self.to_desc(1))

    @classmethod
    def get_md5(cls, desc):
        """Get desc's short md5 code.

        :param desc: network description.
        :type desc: str.
        :return: short MD5 code.
        :rtype: str.

        """
        code = hashlib.md5(json.dumps(desc, sort_keys=True).encode('utf-8')).hexdigest()
        return code[:8]

    @property
    def model_name(self):
        """Return model name."""
        return self.__class__.__name__


class OperatorSerializable(Serializable):
    """Seriablizable for Operator class."""

    def to_desc(self, level=None):
        """Convert Operator to desc dict.

        :param level: Specifies witch level to convert. all conversions are performed as default.
        """
        desc = dict(type=self.model_name)
        desc.update(self.desc)
        return desc

    @classmethod
    def from_desc(cls, desc):
        """Create Operator class by desc."""
        return ClassFactory.get_instance(ClassType.SEARCH_SPACE, desc)


class ModuleSerializable(Serializable):
    """Seriablizable Module class."""

    def to_desc(self, level=None):
        """Convert Module to desc dict.

        :param level: Specifies witch level to convert. all conversions are performed as default.
        """
        if level is not None:
            self._target_level = level
        if self._target_level and self._target_level == self._deep_level:
            desc = {'type': self.__class__.__name__}
            desc.update(self.desc)
            return desc
        desc = {'modules': [], "type": self.__class__.__name__}
        if self._losses:
            desc['loss'] = self._losses
        child_level = self._deep_level + 1
        for name, module in self.named_children():
            module._deep_level = child_level
            module._target_level = self._target_level
            sub_desc = module.to_desc()
            desc['modules'].append(name)
            desc[name] = sub_desc
        return desc

    def update_from_desc(self, desc):
        """Update desc according to desc."""
        for key, value in desc.items():
            if key == 'type' or not hasattr(self, key):
                continue
            child_module = getattr(self, key)
            if hasattr(child_module, 'add_module'):
                self.add_module(key, value)
            else:
                child_module.update_from_desc(value)

    @classmethod
    def from_desc(cls, desc):
        """Create Model from desc."""
        module_groups = desc.get('modules')
        module_type = desc.get('type', 'Sequential')
        loss = desc.get('loss')
        modules = OrderedDict()
        for group_name in module_groups:
            module_desc = deepcopy(desc.get(group_name))
            if 'modules' in module_desc:
                module = cls.from_desc(module_desc)
            else:
                cls_name = module_desc.get('type')
                if not ClassFactory.is_exists(ClassType.SEARCH_SPACE, cls_name):
                    return None
                module = ClassFactory.get_instance(ClassType.SEARCH_SPACE, module_desc)
            modules[group_name] = module
        if ClassFactory.is_exists(SearchSpaceType.CONNECTIONS, module_type):
            connections = ClassFactory.get_cls(SearchSpaceType.CONNECTIONS, module_type)
        else:
            connections = ClassFactory.get_cls(SearchSpaceType.CONNECTIONS, 'Sequential')
        model = list(modules.values())[0] if len(modules) == 1 else connections(modules)
        if loss:
            model.add_loss(ClassFactory.get_cls(ClassType.LOSS, loss))
        return model
