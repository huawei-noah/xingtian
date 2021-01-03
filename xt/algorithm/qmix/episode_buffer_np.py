"""
Make episode buffer for qmix algorithm.

# DISCLAMER:
codes are mainly referenced and copied from:
https://github.com/oxwhirl/pymarl/blob/master/src/components/episode_buffer.py
"""
from types import SimpleNamespace as SN

import numpy as np

from absl import logging


class EpisodeBatchNP:
    """Implemente episode batch using numpy."""

    def __init__(self, scheme, groups, batch_size, max_seq_length,
                 data=None, preprocess=None):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        # self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size,
                             max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme  # actions configure in scheme
                new_k = preprocess[k][0]  # "actions_onehot", why need new_k?
                transforms = preprocess[k][1]  # user could define much transform

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:  # used the last shape ?
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {"vshape": vshape, "dtype": dtype}
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {
                "vshape": (1, ),
                "dtype": np.long
            },
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, \
                "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)  # how to ?
            dtype = field_info.get("dtype", np.float32)

            if isinstance(vshape, int):
                vshape = (vshape, )

            if group:
                assert (group in groups), \
                    "Group {} must have its number of members" \
                    " defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = np.zeros(
                    (batch_size, *shape), dtype=dtype)
            else:
                logging.debug("field_key-{}, dtype: {}".format(field_key, dtype))
                self.data.transition_data[field_key] = np.zeros(
                    (batch_size, max_seq_length, *shape), dtype=dtype)

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, val in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", np.float32)
            val = np.array(val, dtype=dtype)
            # print("_slices value: ", _slices)
            self._check_safe_view(val, target[k][_slices])
            # target[k][_slices] = v.view_as(target[k][_slices])
            _target_shape = target[k][_slices].shape
            target[k][_slices] = val.reshape(*_target_shape)

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                val = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    val = transform.transform(val)
                # target[new_k][_slices] = v.view_as(target[new_k][_slices])
                target[new_k][_slices] = val

    @staticmethod
    def _check_safe_view(v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(
                        v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {
                self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                for key in item if "group" in self.scheme[key]
            }
            ret = EpisodeBatchNP(
                new_scheme,
                new_groups,
                self.batch_size,
                self.max_seq_length,
                data=new_data,
            )
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatchNP(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data)
            return ret

    @staticmethod
    def _get_num_items(indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    @staticmethod
    def _new_data_sn():
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    @staticmethod
    def _parse_slices(items):
        parsed = []
        # Only batch slice given, add full time slice
        # slice a:b , int i , # [a,b,c]
        if isinstance(items, slice) \
           or isinstance(items, int) \
           or (isinstance(items, (list, np.ndarray))):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            # NOTEs: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item + 1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return tuple(parsed)

    def max_t_filled(self):
        return np.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{}" " Max_seq_len:{} Keys:{} Groups:{}".format(
            self.batch_size, self.max_seq_length, self.scheme.keys(), self.groups.keys())


class ReplayBufferNP(EpisodeBatchNP):
    def __init__(self, scheme, groups, buffer_size, max_seq_length,
                 preprocess=None):
        super(ReplayBufferNP, self).__init__(
            scheme, groups, buffer_size, max_seq_length, preprocess=preprocess)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(
                ep_batch.data.transition_data,
                slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                slice(0, ep_batch.max_seq_length),
                mark_filled=False,
            )
            self.update(
                ep_batch.data.episode_data,
                slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
            )
            self.buffer_index = self.buffer_index + ep_batch.batch_size
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size,
                                      replace=False)
            return self[ep_ids]

    def sample_with_id(self, ids):
        if len(ids) == self.episodes_in_buffer:
            return self[:self.episodes_in_buffer]
        else:
            return self[ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(
            self.episodes_in_buffer,
            self.buffer_size,
            self.scheme.keys(),
            self.groups.keys(),
        )


def check_equal(rb_torch, rb_np):
    # test transition
    transition_torch = rb_torch.data.transition_data
    transition_np = rb_np.data.transition_data
    for k, v in transition_torch.items():
        assert k in transition_np.keys(), "{} not in np".format(k)
        assert (np.array(transition_np[k]) == v.numpy()).all(), "{} vs {}".format(transition_np[k], v)
        # print("pass {}".format(k))

    # self.data.episode_data = {}
