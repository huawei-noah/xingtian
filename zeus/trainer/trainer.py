# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Trainer."""
import glob
import logging
import zeus
from zeus.common import FileOps, init_log
from zeus.common import ClassFactory, ClassType
from zeus.common.config import Config
from zeus.trainer.callbacks import CallbackList
from zeus.trainer.conf import TrainerConfig
from zeus.trainer.distributed_worker import DistributedWorker
from zeus.trainer.modules.losses import Loss
from zeus.trainer.modules.lr_schedulers import LrScheduler
from zeus.trainer.modules.optimizer import Optimizer
from zeus.trainer.utils import WorkerTypes
from zeus.trainer.tf_utils import TFVariables
from zeus.datasets import Adapter
from zeus.common.general import General

if zeus.is_torch_backend():
    import torch
    from zeus.metrics.pytorch.metrics import Metrics

    try:
        import horovod.torch as hvd
    except Exception:
        # logging.warning("horovod not been installed, {}".format(str(e)))
        pass
    try:
        import apex
        from apex import amp
    except Exception:
        # logging.warning("apex not been installed, {}".format(str(e)))
        pass
elif zeus.is_tf_backend():
    import tensorflow as tf
    from zeus.metrics.tensorflow.metrics import Metrics

    try:
        import horovod.tensorflow as hvd
    except Exception:
        # logging.warning("horovod not been installed, {}".format(str(e)))
        pass
elif zeus.is_ms_backend():
    from mindspore import context
    from mindspore.train import Model as MsModel
    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
    from .callbacks.ms_callbacks import EvalCallBack
    from zeus.metrics.mindspore.metrics import Metrics

if zeus.is_npu_device() and zeus.is_tf_backend():
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator import npu_ops
    from hccl.manage.api import get_local_rank_id
    from hccl.manage.api import get_rank_size
    from hccl.manage.api import get_rank_id
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.TRAINER)
class Trainer(DistributedWorker):
    """Trainer class.

    :param model: input model, defaults to None
    :type model: tf model, optional
    :param id: id of the model, defaults to None
    :type id: int, optional
    :param hps: hyperparameters, defaults to None
    :type hps: dict, optional
    """

    config = TrainerConfig()

    def __init__(self, model=None, id=None, hps=None,
                 load_ckpt_flag=False, model_desc=None,
                 lazy_build=True, **kwargs):
        super(Trainer, self).__init__()
        self.worker_type = WorkerTypes.TRAINER
        Trainer.__worker_id__ += 1
        if id is not None:
            self._worker_id = id
        else:
            self._worker_id = Trainer.__worker_id__

        # Data Memeber list of Trainer
        self.is_chief = True
        self.use_cuda = self.config.cuda
        self.epochs = self.config.epochs
        self.do_validation = True
        self.auto_save_ckpt = True
        self.auto_save_perf = True
        self.skip_train = False
        self.valid_interval = self.config.valid_interval
        self.hps = hps
        self.model = model
        self.model_desc = model_desc
        self.optimizer = None
        self.lr_scheduler = None
        self.loss = None
        self.use_syncbn = self.config.syncbn
        self.use_amp = self.config.amp
        self.train_metrics = None
        self.valid_metrics = None
        self.call_metrics_on_train = self.config.call_metrics_on_train
        self.train_verbose = self.config.train_verbose
        self.valid_verbose = self.config.valid_verbose
        self.train_report_steps = self.config.train_report_steps
        self.valid_report_steps = self.config.valid_report_steps
        self.train_loader = None
        self.valid_loader = None
        self.train_step = None
        self.valid_step = None
        self.make_batch = None
        self.model_fn = None
        self.train_input_fn = None
        self.valid_input_fn = None
        self.callbacks = None
        self.performance = None
        self.runtime = None
        self.visual_data = {}
        self.load_ckpt_flag = load_ckpt_flag
        self.distributed = self.config.distributed
        # Used by TimmTrainerCallbacks since it builds its trainer in
        # the before_train callback
        self.lazy_built = self.config.lazy_built
        # Indicate whether the necessary components of a trainer
        # has been built for running
        self._world_size = 1
        self._rank_id = 0
        self._local_rank_id = 0
        self.config.kwargs = kwargs
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        worker_path = self.get_local_worker_path()
        self.model_path = FileOps.join_path(worker_path, self.model_pickle_file_name)
        self.checkpoint_file = FileOps.join_path(worker_path, self.checkpoint_file_name)
        self.weights_file = FileOps.join_path(worker_path, "model_{}.pth".format(self.worker_id))
        self.loss_input = kwargs.get('loss_input', None)
        if not lazy_build:
            self.init_trainer()

    def _set_default_funcs(self):
        if zeus.is_torch_backend():
            self.make_batch = self._default_make_batch
            self.train_step = self._default_train_step
            self.valid_step = self._default_valid_step
        elif zeus.is_tf_backend():
            self.model_fn = self._default_model_fn
            self.train_input_fn = self._default_train_input_fn
            self.valid_input_fn = self._default_valid_input_fn

    def _set_condition(self):
        self._init_tf_session()
        self._init_distributed_setting()
        self._init_cuda_setting()
        self._init_tf_estimator()
        self._init_ms_context()

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        init_log(level=General.logger.level,
                 log_file="log_worker_{}.txt".format(self.worker_id),
                 log_path=self.local_log_path)
        self._set_default_funcs()
        self._set_condition()
        self._init_callbacks()
        self.callbacks.init_trainer()
        if not self.lazy_built:
            self.build()
        self._train_loop()

    def build(self):
        """Build the trainer by assembling the necessary components."""
        self._init_hps(self.hps)
        logging.debug("Trainer Config: {}".format(self.config))
        self.do_validation = self.config.with_valid
        self.use_syncbn = self.config.syncbn
        if self.use_syncbn and zeus.is_torch_backend():
            self.model = apex.parallel.convert_syncbn_model(self.model)
        self.train_loader = self._init_dataloader(mode='train')
        self.valid_loader = self._init_dataloader(mode='val')
        self.batch_num_train = self.train_loader.get_dataset_size() if zeus.is_ms_backend() else len(self.train_loader)
        self.batch_num_valid = self.valid_loader.get_dataset_size() if zeus.is_ms_backend() else len(self.valid_loader)

        if zeus.is_torch_backend():
            self.optimizer = Optimizer()(model=self.model, distributed=self.distributed)
            if hasattr(self.model, 'add_loss'):
                loss_cls = Loss()()
                self.model.add_loss(loss_cls)
                self.loss = self.model.overall_loss()
            else:
                self.loss = Loss()()
            self.lr_scheduler = LrScheduler()(self.optimizer)
        elif zeus.is_ms_backend():
            self.optimizer = Optimizer()(model=self.model)
            if hasattr(self.model, 'add_loss'):
                loss_cls = Loss()()
                self.model.add_loss(loss_cls)
                self.loss = self.model.overall_loss()
            else:
                self.loss = Loss()()
            self.metric_name = self.config.metric().type
        # Some trainer has different train batch size from valid batch
        self.train_metrics = self._init_metrics() if zeus.is_torch_backend() else None
        self.valid_metrics = self._init_metrics()
        self._init_horovod_setting()
        if self.use_amp and zeus.is_torch_backend():
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')

    def init_trainer(self):
        """Init Train Op."""
        init_log(level=General.logger.level,
                 log_file="log_worker_{}.txt".format(self.worker_id),
                 log_path=self.local_log_path)
        self._set_default_funcs()
        self._set_condition()
        self._init_callbacks()
        self.callbacks.init_trainer()

        self.init_train_op()

    def init_train_op(self):
        """Init Train Op."""
        if zeus.is_tf_backend():
            with self.graph.as_default():
                self._init_train_op()

    def train(self, inputs, labels):
        """Train model."""
        if zeus.is_tf_backend():
            feed_dict = {}
            with self.graph.as_default():
                for i in range(len(inputs)):
                    feed_dict.update({self.inputs[i]: inputs[i]})

                for i in range(len(labels)):
                    feed_dict.update({self.labels[i]: labels[i]})

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
                return loss

    def predict(self, input):
        """Inference model."""
        if zeus.is_tf_backend():
            with self.graph.as_default():
                feed_dict = {self.input: input}
                out = self.sess.run(self.logits, feed_dict)
                return out

    def save(self, file_name):
        """Save model."""
        if zeus.is_tf_backend():
            with self.graph.as_default():
                self.actor_var.save_weights(file_name + ".npz")

            return file_name + ".npz"

    def load(self, model_name, by_name):
        """Load model."""
        if zeus.is_tf_backend():
            with self.graph.as_default():
                self.actor_var.set_weights_with_npz(model_name)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        if zeus.is_tf_backend():
            with self.graph.as_default():
                self.actor_var.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        if zeus.is_tf_backend():
            with self.graph.as_default():
                return self.actor_var.get_weights()

    def _create_tensor(self, tensor_list):
        ret_list = []

        for tensor in tensor_list:
            tensor_type = tensor['type']
            tensor_shape = tensor['shape']
            tensor_name = tensor['name']

            if type(tensor_shape) is list:
                tf_tensor = tf.placeholder(tensor_type, name=tensor_name,
                                           shape=(None, ) + tuple(tensor_shape))
            else:
                tf_tensor = tf.placeholder(tensor_type, name=tensor_name,
                                           shape=(None, tensor_shape))
            ret_list.append(tf_tensor)

        return ret_list

    def _init_train_op(self):
        if self.loss_input is not None:
            self.inputs = self._create_tensor(self.loss_input['inputs'])
            self.labels = self._create_tensor(self.loss_input['labels'])

            self.input = self.inputs[0]
            logits = self.model(self.input)
            self.logits = logits
            self.actor_var = TFVariables(logits, self.sess)

            loss = Loss()()
            self.loss = loss(logits, self.labels)

            self.optimizer = Optimizer()(distributed=self.distributed)
            grads_and_var = self.optimizer.compute_gradients(self.loss)
            grads, var = zip(*grads_and_var)
            grads_and_var = list(zip(grads, var))
            self.train_op = self.optimizer.apply_gradients(grads_and_var)
            self.sess.run(tf.initialize_all_variables())

    def _init_cuda_setting(self):
        """Init CUDA setting."""
        if not zeus.is_torch_backend():
            return
        if not self.config.cuda:
            self.config.device = -1
            return
        self.config.device = self.config.cuda if self.config.cuda is not True else 0
        self.use_cuda = True
        if self.distributed:
            torch.cuda.set_device(self._local_rank_id)
        torch.cuda.manual_seed(self.config.seed)

    def _init_distributed_setting(self):
        if not self.distributed:
            return
        if zeus.is_npu_device():
            self.npu_init = npu_ops.initialize_system()
            self.npu_shutdown = npu_ops.shutdown_system()
            self.sess.run(self.npu_init)
        self._world_size = hvd.size() if zeus.is_gpu_device() else get_rank_size()
        self._rank_id = hvd.rank() if zeus.is_gpu_device() else get_rank_id()
        self._local_rank_id = hvd.local_rank() if zeus.is_gpu_device() else get_local_rank_id()

    def _init_horovod_setting(self):
        """Init horovod setting."""
        self.is_chief = True
        if self.distributed and zeus.is_torch_backend():
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            if hvd.rank() != 0:
                self.is_chief = False
            else:
                self.is_chief = True

    def _init_hps(self, hps=None):
        """Load hps from file."""
        if hps is not None:
            self.hps = hps
        elif self.config.hps_file is not None:
            desc_file = self.config.hps_file.replace("{local_base_path}", self.local_base_path)
            self.hps = Config(desc_file)
        elif self.config.hps_folder is not None:
            folder = self.config.hps_folder.replace("{local_base_path}", self.local_base_path)
            pattern = FileOps.join_path(folder, "desc_*.json")
            desc_file = glob.glob(pattern)[0]
            self.hps = Config(desc_file)
        if self.hps and self.hps.get('trainer'):
            self.config.from_json(self.hps.get('trainer'))
            self.epochs = self.config.epochs

    def _init_metrics(self, metrics=None):
        """Init metrics."""
        if metrics is not None:
            return metrics
        else:
            return Metrics()

    def _init_dataloader(self, mode, loader=None):
        """Init dataloader."""
        if loader is not None:
            return loader
        if mode == "train" and self.hps is not None and self.hps.get("dataset") is not None:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode, hps=self.hps.get("dataset"))
        else:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode)
        if self.distributed and mode == "train":
            dataset.set_distributed(self._world_size, self._rank_id)
        # adapt the dataset to specific backend
        dataloader = Adapter(dataset).loader
        return dataloader

    def _train_loop(self):
        """Do the training with data, callbacks and step functions etc."""
        # Allow user to build trainer in before_train() callback, but they
        # should set lazy_built in configuration file to True
        self.callbacks.before_train()
        if self.skip_train:
            return
        repeat_time = 1 if zeus.is_ms_backend() else self.epochs
        for epoch in range(repeat_time):
            epoch_logs = {'train_num_batches': self.batch_num_train}
            if self.do_validation:
                epoch_logs.update({'valid_num_batches': self.batch_num_valid})
            self.callbacks.before_epoch(epoch, epoch_logs)
            self._train_epoch()
            if self.do_validation and self._should_run_validation(epoch):
                self._valid_epoch()
            self.callbacks.after_epoch(epoch)
        self.callbacks.after_train()
        if self.distributed:
            self._shutdown_distributed()

    def _train_epoch(self):
        if zeus.is_torch_backend():
            self.model.train()
            for batch_index, batch in enumerate(self.train_loader):
                batch = self.make_batch(batch)
                batch_logs = {'train_batch': batch}
                self.callbacks.before_train_step(batch_index, batch_logs)
                train_batch_output = self.train_step(batch)
                batch_logs.update(train_batch_output)
                if self.config.is_detection_trainer:
                    batch_logs.update({'is_detection_trainer': True})
                self.callbacks.after_train_step(batch_index, batch_logs)
        elif zeus.is_tf_backend():
            self.estimator.train(input_fn=self.train_input_fn,
                                 steps=len(self.train_loader),
                                 hooks=self._init_logging_hook())
        elif zeus.is_ms_backend():
            self.ms_model = MsModel(network=self.model,
                                    loss_fn=self.loss,
                                    optimizer=self.optimizer,
                                    metrics={self.metric_name: self.valid_metrics()})
            config_ck = CheckpointConfig(save_checkpoint_steps=self.config.save_steps)
            # save the network model and parameters for subsequence fine-tuning
            save_path = self.get_local_worker_path(self.step_name, self.worker_id)
            ckpoint_cb = ModelCheckpoint(config=config_ck, directory=save_path)
            loss_cb = LossMonitor(per_print_times=self.config.report_freq)
            eval_cb = EvalCallBack(self.ms_model, self.valid_loader)
            self.ms_model.train(epoch=self.epochs,
                                train_dataset=self.train_loader,
                                callbacks=[ckpoint_cb, loss_cb, eval_cb],
                                dataset_sink_mode=self.dataset_sink_mode)

    def _valid_epoch(self):
        self.callbacks.before_valid()
        valid_logs = None
        if zeus.is_torch_backend():
            self.model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(self.valid_loader):
                    batch = self.make_batch(batch)
                    batch_logs = {'valid_batch': batch}
                    self.callbacks.before_valid_step(batch_index, batch_logs)
                    valid_batch_output = self.valid_step(batch)
                    self.callbacks.after_valid_step(batch_index, valid_batch_output)
        elif zeus.is_tf_backend():
            eval_metrics = self.estimator.evaluate(input_fn=self.valid_input_fn,
                                                   steps=len(self.valid_loader))
            self.valid_metrics.update(eval_metrics)
            valid_logs = dict()
            valid_logs['cur_valid_perfs'] = self.valid_metrics.results
        elif zeus.is_ms_backend():
            eval_metrics = self.ms_model.eval(valid_dataset=self.valid_loader,
                                              dataset_sink_mode=self.dataset_sink_mode)

            self.valid_metrics.update(eval_metrics)
            valid_logs = dict()
            valid_logs['cur_valid_perfs'] = self.valid_metrics.results
        self.callbacks.after_valid(valid_logs)

    def _default_make_batch(self, batch):
        """Unpack batch to get input and target."""
        input, target = batch
        if self.use_cuda and not self.config.is_detection_trainer:
            input, target = input.cuda(), target.cuda()
        return (input, target)

    def _default_train_step(self, batch):
        input, target = batch
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, target)
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.optimizer.synchronize()
            with self.optimizer.skip_synchronize():
                self.optimizer.step()
        else:
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': output,
                'lr': self.lr_scheduler.get_lr()}

    def _default_valid_step(self, batch):
        input, target = batch
        if self.config.is_detection_trainer:
            output = self.model(input, forward_train=False)
        else:
            output = self.model(input)
        return {'valid_batch_output': output}

    def _init_minimize_op(self, loss, global_step, var_list=None):
        """Init loss minimize operation, include loss scale method."""
        loss_scale = self.config.loss_scale if self.use_amp else 1.
        if loss_scale != 1:
            scaled_grad_vars = self.optimizer.compute_gradients(loss * loss_scale, var_list=var_list)
            unscaled_grad_vars = []
            for grad, var in scaled_grad_vars:
                unscaled_grad_vars.append((grad, var) if grad is None else (grad / loss_scale, var))
            minimize_op = self.optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            grad_vars = self.optimizer.compute_gradients(loss, var_list=var_list)
            minimize_op = self.optimizer.apply_gradients(grad_vars, global_step)
        return minimize_op

    def _default_train_input_fn(self):
        return self.train_loader.input_fn()

    def _default_valid_input_fn(self):
        return self.valid_loader.input_fn()

    def _default_model_fn(self, features, labels, mode):
        """Define model_fn used by TensorFlow Estimator.

        :params features: input features
        :type features: tensorflow tensors
        :params labels: label data
        :type labels: tensorflow tensors
        :params mode: mode of estimator
        :type mode: tf.estimator.ModeKeys
        :return: tensorflow EstimatorSpec
        :rtype: tf.estimator.EstimatorSpec
        """
        logging.info('model function action')
        self.model.training = mode == tf.estimator.ModeKeys.TRAIN
        logits = self.model(features)
        logits = tf.cast(logits, tf.float32)
        if hasattr(self.model, 'add_loss'):
            loss_cls = Loss()()
            self.model.add_loss(loss_cls)
            self.loss = self.model.overall_loss()
        else:
            self.loss = Loss()()
        loss = self.loss(logits, labels)
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.compat.v1.train.get_or_create_global_step()
            epoch = tf.cast(global_step, tf.float32) / tf.cast(len(self.train_loader), tf.float32)
            self.optimizer = Optimizer()(distributed=self.distributed)
            self.lr_scheduler = LrScheduler()(optimizer=self.optimizer)
            self.lr_scheduler.step(epoch)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            loss_scale = self.config.loss_scale if self.use_amp else 1
            minimize_op = self.optimizer.step(loss, loss_scale, global_step)
            train_op = tf.group(minimize_op, update_ops)

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = self.valid_metrics(logits, labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops)

    def _should_run_validation(self, epoch):
        # Zero valid_interval means doesn't run _valid_loop of the trainer
        # and user may provide _valid_loop in other callbacks
        if self.valid_interval == 0:
            return False
        else:
            return epoch % self.valid_interval == 0 or (epoch + 1) == self.epochs

    def _init_callbacks(self):
        disables = []
        customs = self.config.callbacks or []
        if customs and not isinstance(customs, list):
            customs = [customs]
        if not self.config.model_statistics:
            disables.append('ModelStatistics')
        self.callbacks = CallbackList(customs, disables)
        self.callbacks.set_trainer(self)

    def _metric_average(self, val, name):
        """Do metric average.

        :param val: input value
        :param name: metric name
        :return:
        """
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    @property
    def _first_rank(self):
        """Check if the first rank."""
        if self.distributed and hvd.rank() != 0:
            return False
        else:
            return True

    def _backup(self):
        """Backup result worker folder."""
        if self.need_backup is True and self.backup_base_path is not None:
            backup_worker_path = FileOps.join_path(
                self.backup_base_path, self.get_worker_subpath())
            FileOps.copy_folder(
                self.get_local_worker_path(self.step_name, self.worker_id), backup_worker_path)

    def _save_visual_data(self, is_train=True, pfms=None, loss=None, lr=None):
        # TODO Will move to metric base class later.
        for _name, value in pfms.items():
            if is_train:
                _name = "{}_{}".format("t", _name)
            else:
                _name = "{}_{}".format("v", _name)
            if isinstance(value, list):
                for i, _item in enumerate(value):
                    _name = "{}_{}".format(_name, i)
                    self.visual_data[_name] = _item.data.item()
            elif isinstance(value, dict):
                for k, v in value.keys():
                    _name = "{}_{}".format(_name, k)
                    self.visual_data[_name] = v
            elif value is not None:
                self.visual_data[_name] = value.data.item()
        if loss is not None:
            self.visual_data["loss"] = loss
        if lr is not None:
            self.visual_data["lr"] = lr

    def _init_tf_estimator(self):
        """Init tensorflow estimator."""
        if not zeus.is_tf_backend():
            return
        sess_config = self._init_session_config()
        if zeus.is_gpu_device():
            self._init_gpu_estimator(sess_config)
        elif zeus.is_npu_device():
            self._init_npu_estimator(sess_config)

    def _init_tf_session(self):
        if not zeus.is_tf_backend():
            return
        sess_config = self._init_session_config()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session(config=sess_config)

    def _init_session_config(self):
        sess_config = self._init_gpu_session_config() if zeus.is_gpu_device() else \
            self._init_npu_session_config()
        return sess_config

    def _init_logging_hook(self):
        logging_hook = []
        if zeus.is_gpu_device() and self.distributed:
            logging_hook += [hvd.BroadcastGlobalVariablesHook(0)]
        return logging_hook

    def _init_gpu_estimator(self, sess_config):
        """Init tensorflow estimator."""
        distribution = None
        if not self.distributed and General._parallel and General.devices_per_trainer > 1:
            distribution = tf.contrib.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(model_dir=self.get_local_worker_path(),
                                        save_checkpoints_steps=self.config.save_steps,
                                        log_step_count_steps=self.config.report_freq,
                                        session_config=None if distribution else sess_config,
                                        train_distribute=distribution)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                config=config)

    def _init_npu_estimator(self, sess_config):
        model_dir = self.get_local_worker_path()
        config = NPURunConfig(model_dir=model_dir,
                              save_checkpoints_steps=self.config.save_steps,
                              log_step_count_steps=self.config.report_freq,
                              session_config=sess_config,
                              enable_data_pre_proc=True,
                              iterations_per_loop=1)
        self.estimator = NPUEstimator(model_fn=self.model_fn,
                                      config=config)

    def _init_gpu_session_config(self):
        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        if self.distributed:
            sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
        return sess_config

    def _init_npu_session_config(self):
        sess_config = tf.ConfigProto()
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        if self.use_amp:
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["use_off_line"].b = True
        # custom_op.parameter_map['hcom_parallel'].b = True
        # custom_op.parameter_map["enable_data_pre_proc"].b = True
        # custom_op.parameter_map["mix_compile_mode"].b = True  # mixed calculation
        # custom_op.parameter_map["min_group_size"].b = 1
        return sess_config

    def _init_ms_context(self):
        if not zeus.is_ms_backend():
            return
        if zeus.is_npu_device():
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        self.dataset_sink_mode = True if zeus.is_npu_device() else False

    def _shutdown_distributed(self):
        if zeus.is_npu_device() and self.distributed:
            self.sess.run(self.npu_shutdown)
            self.sess.close()
