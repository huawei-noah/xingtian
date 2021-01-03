# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""HostEvaluator used to do evaluate process on gpu."""
import logging
import numpy as np
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.common.general import General
from zeus.common.utils import init_log
from .tools.evaluate_davinci_bolt import evaluate
from .conf import DeviceEvaluatorConfig
from zeus.report import ReportClient
from .evaluator import Evaluator
from zeus.trainer.utils import WorkerTypes
import os
import datetime


@ClassFactory.register(ClassType.DEVICE_EVALUATOR)
class DeviceEvaluator(Evaluator):
    """Evaluator is a davinci and mobile evaluator.

    :param args: arguments from user and default config file
    :type args: dict or Config, default to None
    :param train_data: training dataset
    :type train_data: torch dataset, default to None
    :param valid_data: validate dataset
    :type valid_data: torch dataset, default to None
    :param worker_info: the dict worker info of workers that finished train.
    :type worker_info: dict or None.
    """

    config = DeviceEvaluatorConfig()

    def __init__(self, worker_info=None, model=None, saved_folder=None, saved_step_name=None,
                 model_desc=None, weights_file=None, **kwargs):
        """Init DeviceEvaluator."""
        super(Evaluator, self).__init__()
        # self.backend = self.config.backend
        self.hardware = self.config.hardware
        self.remote_host = self.config.remote_host
        self.calculate_metric = self.config.calculate_metric
        self.model = model
        self.worker_info = worker_info
        self.worker_type = WorkerTypes.DeviceEvaluator
        if worker_info is not None and "step_name" in worker_info and "worker_id" in worker_info:
            self.step_name = self.worker_info["step_name"]
            self.worker_id = self.worker_info["worker_id"]
        self.model_desc = model_desc

        self.weights_file = weights_file
        self.saved_folder = saved_folder
        self.saved_step_name = saved_step_name

    def valid(self):  # noqa: C901
        """Validate the latency in davinci or bolt."""
        test_data = os.path.join(self.get_local_worker_path(self.step_name, self.worker_id), "input.bin")
        latency_sum = 0
        data_num = 0
        global_step = 0
        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        job_id = self.step_name + "_" + str(self.worker_id) + "_" + now_time
        logging.info("The job id of evaluate service is {}.".format(job_id))
        if zeus.is_torch_backend():
            import torch
            from zeus.metrics.pytorch import Metrics
            metrics = Metrics(self.config.metric)
            for step, batch in enumerate(self.valid_loader):
                if isinstance(batch, list) or isinstance(batch, tuple):
                    data = batch[0]
                    target = batch[1]
                else:
                    raise ValueError("The dataset format must be tuple or list,"
                                     "but get {}.".format(type(batch)))
                input_shape = data.shape
                data_num += data.size(0)
                for i in range(data.size(0)):
                    if not self.calculate_metric and global_step >= 10:
                        break
                    one_data = data[i:i + 1]
                    one_target = target[i:i + 1]
                    if torch.is_tensor(one_data):
                        one_data = one_data.numpy()
                    one_data.tofile(test_data)
                    reuse_model = False if global_step == 0 else True
                    results = evaluate("pytorch", self.hardware, self.remote_host,
                                       self.model, None, test_data, one_data.shape, reuse_model, job_id)
                    latency = np.float(results.get("latency"))
                    latency_sum += latency

                    if global_step == 0:
                        real_output = self.model(torch.Tensor(one_data))
                        if isinstance(real_output, tuple):
                            output_shape = real_output[0].shape
                        else:
                            output_shape = real_output.shape
                    if self.calculate_metric:
                        out_data = np.array(results.get("out_data")).astype(np.float32)

                        output = out_data.reshape(output_shape)
                        output = torch.Tensor(output)
                        metrics(output, one_target)
                        pfms = metrics.results
                    else:
                        pfms = {}

                    global_step += 1
                    if global_step % self.config.report_freq == 0:
                        logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                            step + 1, len(self.valid_loader), latency, pfms))

        elif zeus.is_tf_backend():
            import tensorflow as tf
            from zeus.metrics.tensorflow.metrics import Metrics
            valid_data = self.valid_loader.input_fn()
            metrics = Metrics(self.config.metric)
            iterator = valid_data.make_one_shot_iterator()
            one_element = iterator.get_next()
            total_metric = {}
            weight_file = self.get_local_worker_path(self.step_name, self.worker_id)
            for step in range(len(self.valid_loader)):
                with tf.Session() as sess:
                    batch = sess.run(one_element)
                data = batch[0]
                target = batch[1]
                input_shape = data.shape
                data_num += input_shape[0]
                for i in range(input_shape[0]):
                    if not self.calculate_metric and global_step >= 10:
                        break
                    one_data = data[i:i + 1]
                    one_target = target[i:i + 1]
                    one_data.tofile(test_data)

                    if global_step == 0:
                        input_tf = tf.placeholder(tf.float32, shape=one_data.shape, name='input_tf')
                        self.model.training = False
                        output = self.model(input_tf)
                        if isinstance(output, tuple):
                            output_shape = output[0].shape
                        else:
                            output_shape = output.shape

                    reuse_model = False if global_step == 0 else True
                    results = evaluate("tensorflow", self.hardware, self.remote_host,
                                       self.model, weight_file, test_data, one_data.shape, reuse_model, job_id)
                    latency = np.float(results.get("latency"))
                    latency_sum += latency

                    if self.calculate_metric:
                        out_data = np.array(results.get("out_data")).astype(np.float32)
                        output = out_data.reshape(output_shape)

                        eval_metrics_op = metrics(tf.convert_to_tensor(output), tf.convert_to_tensor(one_target))
                        for name, value in eval_metrics_op.items():
                            eval_metrics_op[name] = value[1]
                        with tf.Session() as sess:
                            # sess.run(tf.global_variables_initializer())
                            sess.run(tf.local_variables_initializer())
                            eval_metrics = sess.run(eval_metrics_op)
                            for name, value in eval_metrics.items():
                                if global_step == 0:
                                    total_metric[name] = value
                                else:
                                    total_metric[name] += value
                                eval_metrics[name] = total_metric[name] / (global_step + 1)
                            logging.info("The eval_metrics of davinvi_mobile_evaluator is {}.".format(eval_metrics))
                            metrics.update(eval_metrics)
                            pfms = metrics.results
                    else:
                        pfms = {}

                    global_step += 1

                    if global_step % self.config.report_freq == 0:
                        logging.info("step [{}/{}], latency [{}], valid metric [{}]".format(
                            step + 1, len(self.valid_loader), latency, pfms))

        latency_avg = latency_sum / global_step
        logging.info("The latency in {} is {} ms.".format(self.hardware, latency_avg))

        if self.config.evaluate_latency:
            pfms["latency"] = latency_avg
        logging.info("valid performance: {}".format(pfms))
        return pfms

    def train_process(self):
        """Validate process for the model validate worker."""
        init_log(level=General.logger.level,
                 log_file="device_evaluator_{}.log".format(self.worker_id),
                 log_path=self.local_log_path)
        logging.info("start davinci or mobile evaluate process")
        self.load_model()
        self.valid_loader = self._init_dataloader(mode='test')
        performance = self.valid()
        logging.info("Evaluator result in davinci/bolt: {}".format(performance))
        self._broadcast(performance)
        logging.info("finished davinci or mobile evaluate for id {}".format(self.worker_id))

    def _broadcast(self, pfms):
        """Boadcase pfrm to record."""
        record = ReportClient.get_record(self.step_name, self.worker_id)
        if record.performance:
            record.performance.update(pfms)
        else:
            record.performance = pfms
        ReportClient.broadcast(record)
        logging.info("valid record: {}".format(record))
