# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The EvaluateService of client."""
import os
import requests
import logging
from .pytorch2onnx import pytorch2onnx
import subprocess
import pickle


def evaluate(backend, hardware, remote_host, model, weight, test_data, input_shape=None, reuse_model=False,
             job_id=None):
    """Evaluate interface of the EvaluateService.

    :param backend: the backend can be one of "tensorflow", "caffe" and "pytorch"
    :type backend: str
    :param hardware: the backend can be one of "Davinci", "Bolt"
    :type hardware: str
    :param remote_host: the remote host ip and port of evaluate service
    :type remote_host: str
    :param model: model file, .pb file for tensorflow and .prototxt for caffe, and a model class for Pytorch
    :type model: str or Class
    :param weight: .caffemodel file for caffe
    :type weight: str
    :param test_data: binary file, .data or .bin
    :type test_data: str
    :return: the latency in Davinci or Bolt
    :rtype: float
    """
    if backend not in ["tensorflow", "caffe", "pytorch"]:
        raise ValueError("The backend only support tensorflow, caffe and pytorch.")

    if hardware not in ["Davinci", "Bolt"]:
        raise ValueError("The hardware only support Davinci and Bolt.")
    else:
        if not reuse_model:
            if backend == "pytorch":
                if input_shape is None:
                    raise ValueError("To convert the pytorch model to onnx model, the input shape must be provided.")
                elif hardware == "Bolt":
                    model = pytorch2onnx(model, input_shape)
                else:
                    base_save_dir = os.path.dirname(test_data)
                    model_file = os.path.join(base_save_dir, "torch_model.pkl")
                    shape_file = os.path.join(base_save_dir, "input_shape.pkl")
                    with open(model_file, "wb") as f:
                        pickle.dump(model, f)
                    with open(shape_file, "wb") as f:
                        pickle.dump(input_shape, f)
                    env = os.environ.copy()
                    command_line = ["bash", "../../zeus/evaluator/tools/pytorch2caffe.sh",
                                    model_file, shape_file]
                    try:
                        subprocess.check_output(command_line, env=env)
                    except subprocess.CalledProcessError as exc:
                        logging.error("convert torch model to caffe model failed.\
                                      the return code is: {}.".format(exc.returncode))
                    model = os.path.join(base_save_dir, "torch2caffe.prototxt")
                    weight = os.path.join(base_save_dir, "torch2caffe.caffemodel")
                    backend = "caffe"
            elif backend == "tensorflow":
                pb_model_file = os.path.join(os.path.dirname(test_data), "tf_model.pb")
                if os.path.exists(pb_model_file):
                    os.remove(pb_model_file)

                freeze_graph(model, weight, pb_model_file, input_shape)
                model = pb_model_file

            model_file = open(model, "rb")
            data_file = open(test_data, "rb")
            if backend == "caffe":
                weight_file = open(weight, "rb")
                upload_data = {"model_file": model_file, "weight_file": weight_file, "data_file": data_file}
            else:
                upload_data = {"model_file": model_file, "data_file": data_file}
        else:
            data_file = open(test_data, "rb")
            upload_data = {"data_file": data_file}
        evaluate_config = {"backend": backend, "hardware": hardware, "remote_host": remote_host,
                           "reuse_model": reuse_model, "job_id": job_id}
        evaluate_result = requests.post(remote_host, files=upload_data, data=evaluate_config,
                                        proxies={"http": None}).json()
        # evaluate_result = requests.get(remote_host, proxies={"http": None}).json()
        if evaluate_result.get("status_code") != 200:
            logging.error("Evaluate failed! The return code is {}, the timestmap is {}."
                          .format(evaluate_result.get("status_code"), evaluate_result.get("timestamp")))
        else:
            logging.info("Evaluate sucess! The latency is {}.".format(evaluate_result["latency"]))
    return evaluate_result


def freeze_graph(model, weight_file, output_graph_file, input_shape):
    """Freeze the tensorflow graph.

    :param model: the tensorflow model
    :type model: str
    :param output_graph_file: the file to save the freeze graph model
    :type output_graph_file: str
    """
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    with tf.Graph().as_default() as graph:
        input_holder = tf.placeholder(dtype=tf.float32, shape=input_shape)
        model.training = False
        output = model(input_holder)
        if isinstance(output, tuple):
            output_name = [output[0].name.split(":")[0]]
        else:
            output_name = [output.name.split(":")[0]]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # if weight_file is None, only latency can be evaluated
            if weight_file is not None:
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(weight_file))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_name)

            with tf.gfile.FastGFile(output_graph_file, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
