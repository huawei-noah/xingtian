import os
import sys
import time

project_basedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_basedir)

import tensorflow as tf

from agent import resnet
from env.cchess_env import create_uci_labels
from config import conf


network = resnet.get_model(
    None, create_uci_labels(),
    gpu_core=[0],
    filters=conf.TrainingConfig.network_filters,
    num_res_layers=conf.TrainingConfig.network_layers
)
(sess, graph), _ = network

stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
network_dir = conf.ResourceConfig.model_dir
dst = os.path.join(network_dir, "{}".format(stamp))

with graph.as_default():
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.save(sess, dst)
