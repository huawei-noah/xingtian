# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Create actor launching module.

User could launch the Actor for explore, or Evaluator for evaluation.
"""

from multiprocessing import Process
from subprocess import Popen

from absl import logging
import setproctitle
from xt.framework.broker import Controller, Broker
from xt.framework.default_config import DEFAULT_NODE_CONFIG
from xt.framework.remoter import remote_run
from zeus.common.util.common import get_host_ip
from zeus.common.util.logger import VERBOSITY_MAP


def launch_remote_broker(user, passwd, actor_ip, host_ip, broker_id,
                         push_port, pull_port, remote_env, verbosity):
    """Start remote actor through fabric."""
    cmd = (
        '"import xt; from xt import start_broker; '
        "start_broker({}, {}, {}, '{}', '{}')\"".format(
            broker_id, push_port, pull_port, host_ip, verbosity
        )
    )
    cmd = " ".join(["python3", "-c", cmd])
    logging.info("start remote broker with: \n{}".format(cmd))

    # need call it with background
    remote_process = Process(
        target=remote_run, args=(actor_ip, user, passwd, cmd, remote_env)
    )
    remote_process.start()


def launch_local_broker(broker_id, push_port, pull_port,
                        server_ip="127.0.0.1", verbosity="info"):
    """
    Run actor in local node.

    The process called by this command could been still alive.
    i.e., run as a foreground task.
    we use `subprocess.Popen.run` currently.
    """
    cmd = (
        "import xt; from xt import start_broker; "
        "start_broker({}, {}, {}, '{}', '{}')".format(
            broker_id, push_port, pull_port, server_ip, verbosity
        )
    )
    logging.info("start launching Broker with: {}".format(cmd))
    Popen(["python3", "-c", cmd])


def start_broker_elf(broker_id, push_port, pull_port,
                     server_ip="127.0.0.1", verbosity="info"):
    """Create a broker and start it."""
    logging.set_verbosity(VERBOSITY_MAP.get(verbosity, logging.INFO))
    logging.info("set verbosity in broker: {}".format(verbosity))

    broker_obj = Broker(server_ip, broker_id, push_port, pull_port)

    broker_obj.start()


def launch_broker(config_info, verbosity="info"):
    """Run actor in local node, unify the act launcher api."""
    node_config_list = config_info.get("node_config", DEFAULT_NODE_CONFIG)

    broker_controller = Controller(node_config_list.copy())
    # controller.start()
    server_port_info = broker_controller.port_info

    # port for broker client
    train_port = server_port_info["recv"]["port"]
    predict_port = list([_d["port"] for _d in server_port_info["send"]])

    server_ip = get_host_ip()
    local_ip = "127.0.0.1"
    for index, data in enumerate(node_config_list):
        ip = data[0]

        if ip in (server_ip, local_ip):
            try:
                launch_local_broker(index, train_port, predict_port[index],
                                    local_ip, verbosity)
                logging.info("launch local broker with lib success")
            except BaseException as err:
                logging.exception(err)
        else:
            user = data[1]
            passwd = data[2]

            _remote_env = config_info.get("remote_env")
            if not _remote_env:
                logging.fatal("remote node must assign conda env")
            launch_remote_broker(
                user,
                passwd,
                ip,
                server_ip,
                index,
                train_port,
                predict_port[index],
                remote_env=_remote_env,
                verbosity=verbosity,
            )

    return broker_controller
