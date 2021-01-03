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
Manage remote node.

Contains:
    0. distribute xingtian wheel to remote node;
    1. setup broker;
    2. distribute model file to remote node for explore.
"""
import os
import glob
import shutil
from time import sleep
import subprocess
from absl import logging
from fabric2 import Connection
from zeus.common.util.common import get_host_ip

# logging.set_verbosity(logging.DEBUG)


def remote_run(server_ip, host, passwd, cmd, remote_env):
    """Run command in remote node."""
    print("remote_env:", remote_env)
    _env_export = "export PATH={}/bin:$PATH".format(remote_env["conda"])
    if "env" in remote_env.keys():
        for _key, _val in remote_env["env"].items():
            _env_export += "&& export {}={}".format(_key, _val)

    with Connection(
            server_ip, user=host, connect_kwargs={"password": passwd}
    ) as connect:
        with connect.prefix("{}".format(_env_export)):
            connect.run(cmd, pty=False, echo=True)


def __check_both_env_are_local(env_config_list, local_ip_set):
    env_ip_list = [_data[0] for _data in env_config_list]
    for _ip in env_ip_list:
        if _ip not in local_ip_set:
            return False
    return True


def dist_model(src_model, node_info, remote_path="xt_archive"):
    """
    Distribute model to remote node.

    :param src_model:
    :param node_info:
    :param remote_path:
    :return:
    """
    # optimize local node check
    if node_info[0] in ("127.0.0.1",):
        return None
    server_ip = get_host_ip()
    if node_info[0] == server_ip:
        return None

    _basename = os.path.basename(src_model)
    if _basename in ("none", "None", None):  # init model with (none, none)
        return None

    target_file = glob.glob("{}*".format(src_model))

    _ip, _user, _password = node_info
    destination_model = os.path.join("/home", _user, remote_path + "/")
    with Connection(_ip, user=_user,
                    connect_kwargs={"password": _password}
                    ) as connect:
        # fixme: multi-case running on the same node
        _workspace = os.path.join("/home", _user, remote_path)
        for _item in target_file:
            logging.debug("dist model: {}--> {}".format(_item, destination_model))
            connect.put(_item, destination_model)
    return [destination_model + _basename]


def _package_xt(default_dist_path="./dist"):
    """
    Make package as wheel with `python3 setup.py bdist_wheel.

    :param default_dist_path:
    :return:
    """
    # # remove old zeus
    # if os.path.exists("zeus"):
    #     shutil.rmtree("zeus")

    if not os.path.exists("zeus"):
        shutil.copytree("../zeus", "zeus", ignore=shutil.ignore_patterns('*.pyc'))
        sleep(0.05)

    _cmd = "python3 setup.py bdist_wheel --universal"
    try:
        subprocess.call([_cmd], shell=True,  # stdout=subprocess.PIPE
                        )
    except subprocess.CalledProcessError as err:
        logging.fatal("catch err: {} when package into wheel".format(err))

    return default_dist_path


def distribute_xt_if_need(config, remote_env, remote_path="xt_archive"):
    """
    Distribute Xingtian sourcecode among use's node configure.

    :param config: config instance from config.yaml
    :param remote_env: remote conda environment path
    :param remote_path: path to store the wheel file. 'xt_archive' default.
    :return:
    """
    local_ip = get_host_ip()

    # check could if distribute or not
    remote_ip_list = list()
    for _key in (
            "node_config",
            "test_node_config",
    ):
        if _key not in config.keys():
            continue
        for _ip, _user, _password in config.get(_key):
            # local need not distribute
            if _ip in (local_ip, "127.0.0.1"):
                continue
            remote_ip_list.append((_ip, _user, _password))

    if not remote_ip_list:
        logging.debug("Don't distribute xingtian without remote ip set.")
        return True

    dist_path = _package_xt()
    if not remote_env:
        logging.fatal("must assign remote env in yaml.")

    for _ip, _user, _password in remote_ip_list:
        with Connection(
                _ip, user=_user, connect_kwargs={"password": _password}
        ) as connect:
            _workspace = os.path.join("/tmp")
            target_whl = glob.glob("{}/xingtian*.whl".format(dist_path))
            logging.info("found dist: {}".format(target_whl))
            for _whl in target_whl:
                _name = os.path.basename(_whl)
                _remote_cmd = "pip install {}/{} --upgrade --force-reinstall --no-deps".format(
                    _workspace, _name)
                logging.info(
                    "re-install xingtian in remote-{} conda env {} >>> \n"
                    "{}".format(_ip, remote_env["conda"], _remote_cmd)
                )

                connect.put(os.path.join(dist_path, _name), remote=_workspace)
                with connect.prefix("export PATH={}/bin:$PATH".format(remote_env["conda"])):
                    connect.run(_remote_cmd, pty=False)
