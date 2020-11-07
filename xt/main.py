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
# THE SOFTWARE
"""
DESC: Main entrance for xingtian library.

Usage:
    python main.py -f examples/default_cases/cartpole_ppo.yaml -t train
"""

import argparse
import pprint
import yaml
from absl import logging

from xt.train import main as xt_train
from xt.train import makeup_multi_case
from xt.evaluate import main as xt_eval
from xt.benchmarking import main as xt_benchmarking
from zeus.common.util.get_xt_config import parse_xt_multi_case_paras
from zeus.common.util.get_xt_config import check_if_patch_local_node
from zeus.common.util.get_xt_config import OPEN_TASKS_SET
from zeus.common.util.logger import VERBOSITY_MAP
from xt.framework.remoter import distribute_xt_if_need
from zeus.common.util.logger import set_logging_format
set_logging_format()
# logging.set_verbosity(logging.INFO)


def main():
    """:return: config file for training or testing."""
    parser = argparse.ArgumentParser(description="XingTian Usage.")

    parser.add_argument(
        "-f", "--config_file", required=True, help="""config file with yaml""",
    )
    # fixme: split local and hw_cloud,
    #  source path could read from yaml startswith s3
    parser.add_argument(
        "-s3", "--save_to_s3", default=None, help="save model/records into s3 bucket."
    )
    parser.add_argument(
        "-t",
        "--task",
        # required=True,
        default="train",
        choices=list(OPEN_TASKS_SET),
        help="task choice to run xingtian.",
    )
    parser.add_argument(
        "-v", "--verbosity", default="info", help="logging.set_verbosity"
    )

    args, _ = parser.parse_known_args()
    if _:
        logging.warning("get unknown args: {}".format(_))

    if args.verbosity in VERBOSITY_MAP.keys():
        logging.set_verbosity(VERBOSITY_MAP[args.verbosity])
        pass
    else:
        logging.warning("un-known logging level-{}".format(args.verbosity))

    _exp_params = pprint.pformat(args, indent=0, width=1,)
    logging.info(
        "\n{}\n XT start work...\n{}\n{}".format("*" * 50, _exp_params, "*" * 50)
    )

    with open(args.config_file, "r") as conf_file:
        _info = yaml.safe_load(conf_file)

    _info = check_if_patch_local_node(_info, args.task)
    distribute_xt_if_need(config=_info, remote_env=_info.get("remote_env"))

    if args.task in ("train", "train_with_evaluate"):
        ret_para = parse_xt_multi_case_paras(args.config_file)
        if len(ret_para) > 1:
            makeup_multi_case(args.config_file, args.save_to_s3)
        else:
            xt_train(args.config_file, args.task, args.save_to_s3, args.verbosity)

    elif args.task == "evaluate":
        xt_eval(args.config_file, args.save_to_s3)

    elif args.task == "benchmark":
        # fixme: with benchmark usage in code.
        # xt_benchmark(args.config_file)
        xt_benchmarking()
    else:
        logging.fatal("Get invalid task: {}".format(args.task))


if __name__ == "__main__":
    main()
