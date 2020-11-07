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
Run benchmark.

Usage:  e.g,
python3 benchmarking.py -f config/.cartpole.yaml --start_datetime 20190624-163110

if arise multi-scaler on web, you NEED re-run tensorboard !!!  bugs

Notes
-----
    1) -f support list of config file
"""

import argparse
import subprocess

from zeus.common.util.default_xt import XtBenchmarkConf as xt_bm_config  # pylint: disable=C0413
from zeus.visual.visual_rewards import display_rewards  # pylint: disable=C0413


def main():
    """
    DESC: The main entrance for benchmark.

    Returns: tensorboard handler
    """
    parser = argparse.ArgumentParser(description="benchmark tools.")

    parser.add_argument(
        '-f',
        '--config_file',
        nargs="*",  # '+',
        # required=True,
        help="""Read Benchmark_id & agent_name form the (config file),
            support config file List""",
    )
    parser.add_argument('-d', "--data_path", nargs="*",
                        help="""read data from special paths.""")
    parser.add_argument('-s',
                        '--reward_set',
                        default="both",
                        choices=["both", "eval"],
                        help="""which reward to be display by tensorboard,
            default, usage 'eval', support 'eval' and 'both' now.
            'both' equal to 'eval & train' .""")

    parser.add_argument('--db_root',
                        default="{}".format(xt_bm_config.default_db_root),
                        help="the root path to read database file.")
    parser.add_argument('-x',
                        '--use_index',
                        default="step",
                        choices=["step", "sec"],
                        help="""x-axis setting, contains: 'step'&'sec', default 'step'.""")

    parser.add_argument('-o', '--output', default="tensorboard",
                        choices=["tensorboard", ],
                        help="""plot into image or display on tensorboard.""")

    args, _ = parser.parse_known_args()
    if _:
        print("get unknown args: {}".format(_))
    print("\n\nstart display with args: {} \n\n".format([(_arg, getattr(args, _arg)) for _arg in vars(args)]))

    if args.config_file:
        print(args.config_file)
        if args.output == "tensorboard":
            if args.reward_set in ["eval", "both"]:
                display_rewards(args, args.reward_set)
            else:
                print("Error: non-support reward_set value:{}, yet!".format(args.reward_set))
    else:
        print("start single history tensorboard...")

    # start tensorboard
    vision_call = subprocess.Popen("tensorboard --logdir={}".format(xt_bm_config.default_tb_path), shell=True)
    vision_call.wait()


if __name__ == "__main__":
    main()
