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
import subprocess
import os
from time import sleep

import numpy as np
from zeus.visual.tensorboarder import SummaryBoard


def test(logdir_root="/tmp/.xt_data/tensorboard"):
    """
    test for tensorboard
    :param logdir_root: tensorboard file path
    """
    if not os.path.isdir(logdir_root):
        os.makedirs(logdir_root)

    summary = SummaryBoard(logdir_root)
    sleep(1)
    summary2 = SummaryBoard(logdir_root)

    for dummy_index in range(1, 500, 2):
        loss = -dummy_index * 10 * np.log10(dummy_index)
        reward = dummy_index * np.exp2(dummy_index / 100)

        if dummy_index % 5 == 0:
            summary.add_scalar("data/train_loss", loss, dummy_index / 5,
                               walltime=dummy_index, flush=True)
            summary2.add_scalar("data/train_reward", reward, dummy_index / 5,
                                walltime=dummy_index, flush=True)

    vision_call = subprocess.Popen(
        "tensorboard --logdir={}".format(logdir_root), shell=True
    )
    vision_call.wait()


if __name__ == "__main__":
    test()
