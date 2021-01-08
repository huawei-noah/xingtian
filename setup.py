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
import os
import re
import codecs
from glob import glob
from setuptools import find_packages, setup
from subprocess import Popen, PIPE

here = os.path.abspath(os.path.dirname(__file__))
# print(here)

com_smi = Popen(['command -v nvidia-smi'], stdout=PIPE, shell=True)
com_out = com_smi.communicate()[0].decode("UTF-8")
allow_gpu = com_out != ""

install_requires = list()

with codecs.open(os.path.join(here, 'requirements.txt'), 'r') as rf:
    for line in rf:
        package = line.strip()
        install_requires.append(package)
# if allow_gpu:
#     install_requires.append("tensorflow-gpu==1.15.0")
# else:
#     install_requires.append("tensorflow==1.15.0")

with open(os.path.join(here, 'xt', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
    name="xingtian",
    version=version,
    python_requires=">=3.6.*",
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages(),
    data_files=[
        ("xt", glob("xt/environment/sumo/*/sumo*")),
    ],
    description="XingTian Library enables easy usage on the art "
                "Reinforcement Learning algorithms.",
    author="XingTian development team",
    license="MIT",
    url="https://github.com/huawei-noah/xingtian",
    entry_points={
        'console_scripts': [
            'xt_main=xt.main:main',
        ],
    }
)
