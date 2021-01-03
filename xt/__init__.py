"""
XT init module.

We register all the system module in here.
"""
from xt.train import main as train
from xt.evaluate import main as evaluate
from xt.benchmarking import main as benchmarking
from xt.framework.broker_launcher import start_broker_elf as start_broker
from zeus.common.util.register import register_xt_defaults
register_xt_defaults()

__version__ = '0.3.0'

__ALL__ = ["train", "evaluate", "benchmarking", "start_broker"]
