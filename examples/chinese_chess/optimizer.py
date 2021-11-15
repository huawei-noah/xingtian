import logging
import os

import tensorflow as tf

from server import Exchanger
from server import Trainer
from lib.utils import init_dir, sorted_custom
from config.conf import ResourceConfig
import moxing as mox


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def restart_training():
    data_dirs = mox.file.list_directory(ResourceConfig.new_data_yundao_dir)
    data_dirs = sorted_custom(data_dirs)
    data_dirs = data_dirs[-110:-1]
    for data_dir in data_dirs:
        try:
            mox.file.copy_parallel(
                os.path.join(ResourceConfig.new_data_yundao_dir, data_dir),
                os.path.join(ResourceConfig.distributed_datadir, data_dir)
            )
        except:
            logging.error('download {} error'.format(data_dir))
    logging.info('download done')

    latest_weights = mox.file.list_directory(ResourceConfig.pool_weights_yundao_dir)
    latest_weights = [weight[:-6] for weight in latest_weights if '.index' in weight]
    latest_weights.sort()
    latest_weight = latest_weights[-1]
    for name in ['data-00000-of-00001', 'meta', 'index']:
        file_name = '{}.{}'.format(latest_weight, name)
        mox.file.copy(
            os.path.join(ResourceConfig.pool_weights_yundao_dir, file_name),
            os.path.join(ResourceConfig.model_dir, file_name)
        )


if __name__ == "__main__":
    restart_training()
    init_dir()
    # run exchanger
    exchanger = Exchanger()
    exchanger.start()
    # start training
    optimizer = Trainer()
    optimizer.train()
