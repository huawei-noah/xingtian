import os
import time
import logging
from multiprocessing import Process

import moxing as mox
from config.conf import ResourceConfig, TrainingConfig
from lib.utils import sorted_custom


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] [%(message)s]",
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )


class Exchanger:
    def __init__(self):
        self.info = "This class is created for exchange data from yundao and local machine."

    def donwload_block_games(self):
        while True:
            try:
                data_dirs = mox.file.list_directory(ResourceConfig.new_data_yundao_dir)
                if not data_dirs:
                    mox.file.make_dirs(os.path.join(ResourceConfig.new_data_yundao_dir, '0'))
                else:
                    data_dirs = sorted_custom(data_dirs)
                    data_dirs = data_dirs[-3:]
                    for data_dir in data_dirs:
                        # checking latest three dirs
                        remote_newest_path = os.path.join(ResourceConfig.new_data_yundao_dir, str(data_dir))
                        local_newest_path = os.path.join(ResourceConfig.distributed_datadir, str(data_dir))
                        data_files = mox.file.list_directory(remote_newest_path)
                        if len(data_files) > ResourceConfig.block_min_games - 1 and \
                                not os.path.exists(local_newest_path):
                            mox.file.copy_parallel(remote_newest_path, local_newest_path)
                            logging.info('downloading data {}'.format(remote_newest_path))
            except:
                logging.error('donwload error')
                pass
            self.sleep(300)

    def upload_weight(self):
        while True:
            try:
                local_models = os.listdir(ResourceConfig.model_dir)
                local_models = sorted([i[:-6] for i in local_models if '.index' in i])

                remote_newest_path = os.path.join(ResourceConfig.pool_weights_yundao_dir, local_models[-1])
                if not mox.file.exists(remote_newest_path + '.index'):
                    for f in ['data-00000-of-00001', 'meta', 'index']:
                        src = os.path.join(ResourceConfig.model_dir, '{}.{}'.format(local_models[-1], f))
                        dst = '{}.{}'.format(remote_newest_path, f)
                        mox.file.copy(src, dst)
                    logging.info('uploading weights {}'.format(local_models[-1]))

                # delete oldest weight
                if len(local_models) > TrainingConfig.max_model_num:
                    oldest_weight = local_models[0]
                    for name in ['data-00000-of-00001', 'meta', 'index']:
                        file_name = '{}.{}'.format(oldest_weight, name)
                        os.remove(os.path.join(ResourceConfig.model_dir, file_name))
                        logging.info('deleting weight {}'.format(file_name))
            except:
                logging.error('upload error')
            self.sleep(300)

    def sleep(self, seconds):
        time.sleep(seconds)

    def upload_tf_record(self):
        while True:
            try:
                mox.file.copy_parallel(ResourceConfig.tensorboard_dir, ResourceConfig.tensorboard_yundao_dir)
            except:
                logging.info('upload tf error')
            self.sleep(600)

    def start(self):
        ps = []
        p = Process(target=self.donwload_block_games, name="worker_donwload_block_games")
        ps.append(p)
        p = Process(target=self.upload_weight, name="worker_upload_weight")
        ps.append(p)
        p = Process(target=self.upload_tf_record, name="upload_tf_record")
        ps.append(p)
        for i in ps:
            i.start()
