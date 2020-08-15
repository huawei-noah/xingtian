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
import socket
import time
from subprocess import Popen

import redis

MAX_ACTOR_NUM = 40
MAX_LEARNER_NUM = 10
START_PORT = 20000
PORTNUM_PERLEARNER = MAX_ACTOR_NUM + 1


class CommConf(object):
    def __init__(self):
        try:
            redis.Redis(host="127.0.0.1", port=6379, db=0).ping()
        except redis.ConnectionError:
            Popen("echo save '' | setsid redis-server -", shell=True)
            time.sleep(0.3)

        self.redis = redis.Redis(host="127.0.0.1", port=6379, db=0)
        self.pool_name = "port_pool"
        if not self.redis.exists(self.pool_name):
            self.init_portpool()

    def init_portpool(self):
        ''' init port pool '''
        start_port = START_PORT
        try_num = 10

        for _ in range(MAX_LEARNER_NUM):
            for _ in range(try_num):
                check_flag, next_port = self.check_learner_port(start_port)
                if not check_flag:
                    break
                else:
                    start_port = next_port

            self.redis.lpush(self.pool_name, start_port)
            self.redis.incr('port_num', amount=1)
            self.redis.incr('max_port_num', amount=1)

            start_port = next_port

    def get_start_port(self):
        ''' get start port '''
        if int(self.redis.get('port_num')) == 0:
            raise Exception("Dont have available port")

        start_port = self.redis.lpop(self.pool_name)
        self.redis.decr('port_num', amount=1)
        return int(start_port)

    def release_start_port(self, start_port):
        ''' release start port '''
        self.redis.lpush(self.pool_name, start_port)
        self.redis.incr('port_num', amount=1)

        if self.redis.get('port_num') == self.redis.get('max_port_num'):
            self.redis.delete('port_num')
            self.redis.delete('max_port_num')
            self.redis.delete('port_pool')
            print("shutdown redis")
            self.redis.shutdown(nosave=True)

        return

    def check_learner_port(self, start_port):
        ''' check if multi-port is in use '''
        ip = "localhost"
        for i in range(PORTNUM_PERLEARNER):
            if self.check_port(ip, start_port + i):
                return True, start_port + i + 1
        return False, start_port + PORTNUM_PERLEARNER

    def check_port(self, ip, port):
        ''' check if port  is in use '''
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, int(port)))
            s.shutdown(2)
            print("port is used", int(port))
            return True
        except BaseException:
            return False


def get_port(start_port):
    ''' get port used by module '''
    predict_port = start_port + 1
    if (predict_port + MAX_ACTOR_NUM - start_port) > PORTNUM_PERLEARNER:
        raise Exception("port num is not enough")

    return start_port, predict_port


def test():
    ''' test interface'''
    test_comm_conf = CommConf()
    redis_key = 'port_pool'
    print("{} len: {}".format(redis_key, test_comm_conf.redis.llen(redis_key)))
    for _ in range(test_comm_conf.redis.llen(redis_key)):
        pop_val = test_comm_conf.redis.lpop(redis_key)
        print("pop val: {} from '{}'".format(pop_val, redis_key))
    start = time.time()

    test_comm_conf.init_portpool()
    print("use time", time.time() - start)

    train_port = get_port(20000)
    print(train_port)


if __name__ == "__main__":
    test()
