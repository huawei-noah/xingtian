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
"""Communication by zmq."""
import pyarrow
import zmq
from absl import logging
from zeus.common.util.register import Registers

ZMQ_MIN_PORT = 20000
ZMQ_MAX_PORT = 40000


@Registers.comm
class CommByZmq(object):
    """Communication by zmq."""

    def __init__(self, comm_info):
        """Initialize."""
        super(CommByZmq, self).__init__()
        # For Controller, there is no 'addr' parameter given.
        logging.debug("zmq start with comm_info: {}".format(comm_info))
        addr = comm_info.get("addr", "*")
        port = comm_info.get("port")
        zmq_type = comm_info.get("type", "PUB")

        comm_type = {
            "PUB": zmq.PUB,
            "SUB": zmq.SUB,
            "PUSH": zmq.PUSH,
            "PULL": zmq.PULL,
            "REP": zmq.REP,
            "REQ": zmq.REQ,
        }.get(zmq_type)

        context = zmq.Context()
        socket = context.socket(comm_type)

        self._type = zmq_type
        self.bound_port = None
        if "*" in addr:
            # socket.bind("tcp://*:" + str(port))
            bound_port = socket.bind_to_random_port("tcp://*",
                                                    min_port=ZMQ_MIN_PORT,
                                                    max_port=ZMQ_MAX_PORT,
                                                    max_tries=100)
            self.bound_port = bound_port
        else:
            socket.connect("tcp://" + str(addr) + ":" + str(port))

        self.socket = socket

    def send(self, ctr_info, data, name=None, block=True):
        """Send message."""
        # msg = pickle.dumps(data)
        msg = []
        msg.append(pyarrow.serialize(ctr_info).to_buffer())
        msg.append(pyarrow.serialize(data).to_buffer())
        self.socket.send_multipart(msg)

    def recv(self, name=None, block=True):
        """Receive message."""
        msg = self.socket.recv_multipart()
        ctr_info = pyarrow.deserialize(msg[0])
        data = pyarrow.deserialize(msg[1])
        # data = pickle.loads(msg)
        return ctr_info, data

    def send_bytes(self, ctr_info, data):
        """Send bytes."""
        msg = list()
        msg.append(ctr_info)
        msg.append(data)
        self.socket.send_multipart(msg, copy=False)

    def recv_bytes(self, block):
        """Receive bytes."""
        recv_data = self.socket.recv_multipart()
        ctr_info, data = recv_data[0], recv_data[1]

        return ctr_info, data

    def __str__(self):
        """Rewrite the ste func, to return the class info."""
        return str({
            "port": self.bound_port,
            "type": self._type
        })

    def close(self):
        """Close."""
        if self.socket:
            self.socket.close()
