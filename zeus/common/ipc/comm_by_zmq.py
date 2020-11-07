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
from zeus.common.util.register import Registers


@Registers.comm
class CommByZmq(object):
    """Communication by zmq."""

    def __init__(self, comm_info):
        """Initialize."""
        super(CommByZmq, self).__init__()
        # For master, there is no 'addr' parameter given.
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

        if "*" in addr:
            socket.bind("tcp://*:" + str(port))
        else:
            socket.connect("tcp://" + str(addr) + ":" + str(port))

        self.socket = socket

    def send(self, data, name=None, block=True):
        """Send message."""
        # msg = pickle.dumps(data)
        msg = pyarrow.serialize(data).to_buffer()
        self.socket.send(msg)

    def recv(self, name=None, block=True):
        """Receive message."""
        msg = self.socket.recv()
        data = pyarrow.deserialize(msg)
        # data = pickle.loads(msg)
        return data

    def send_bytes(self, data):
        """Send bytes."""
        self.socket.send(data, copy=False)

    def recv_bytes(self):
        """Receive bytes."""
        data = self.socket.recv()
        return data

    def close(self):
        """Close."""
        if self.socket:
            self.socket.close()
