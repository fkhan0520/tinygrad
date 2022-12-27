import zmq
import time
import numpy as np
from collections import namedtuple
import torch
import base64
import json
import lzma

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
# socket.setsockopt(zmq.SNDBUF, 256 * 1024)
socket.connect(f"tcp://35.247.25.188:{port}")


xshape = (4, 512, 21, 21)
yshape = (1024, 512, 3, 3)
inC = {'H': 3, 'W': 3, 'groups': 1, 'rcout': 1024, 'cin': 512, 'oy': 19, 'ox': 19, 'iy': 19, 'ix': 19, 'sy': 1, 'sx': 1, 'bs': 1, 'cout': 1024, 'py': 1, 'py_': 1, 'px': 1, 'px_': 1, 'dy': 1, 'dx': 1, 'out_shape': (4, 1024, 19, 19)}

convargs = namedtuple('convargs', inC)
C = convargs(**inC)

def encodearray(arr):
    shapearray = np.array(arr.shape)
    print(arr.size * arr.itemsize)
    return [str(arr.dtype).encode('utf-8'), arr.tobytes(), np.array(arr.shape).tobytes()]


def runconv(x, w, C):
    return torch.conv2d(x, w, stride=(C['sy'], C['sx']), groups=C['groups'], dilation=(C['dy'], C['dx']))

def runconv_np(x,w,C):
  arg = (
    (C.bs, C.groups*C.cin*x.shape[2]*x.shape[3]), (C.groups, C.cin*x.shape[2]*x.shape[3]),
    (C.oy, C.sy*x.shape[3]), (C.ox, C.sx), (C.cin, x.shape[2]*x.shape[3]), (C.H, C.dy*x.shape[3]), (C.W, C.dx))
  tx = np.lib.stride_tricks.as_strided(x.ravel().reshape(x.shape), shape=[y[0] for y in arg], strides=[y[1]*x.dtype.itemsize for y in arg])
  tw = w.reshape(C.groups, C.rcout, C.cin, C.H, C.W)
  out = np.einsum("nGhwCHW, GkCHW -> nGkhw", tx.ravel().reshape(tx.shape), tw.ravel().reshape(tw.shape))
  return out.reshape(C.bs, C.groups*C.rcout, C.oy, C.ox)

for i in range(10):
    x2raw = torch.randn(*xshape)
    w2raw = torch.randn(*yshape)
    start = time.time()
    x2n = x2raw.numpy()
    w2n = w2raw.numpy()
    # d = {
    #     "x": encodearray(x2n),
    #     "w": encodearray(w2n),
    #     "C": inC,
    # }
    to_send = []
    to_send.extend(encodearray(x2n))
    to_send.extend(encodearray(w2n))
    to_send.extend([json.dumps(inC).encode('utf-8')])
    mp = socket.send_multipart(to_send)
    print(mp, type(mp))
    msg = socket.recv_multipart()
    print(len(msg), time.time() - start)

    cstart = time.time()
    res = runconv(x2raw, w2raw, inC)
    print(f"local conv takes {time.time() - cstart}")
    npstart = time.time()
    res = runconv_np(x2n, w2n, C)
    print(f"local np conv takes {time.time() - npstart}")
    print("--------------")

    # start = time.time()
    # socket.send_pyobj([1,3,4])
    # print("serialize/send time = ", time.time() - start)
    # msg = socket.recv_pyobj()
    # print(type(msg))
    # print(time.time()-start)

