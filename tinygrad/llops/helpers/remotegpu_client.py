import zmq
import numpy as np
import json
import os

remote_gpu_client = None

def encodearray(arr):
    shapearray = np.array(arr.shape)
    return [
        str(arr.dtype).encode("utf-8"),
        arr.tobytes(),
        shapearray.tobytes(),
    ]

def decodearray(parts):
    dtype = np.dtype(parts[0].decode("utf-8"))
    size = np.frombuffer(parts[2], dtype=np.int64)
    return np.frombuffer(parts[1], dtype=dtype).reshape(size)

class RemoteGPUClient:
    def __init__(self, host, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect(f"tcp://{host}:{port}")
    
    def runconv(self, array, weights, convargs):
        to_send = []
        to_send.extend(encodearray(array))
        to_send.extend(encodearray(weights))
        to_send.append(json.dumps(convargs).encode("utf-8"))
        self.socket.send_multipart(to_send)
        response = self.socket.recv_multipart()
        return decodearray(response)

def get_remote_gpu_client():
    global remote_gpu_client
    if not remote_gpu_client:
        remote_gpu_client = RemoteGPUClient(
            os.environ.get("RGPU_HOST"),
            os.environ.get("RGPU_PORT"),
        )
    return remote_gpu_client


