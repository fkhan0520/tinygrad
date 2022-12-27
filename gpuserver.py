import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind(f"tcp://*:{port}")

while True:
    print("listening...")
    msg = socket.recv_pyobj()
    print(type(msg), len(msg))
    socket.send_pyobj(msg["x"])