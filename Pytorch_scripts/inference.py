import socket
import time
import numpy as np
host, port = "127.0.0.1", 25001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

while True:
    receivedData = sock.recv(1024)
    y = np.frombuffer(receivedData, dtype=np.float32)
    print("Received: ")
    print(y)
    y = y * 2
    y = y.tobytes()
    sock.sendall(y)
    print("Send: ")
    print(np.frombuffer(y, dtype=np.float32))
