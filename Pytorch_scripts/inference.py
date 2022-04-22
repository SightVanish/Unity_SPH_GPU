import logging
import os
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
from torch_geometric.data import Data
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import socket

save_path = "./model.pth"
print("Inference on " + str(device))

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(11, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 8)
        self.conv5 = GCNConv(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv4(x, edge_index)

        return x

model = GNN()
model.load_state_dict(torch.load(save_path, map_location=device))
print("Abstract of model:")
print(model)

"""
Basic properties,
"""
num_particles = 4000
num_neighbours = 200 * 8

"""
Connect to Unity.
"""

"""
host, port = "127.0.0.1", 25001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))


frame_counter = 0
while True: 
    # features
    features_data = bytes[0]
    for i in range(num_particles):
        features_data += sock.recv(4*11)
    features = sock.recv(4*num_particles*11)
    features = np.frombuffer(features, dtype=np.float32)
    features = torch.tensor(features.reshape(num_particles, -1), dtype=torch.float32)
    
    # # neighbours_track
    # neighbours_track = sock.recv(4*num_particles)
    # neighbours_track = np.frombuffer(neighbours_track, dtype=np.int32)

    # # neighbour_list
    # neighbour_list = sock.recv(4 * np.sum(neighbours_track))
    # neighbour_list = np.frombuffer(neighbour_list, dtype=np.int32)

    # # neighbour_list--test
    # neighbour_list = sock.recv(4 * num_particles * num_neighbours)
    # neighbour_list = np.frombuffer(neighbour_list, dtype=np.int32).reshape(num_particles, num_neighbours)

    # # construct dataset
    # edge_index = torch.zeros([np.sum(neighbours_track), 2], dtype=torch.long)
    # index = 0
    # for i in range(num_particles):
    #     current_num_neis = neighbours_track[i]
    #     for j in range(current_num_neis):
    #         # if (index < )
    #         edge_index[index, 0] = i
    #         edge_index[index, 1] = neighbour_list[i, j]
    #         index = index + 1

    # edge_index = torch.zeros([2, 2], dtype=torch.long)
    # edge_index[0, 1] = 1
    # edge_index[0, 1] = 2
    # edge_index[1, 1] = 2
    # edge_index[1, 1] = 1

    # dataset = Data(x=features, edge_index=edge_index.t().contiguous())
    # output = model(dataset)

    
    # output = torch.zeros(num_particles, dtype=torch.float32)
    # output_bytes = output.detach().numpy().tobytes()
    # sock.sendall(output_bytes)
    
    # frame_counter = frame_counter + 1
    # print("num frame: {:5}".format(frame_counter))
    """


def sending_and_reciveing():
    s = socket.socket()
    socket.setdefaulttimeout(None)
    print('socket created ')
    port = 60000
    s.bind(('127.0.0.1', port)) #local host
    s.listen(30) #listening for connection for 30 sec?
    print('socket listensing ... ')
    while True:
        c, addr = s.accept() #when port connected

        print("Received:")
        bytes_received = c.recv(4000*11*4)
        print(len(bytes_received)/4)
        
        array_received = np.frombuffer(bytes_received, dtype=np.float32)       

        # while (len(results) > 0):
        #     print(results)
        #     results = s.recv(4096)
        print("Received:")
        print(array_received.shape)

        time.sleep(1)

        bytes_to_send = 2 * array_received
        print("Sent:")
        print(bytes_to_send.shape)
        
        # bytes_to_send = np.array([1,2,3], dtype=np.float32)
        c.sendall(bytes_to_send.tobytes()) #sending back
        c.close()
        

sending_and_reciveing() 