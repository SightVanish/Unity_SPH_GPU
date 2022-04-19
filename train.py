from cmath import pi
import os
import pandas as pd
import torch
import numpy as np
"""
Read input.
"""

input_file = "F:/UnityGames/SPHGPU/input.csv"
output_file = "F:/UnityGames/SPHGPU/output.csv"
data = pd.read_csv(input_file, header=None)
data = np.array(data)
data = torch.tensor(data[:,:-1]) # the last column will be nan, just drop it
# print(data)

data = data * pi

data = np.array(data)
data = pd.DataFrame(data)
data.to_csv(output_file, index=False, header=None)

os.remove(input_file)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data
import time
from torch.autograd import Variable

num_data = 10000
male_heights = np.random.normal(171, 6, num_data)
female_heights = np.random.normal(158, 5, num_data)

male_weights = np.random.normal(70, 10, num_data)
female_weights = np.random.normal(57, 8, num_data)

male_bfrs = np.random.normal(16, 2, num_data)
female_bfrs = np.random.normal(22, 2, num_data)

male_labels = [1] * num_data
female_labels = [-1] * num_data

train_set = np.array([np.concatenate((male_heights, female_heights)), 
                      np.concatenate((male_weights, female_weights)), 
                      np.concatenate((male_bfrs, female_bfrs)), 
                      np.concatenate((male_labels, female_labels)),]).T
np.random.shuffle(train_set)
train_set = torch.tensor(train_set, dtype=torch.float32)

def data_loader(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array) # construct a dataset
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(3, 3)
      self.fc2 = nn.Linear(3, 1)
      # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
      # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
      # self.conv2_drop = nn.Dropout2d()
      # self.fc1 = nn.Linear(320, 50)
      # self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
      x = self.fc2(torch.tanh(self.fc1(x)))
      # return x.squeeze(-1)
      return x
      # x = F.relu(F.max_pool2d(self.conv1(x), 2))
      # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
      # x = x.view(-1, 320)
      # x = F.relu(self.fc1(x))
      # x = F.dropout(x, training=self.training)
      # x = self.fc2(x)
      # return F.log_softmax(x)
    
n_epochs = 5
batch_size = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 20

random_seed = 1
torch.backends.cudnn.enabled = True


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
loss = nn.MSELoss()
device = torch.device('cuda:0')
device = torch.device('cpu')

def train():
  print('training on', device)
  data_iter = data_loader([train_set[:,:3], train_set[:,-1]], batch_size, True)
  network.to(device)

  start_time = time.time()
  for epoch in range(n_epochs):
    
    for X, y in data_iter:
      X, y = X.to(device), y.to(device)

      l = loss(network(X), y)
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
    l=loss(network(train_set[:,:3].to(device)), train_set[:,-1].to(device))
    
    print(f'epoch: {epoch+1}, loss: {l:f}')
  end_time = time.time()
  print(f'time: {end_time-start_time}')






train()
dummy_input = Variable(torch.randn(3))
torch.onnx.export(network.to(device), dummy_input.to(device), 'mlp.onnx')

input_names = [ "input_0" ]
output_names = [ "output_0" ]
# torch.onnx.export(network, 'mnist.onnx')
# torch.onnx.export(network, dummy_input, , input_names=input_names, output_names=output_names)
"""