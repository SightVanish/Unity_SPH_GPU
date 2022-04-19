import pandas as pd
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils import data
import time

"""
Read data.
"""
input_file = "F:/UnityGames/SPHGPU/dataset/input.csv"
input_data = pd.read_csv(input_file, header=None)
input_data = np.array(input_data)
input_data = torch.tensor(input_data[:,:-1]) # the last column will be nan, just drop it
# print(input_data)

"""
Process data.
"""
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


"""
Build model.
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 1)


    def forward(self, x):
        x = self.fc2(torch.tanh(self.fc1(x)))
        return x.squeeze(-1)

"""
Training.
""" 
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

"""
Save model.
"""
save_path = "F:/UnityGames/SPHGPU/Pytorch_scripts/model.pth"
torch.save(network, save_path) 
# torch.save(network.state_dict(), save_path)
print("Save model at " + save_path)
model_dict=torch.load(save_path)
# model_dict = network.load_state_dict(torch.load(save_path))