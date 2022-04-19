from cmath import pi
import os
import pandas as pd
import torch
import numpy as np

def read_input():
    """
    Read input, to tensor.
    """
    input_file = "F:/UnityGames/SPHGPU/dataset/input.csv"
    data = pd.read_csv(input_file, header=None)
    data = np.array(data)
    data = torch.tensor(data[:,:-1]) # the last column will be nan, just drop it
    os.remove(input_file)

def write_output():
    output_file = "F:/UnityGames/SPHGPU/dataset/output.csv"
    data = np.array(data)
    data = pd.DataFrame(data)
    data.to_csv(output_file, index=False, header=None)