import os
import struct
import numpy as np
from array import array
# read data
fpath = "../dataset/input.bin"
f = open(fpath,'rb')
nums=int(os.path.getsize(fpath)/4)
data = struct.unpack('f'*nums,f.read(4*nums))
f.close()
data =  np.array(data, dtype=np.float32)
data = data.reshape(-1, 80000, 12)
print(data.shape)
print(data[0, 0,:])
# output data
"""
data = data.tolist()
s = struct.pack('f'*len(data), *data)
f = open("../dataset/output.bin",'wb')
f.write(s)
f.close()
"""