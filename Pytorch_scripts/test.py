import os
import struct
import numpy as np
fpath = "../dataset/input.bin"
f = open(fpath,'rb')
nums=int(os.path.getsize(fpath)/4)
data = struct.unpack('f'*nums,f.read(4*nums))
f.close()
data =  np.array(data)
print(data)
print(data.shape)