import numpy as np
import torch


np_arr = np.random.rand(3,4)
tensor = torch.asarray(np_arr)


mysize = 3

print(np.repeat(mysize, np_arr.ndim))
print(torch.repeat_interleave(mysize, len(tensor.shape)))