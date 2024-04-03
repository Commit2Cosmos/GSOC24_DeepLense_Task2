import numpy as np
import torch
import torch.nn.functional as F

from adamatch.adamatch import Adamatch



def l2_normalize(vector):
    
    # Sum all the squared elements
    sum_squared_vector = np.sum(np.square(vector))
    
    # Take the square root of the sum
    norm = np.sqrt(sum_squared_vector)
    
    # Divide each element in the vector by the square root
    normalized_vector = vector / norm
    
    return normalized_vector



vector = np.array([0.2, 0.2])

normalized_vector = l2_normalize(vector)
print(normalized_vector)



def custom_function(x):
    return x / torch.sqrt(2 * x ** 2 - 2 * x + 1)


result = custom_function(torch.tensor(vector))

print(result)