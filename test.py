import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(12)

probs = torch.rand(10,)
print(probs)

probs = torch.where(probs < 0.5, 1-probs, probs)
print(probs)