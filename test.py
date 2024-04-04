import numpy as np
import torch
import torch.nn.functional as F
from adamatch.evaluate import plot_cm_roc

np.random.seed(42)

size = 30

labels_list = np.random.randint(2, size=size)
outputs_list = np.random.rand(size)
preds_list = np.random.randint(2, size=size)

eval_output = [0.9, labels_list, outputs_list, preds_list]

plot_cm_roc(eval_output)