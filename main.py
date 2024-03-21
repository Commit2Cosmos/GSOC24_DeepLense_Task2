from utils import load_data, apply_to_channels
from loaders.load_data_lens import _LensData

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import wiener



# ROOT = './data'

# #* Image dims: n x 3 x 101 x 101
# #* Labels count: 0 (lens): 6000 ; 1 (not lens): 6000
# X, y = load_data(ROOT, 'easy', use_cached=True, permute=True)

# print("X shape: ", X.shape)
# print("y shape: ", y.shape)


# pr = list(X)



#! TEST DENOISE_NL
# import time

# tic = time.perf_counter()

# for i in X:
#     pr.append(wiener(i))


# print(f'time taken: {time.perf_counter()-tic}')

# print(X.shape)



#! TESTING DATALOADER

from torchvision import transforms


rotation = 0
translate = (0.0, 0.0)
scale = (1.0, 1.0)


dataset = _LensData(transform=transforms.Compose([
                wiener,
                transforms.ToTensor(),
                transforms.RandomAffine(
                    degrees=rotation, 
                    translate=translate,
                    scale=scale
                ),
                transforms.RandomHorizontalFlip(),
            ]),
            datatype="easy",
            use_cached = True,
            permute = True)


print(len(dataset))




#! PLOT IMAGES
fig, axes = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(15,12))
plt.axis('off')

axes = axes.flatten()

for i, ax in enumerate(axes):
    # d = apply_to_channels(pr[i]).transpose((1,2,0))
    # d = dataset[i]

    ax.set_title(dataset[i][1])
    ax.imshow(dataset[i][0])

plt.tight_layout()
plt.show()