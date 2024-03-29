# from utils import load_data, apply_to_channels
from loaders.load_data_lens import _LensData

import numpy as np
import matplotlib.pyplot as plt


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


from adamatch.data import Wiener, MinMaxNormalizeImage
from torchvision import transforms

test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(101),
                                         MinMaxNormalizeImage(),
                                         ])


source_dataset_train_weak = _LensData( 
                                transform=test_transform,
                                root="./data",
                                datatype="easy",
                                isLabeled=True,
                                use_cached = True,
                                permute = False
                                )



#! PLOT IMAGES
fig, axes = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(15,12))
plt.axis('off')

axes = axes.flatten()

for i in range(0, int(len(axes)/2), 1):
    axes[i].set_title(source_dataset_train_weak[i][1])
    axes[i].imshow(source_dataset_train_weak[i][0].transpose(0, 2))
    im2 = Wiener()(source_dataset_train_weak[i][0]).transpose(0, 2)
    axes[i+4].set_title(source_dataset_train_weak[i][1])
    axes[i+4].imshow(im2)


plt.tight_layout()
plt.show()