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
import random
import torch

train_transform_strong = transforms.Compose([transforms.ToTensor(),
                                                #  transforms.Resize(101),
                                                 transforms.RandomAutocontrast(),
                                                 #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                 #transforms.RandomEqualize(), # only on PIL images
                                                 transforms.RandomInvert(),
                                                 #transforms.RandomPosterize(random.randint(1, 8)), # only on PIL images
                                                 transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                                                 transforms.RandomSolarize(random.uniform(0, 1)),
                                                #  transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                                                 transforms.RandomErasing(),
                                                 MinMaxNormalizeImage()
                                                 ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                        #  transforms.Resize(101),
                                         Wiener(),
                                         MinMaxNormalizeImage(),
                                         ])


source_dataset_strong = _LensData( 
                                transform=train_transform_strong,
                                root="./data",
                                datatype="easy",
                                isLabeled=True,
                                use_cached = True,
                                permute = True
                                )

source_dataset_og = _LensData( 
                                transform=test_transform,
                                root="./data",
                                datatype="easy",
                                isLabeled=True,
                                use_cached = True,
                                permute = True
                                )



#! PLOT IMAGES
fig, axes = plt.subplots(2, 6, sharex='all', sharey='all', figsize=(15,12))
plt.axis('off')

axes = axes.flatten()

for i in range(0, int(len(axes)/2), 1):
    # print(torch.max(source_dataset_og[i][0].transpose(0, 2)))
    # print(torch.min(source_dataset_og[i][0].transpose(0, 2)))
    axes[i].set_title(source_dataset_og[i][1])
    axes[i].imshow(source_dataset_og[i][0].transpose(0, 2))
    # print(torch.max(source_dataset_strong[i][0].transpose(0, 2)))
    # print(torch.min(source_dataset_strong[i][0].transpose(0, 2)))
    axes[i+6].set_title(source_dataset_strong[i][1])
    axes[i+6].imshow(source_dataset_strong[i][0].transpose(0, 2))


# plt.tight_layout()
# plt.show()