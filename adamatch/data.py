import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import random
from scipy.signal import correlate
from loaders.load_data_lens import _LensData
import numpy as np


class MinMaxNormalizeImage:
    def __call__(self, img: torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        normalized_tensor = (img - min_val) / (max_val - min_val)
        return normalized_tensor



class Wiener:
    def __call__(self, im: torch.Tensor, mysize=None, noise=None) -> torch.Tensor:
        
        im = np.asarray(im)
        if mysize is None:
            mysize = [4] * im.ndim
        mysize = np.asarray(mysize)

        if mysize.shape == ():
            mysize = np.repeat(mysize.item(), im.ndim)

        # Estimate the local mean
        lMean = correlate(im, np.ones(mysize), 'same') / np.prod(mysize, axis=0)

        # Estimate the local variance
        lVar = (correlate(im ** 2, np.ones(mysize), 'same') /
            np.prod(mysize, axis=0) - lMean ** 2)

        # Estimate the noise power if needed.
        if noise is None:
            noise = np.mean(np.ravel(lVar), axis=0)

        res = (im - lMean)
        res *= (1 - noise / lVar)
        res += lMean
        out = np.where(lVar < noise, lMean, res)

        return torch.tensor(out, dtype=torch.float32)



def _get_transforms():
    """
    The AdaMatch paper uses CTAugment as its strong augmentations. I'm going to
    create a pipeline of transforms similar to the ones used by CTAugment.
    """
    resize_size = 101

    train_transform_weak = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize(resize_size),
                                               Wiener(),
                                               transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                                               MinMaxNormalizeImage(),
                                               ])


    train_transform_strong = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(resize_size),
                                                 transforms.RandomAutocontrast(),
                                                 #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                                                 #transforms.RandomEqualize(), # only on PIL images
                                                 transforms.RandomInvert(),
                                                 #transforms.RandomPosterize(random.randint(1, 8)), # only on PIL images
                                                 transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                                                 transforms.RandomSolarize(random.uniform(0, 1)),
                                                 transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                                                 transforms.RandomErasing(),
                                                 MinMaxNormalizeImage()
                                                 ])


    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(resize_size),
                                         Wiener(),
                                         MinMaxNormalizeImage(),
                                         ])

    return train_transform_weak, train_transform_strong, test_transform



def get_dataloaders(root="./data", batch_size_source=32, workers=2):

    train_transform_weak, train_transform_strong, test_transform = _get_transforms()


    BATCH_SIZE_source = batch_size_source
    BATCH_SIZE_target = 3 * BATCH_SIZE_source
    
    
    USE_CASHED = True


    #* source datasets
    source_dataset_train_weak = _LensData( 
                                transform=train_transform_weak,
                                root=root,
                                datatype="easy",
                                isLabeled=True,
                                use_cached = USE_CASHED,
                                permute = False
                                )
    

    source_dataset_train_strong = _LensData(
                                transform=train_transform_strong,
                                root=root,
                                datatype="easy",
                                isLabeled=True,
                                use_cached = USE_CASHED,
                                permute = False
                                )


    #* target datasets
    target_dataset_train_weak_labeled = _LensData(
                                transform=train_transform_weak,
                                root=root,
                                datatype="hard",
                                isLabeled=True,
                                use_cached = USE_CASHED,
                                permute = False
                                )
    
    target_dataset_train_strong_labeled = _LensData(
                                transform=train_transform_strong,
                                root=root,
                                datatype="hard",
                                isLabeled=True,
                                use_cached = USE_CASHED,
                                permute = False
                                )
    

    target_dataset_train_weak_unlabeled = _LensData(
                                transform=train_transform_weak,
                                root=root,
                                datatype="hard",
                                isLabeled=False,
                                use_cached = USE_CASHED,
                                permute = False
                                )
    

    target_dataset_train_strong_unlabeled = _LensData(
                                transform=train_transform_strong,
                                root=root,
                                datatype="hard",
                                isLabeled=False,
                                use_cached = USE_CASHED,
                                permute = False
                                )


    target_dataset_test = _LensData(
                                transform=test_transform,
                                root=root,
                                datatype="hard",
                                isLabeled=False,
                                isTest=True,
                                use_cached = USE_CASHED,
                                permute = False
                                )
    

    #* concatenate source and labeled target datasets
    source_dataset_train_weak = ConcatDataset([source_dataset_train_weak, target_dataset_train_weak_labeled])
    source_dataset_train_strong = ConcatDataset([source_dataset_train_strong, target_dataset_train_strong_labeled])


    #* all dataloaders
    source_dataloader_train_weak = DataLoader(source_dataset_train_weak, shuffle=False, batch_size=BATCH_SIZE_source, num_workers=workers)
    source_dataloader_train_strong = DataLoader(source_dataset_train_strong, shuffle=False, batch_size=BATCH_SIZE_source, num_workers=workers)

    target_dataloader_train_weak = DataLoader(target_dataset_train_weak_unlabeled, shuffle=False, batch_size=BATCH_SIZE_target, num_workers=workers)
    target_dataloader_train_strong = DataLoader(target_dataset_train_strong_unlabeled, shuffle=False, batch_size=BATCH_SIZE_target, num_workers=workers)

    target_dataloader_test = DataLoader(target_dataset_test, shuffle=False, batch_size=BATCH_SIZE_target, num_workers=workers)

    return (source_dataloader_train_weak, source_dataloader_train_strong), (target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test)