from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from sklearn.model_selection import train_test_split
import torch

import sys
sys.path.append("./")
from utils import load_data




class _MinMaxNormalizeImage:
    def __call__(self, img: torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        normalized_tensor = (img - min_val) / (max_val - min_val)
        return normalized_tensor
        



class _LensData(Dataset):
    def __init__(self, transform, *args, **kwargs) -> None:
        ROOT = "./data"
        self.X, self.y = load_data(root=ROOT, *args, **kwargs)

        self.transform = transform


    def __len__(self):
        return len(self.y)


    def __getitem__(self, i):
        img = self.X[i]

        labels = self.y[i]

        if self.transform is not None:
            img = self.transform(img)

        return img, labels



class Lens:

    def __init__(self, batch_size=1, num_workers=1, crop_size=101, img_size=101, rotation_degrees=0, translate=(0.0, 0.0), scale=(1.0, 1.0), *, class_samples):

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.crop_size = crop_size
        self.img_size = img_size
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale

        self.class_samples = class_samples


    def __call__(self):
        # possible batch sizes: 5, 10, 13, 20, 26, 41, 52, 65, 82, 130, 164
        train_loader = DataLoader(
            _LensData(
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(self.crop_size),
                    transforms.Resize(self.img_size),
                    transforms.RandomAffine(
                        degrees=self.rotation, 
                        translate=self.translate,
                        scale=self.scale
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    _MinMaxNormalizeImage(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]),
                datatype="easy",
                use_cached = True,
                permute = False
            ),
            
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )


        test_loader = DataLoader(
            _LensData(
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(self.crop_size),
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    _MinMaxNormalizeImage(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]),
                datatype="hard",
                use_cached = True,
                permute = False
            ),
            
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, test_loader, self.img_size