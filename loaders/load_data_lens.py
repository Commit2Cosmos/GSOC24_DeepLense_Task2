from torch.utils.data import Dataset
import numpy as np
import os
import warnings
from utils import load_data, _permute



def array_split(arr: np.array, train_size: float = 0.8):
    split_index = int(len(arr) * train_size)
    arr1, arr2 = np.array_split(arr, [split_index])
    
    return arr1, arr2


class _LensData(Dataset):
    def __init__(self, transform, root="./data", datatype="easy", isLabeled=True, isTest=False, use_cached=True, permute=False) -> None:
        self.transform = transform

        # SAMPLES = 120

        if datatype == "easy":
            isLabeled = True

        if use_cached:
            try:
                if isLabeled:
                    self.X = np.load(os.path.join(root, f'X_{datatype}.npy'))
                    self.y = np.load(os.path.join(root, f'Y_{datatype}.npy')).astype(np.float32)
                else:
                    self.X = np.load(os.path.join(root, f'X_unlabeled.npy'))
                    self.y = np.zeros((self.X.shape[0]), dtype=np.float32)

                    if not isTest:
                        self.X, _ = array_split(self.X)
                    else:
                        _, self.X = array_split(self.X)


                if permute:
                    self.X, self.y = _permute(self.X, self.y)


                #! FOR TESTING ONLY
                # self.X, self.y = self.X[:SAMPLES], self.y[:SAMPLES]

                self.X = self.X.transpose((0,2,3,1))
                print('Cached data found!')
                # print("X cached: ", np.max(self.X, axis=(1,2,3)))
                # print("X cached: ", np.min(self.X, axis=(1,2,3)))
                # print("y cached: ", self.y)

                return
                

            except FileNotFoundError as _:
                warnings.warn('Cached data does not exist! Loading from raw data.')


        if isLabeled:
            self.X, self.y, _ = load_data(root, datatype)

        else:
            _, _, self.X = load_data(root, datatype)
            self.y = np.zeros((self.X.shape[0]), dtype=np.int64)
            
            #* train/test split
            if not isTest:
                self.X, _ = array_split(self.X)
            else:
                _, self.X = array_split(self.X)


        if permute:
            self.X, self.y = _permute(self.X, self.y)

        #! FOR TESTING ONLY
        # self.X, self.y = self.X[:SAMPLES], self.y[:SAMPLES]


        self.X = self.X.transpose((0,2,3,1))
        # print("X shape uncached: ", self.X.shape)



    def __len__(self):
        return len(self.X)


    def __getitem__(self, i):
        img = self.X[i]
        labels = None

        if hasattr(self, 'y'):
            labels = self.y[i]

        if self.transform is not None:
            img = self.transform(img)

        return img, labels