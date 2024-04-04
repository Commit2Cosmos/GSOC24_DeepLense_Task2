from torch.utils.data import Dataset
import numpy as np
import os
import warnings
from utils import load_data, _permute
from sklearn.model_selection import train_test_split


class LensData(Dataset):
    def __init__(self, transform, root="./data", datatype="easy", isLabeled=True, isTest=False, use_cached=True, permute=False) -> None:
        self.transform = transform

        # SAMPLES = 120

        if datatype == "easy":
            isLabeled = True
            isTest = False

        if use_cached:
            try:
                if isLabeled:
                    self.X = np.load(os.path.join(root, f'X_{datatype}.npy'))
                    self.y = np.load(os.path.join(root, f'Y_{datatype}.npy')).astype(np.float32)

                    if datatype == "hard":
                        if not isTest:
                            self.X_, _, self.y_, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                        else:
                            _, self.X, _, self.y = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

                else:
                    self.X = np.load(os.path.join(root, f'X_unlabeled.npy'))
                    self.y = np.zeros((self.X.shape[0]), dtype=np.float32)


                if permute:
                    self.X, self.y = _permute(self.X, self.y)


                #! FOR TESTING ONLY
                # self.X, self.y = self.X[:SAMPLES], self.y[:SAMPLES]

                self.X = self.X.transpose((0,2,3,1))
                print('Cached data found!')

                return
                

            except FileNotFoundError as _:
                warnings.warn('Cached data does not exist! Loading from raw data.')


        X, y, X_unlabeled = load_data(root, datatype)
        
        if isLabeled:
            self.X, self.y = X, y

            #* train/test split
            if not isTest and datatype == "hard":
                self.X_test, _, self.y_test, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            else:
                _, self.X_test, _, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        else:
            self.X = X_unlabeled
            self.y = np.zeros((self.X.shape[0]), dtype=np.int64)
            


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