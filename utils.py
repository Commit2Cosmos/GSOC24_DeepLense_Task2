import numpy as np
from astropy.io import fits
import numpy as np
import pandas as pd
import os
import re

from typing import List



def _permute(X, Y, seed=0):
    np.random.seed(seed)

    indices_l = np.arange(len(X))
    np.random.shuffle(indices_l)

    return X[indices_l], Y[indices_l]



#! Image dims: 3 x 101 x 101
def load_data(root, datatype):
    """
    Splits and saves source domain dataset OR labeled and unlabeled target domain datasets separately
    """


    # number: {data: ..., label: ...} OR {data: ...}
    data = {}

    labels_df = pd.read_csv(os.path.join(root, f"{datatype}_test.csv"))
    labels_df.set_index('ID', inplace=True)

    pattern = r'imageSDSS_([A-Z])-(\d+)\.fits'

    # faulty_numbers = []


    first = True
    for band in (f for f in os.scandir(os.path.join(root, datatype)) if f.is_dir()):
        for file in os.scandir(band):
            if file.is_file() and file.name.endswith(".fits"):
                hdul = fits.getdata(file).copy()
                hdul = _normilise_standardise(hdul)

                index_position = re.search(pattern, file.name)
                num = index_position.group(2)


                # if _is_faulty_image(hdul) and not num in faulty_numbers:
                #     faulty_numbers.append(num)


                if first:
                    if int(num) in labels_df.index:
                        l = labels_df.loc[int(num), "no_source"]
                        data[num] = {'label': l, 'data': []}
                    else:
                        data[num] = {'data': []}

                if not _is_faulty_image(hdul):
                    data[num]['data'].append(hdul)
                
                # elif not num in faulty_numbers:
                    # faulty_numbers.append(num)

        first = False



    #* dealing with faulty images
    dict_keys = list(data.keys())
    # irrecoverable = 0

    for inner_dict in dict_keys:
        # if inner_dict not in faulty_numbers:
        #     continue

        img = np.array(data[inner_dict]['data'])
        colours = img.shape[0]

        if colours < 3:
            del data[inner_dict]

        #* use partially faulty images
        # if colours == 0:
        #     irrecoverable += 1
        #     del data[inner_dict]
        # elif colours == 1:
        #     data[inner_dict]['data'] = np.broadcast_to(img, (3, img.shape[1], img.shape[2]))
        # elif colours == 2:
        #     data[inner_dict]['data'] = np.concatenate((img, np.mean(img, axis=0)[np.newaxis, :]), axis=0)
        

    X = np.array([data[inner_dict]['data'] for inner_dict in data.keys() if data[inner_dict].get('label') is not None])
    y = np.array([data[inner_dict]['label'] for inner_dict in data.keys() if data[inner_dict].get('label') is not None])
    X_unlabeled = None


    if datatype == "easy":
        assert len(X) == len(y)

    else:
        X_unlabeled = np.array([data[inner_dict]['data'] for inner_dict in data.keys() if data[inner_dict].get('label') is None])
        np.save(os.path.join(root, f'X_unlabeled.npy'), X_unlabeled)


    np.save(os.path.join(root, f'X_{datatype}.npy'), X)
    np.save(os.path.join(root, f'Y_{datatype}.npy'), y)
    
        
    #* select faulty images
    # X = np.array([data[inner_dict]['data'] for inner_dict in data.keys() if inner_dict in faulty_numbers])
    # y = np.array([data[inner_dict]['label'] for inner_dict in data.keys() if inner_dict in faulty_numbers])

    #* count unique labels
    # unique_values, counts = np.unique(y, return_counts=True)

    # print("Unique Values:", unique_values)
    # print("Counts:", counts)


    # print("total faulty images: ", len(faulty_numbers))
    # print(f"deleted {str(irrecoverable)} irrecoverables ")

    return X, y, X_unlabeled




def _normilise_standardise(array: np.ndarray, range_min = 0.0, range_max = 1.0):

    #* standardize
    array = (array - np.mean(array)) / np.std(array)

    #* normalize
    min_value = np.min(array)
    max_value = np.max(array)
    
    array = ((array - min_value) / (max_value - min_value)) * (range_max - range_min) + range_min

    return array



def _is_faulty_image(img: np.ndarray, threshold = 0.0005) -> bool:
    res1 = np.sum(np.where(img == 1, True, False))
    return res1/(img.shape[0]*img.shape[1]) > threshold



def apply_to_channels(array: np.ndarray, to_apply: List = [_normilise_standardise]):

    for channel in range(array.shape[0]):
        for fn in to_apply:
            array[channel] = fn(array[channel])

        
    return array