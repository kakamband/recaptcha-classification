import os
import glob

import numpy as np
import pandas as pd


def one_hot_encoder(labels, data):
    sparse_cat = np.arange(start=0, stop=len(labels), step=1)
    ohe_dict = {c: i for c, i in zip(labels, sparse_cat)}
    ohe_labeled = np.zeros((len(data), len(labels)), dtype=np.uint8)
    sparse_labeled = np.zeros((len(data), 1), dtype=np.uint8)
    for index, file_name in enumerate(data):
        ohe_labeled[index, ohe_dict[file_name.split('/')[1]]] = 1
        sparse_labeled[index] = ohe_dict[file_name.split('/')[1]]

    return ohe_labeled, sparse_labeled


if __name__ == '__main__':
    fn_train = sorted(glob.glob(os.path.join('train', '**/*.*'), recursive=True))
    fn_eval = sorted(glob.glob(os.path.join('eval', '**/*.*'), recursive=True))
    classes = sorted(os.listdir('train'))

    ohe_encoded, sparse_encoded = one_hot_encoder(classes, fn_train)


