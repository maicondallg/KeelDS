import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def list_data():
    path1 = os.path.join(BASE_DIR, "data/imbalanced/raw")
    path2 = os.path.join(BASE_DIR, "data/balanced/raw")
    aux = os.listdir(path2)
    aux.append(os.listdir(path2))

    return aux


def load_data(data, imbalanced=False):
    if imbalanced:
        try:
            data = []
            npz = np.load(os.path.join(BASE_DIR, f"data/imbalanced/processed/{data}.npz"))

            for fold in range(0, len(npz), 4):
                x_train, y_train, x_test, y_test = npz[npz.files[fold]], npz[npz.files[fold+1]], npz[npz.files[fold+2]], \
                npz[npz.files[fold+3]]
            data.append((x_train, y_train, x_test, y_test))

            return data

        except:
            raise FileNotFoundError(f"File {data}.pkl not found")

    else:
        try:
            npz = np.load(os.path.join(BASE_DIR, f"data/balanced/processed/{data}.npz"))
            data = []
            for fold in range(0, len(npz), 4):
                x_train, y_train, x_test, y_test = npz[npz.files[fold]], npz[npz.files[fold+1]], npz[npz.files[fold+2]], \
                    npz[npz.files[fold+3]]
                data.append((x_train, y_train, x_test, y_test))

            return data
        except:
            raise FileNotFoundError(f"File {data}.pkl not found")
