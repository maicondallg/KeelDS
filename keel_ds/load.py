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
            return pickle.load(open(os.path.join(BASE_DIR, f"data/imbalanced/processed/{data}.pkl"), "rb"))
        except:
            raise FileNotFoundError(f"File {data}.pkl not found")

    else:
        try:
            return pickle.load(open(os.path.join(BASE_DIR, f"data/balanced/processed/{data}.pkl"), "rb"))
        except:
            raise FileNotFoundError(f"File {data}.pkl not found")
