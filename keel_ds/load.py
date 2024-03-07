import pandas as pd
import numpy as np
import pickle
import os


def list_data():
    return os.listdir("../data/")


def load_data(data, imbalanced=True):
    if imbalanced:
        try:
            return pickle.load(open(f"../data/imbalanced/processed/{data}.pkl", "rb"))
        except:
            raise FileNotFoundError(f"File {data}.pkl not found")

    else:
        try:
            return pickle.load(open(f"../data/balanced/processed/{data}.pkl", "rb"))
        except:
            raise FileNotFoundError(f"File {data}.pkl not found")