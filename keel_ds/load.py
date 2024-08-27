import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def list_data(type_data='balanced'):
    if type_data not in ['balanced', 'imbalanced']:
        raise ValueError("type_data must be 'balanced' or 'imbalanced'")

    if type_data == 'balanced':
        path = os.path.join(BASE_DIR, "data/balanced/processed")

    else:
        path = os.path.join(BASE_DIR, "data/imbalanced/processed")

    return sorted([x[:-4] for x in os.listdir(path)])  # remove .npz extension


def load_data(data, type_data='balanced', raw=False):

    if type_data not in ['balanced', 'imbalanced']:
        raise ValueError("type_data must be 'balanced' or 'imbalanced'")

    try:
        if not raw:
            npz = np.load(os.path.join(BASE_DIR, f"data/{type_data}/processed/{data}.npz"))
            dataset = []
            for fold in range(0, len(npz), 4):
                x_train, y_train, x_test, y_test = npz[npz.files[fold]], npz[npz.files[fold + 1]], npz[
                    npz.files[fold + 2]], \
                    npz[npz.files[fold + 3]]
                dataset.append((x_train, y_train, x_test, y_test))

            return dataset

        else:
            return pd.read_csv(os.path.join(BASE_DIR, f"data/{type_data}/raw/{data}.dat"), header=None)

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {data} not found")


def list_presets():
    return sorted([x[:-4] for x in os.listdir(os.path.join(BASE_DIR, "data/presets"))])


def load_preset(preset_name='preset_for_apriori_10k'):
    try:
        return pickle.load(open(os.path.join(BASE_DIR, f"data/presets/{preset_name}.pkl"), "rb"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Preset {preset_name} not found")
