import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from mdlp.discretization import MDLP
# Package from https://github.com/hlin117/mdlp-discretization


def discretizer(x_train, y_train, x_test, y_test, colunas_discretizaveis):
    disct = MDLP(random_state=1306, min_depth=1)
    le = LabelEncoder()

    x_train_discr = x_train.loc[:, colunas_discretizaveis]
    x_test_discr = x_test.loc[:, colunas_discretizaveis]

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    x_train_disc = disct.fit_transform(x_train_discr, y_train)
    x_test_disc = disct.transform(x_test_discr)

    x_train.loc[:, colunas_discretizaveis] = x_train_disc
    x_test.loc[:, colunas_discretizaveis] = x_test_disc

    return x_train, y_train, x_test, y_test


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a).astype("str")


def split_disc(data, train_index, test_index, colunas_discretizaveis):
    X_train = data.iloc[train_index, :-1]
    y_train = data.iloc[train_index, -1]

    X_test = data.iloc[test_index, :-1]
    y_test = data.iloc[test_index, -1]

    if colunas_discretizaveis != "":
        X_train_disc, y_train, X_test_disc, y_test = discretizer(
            X_train, y_train, X_test, y_test, colunas_discretizaveis
        )
        return X_train_disc, y_train, X_test_disc, y_test

    else:
        return X_train, y_train, X_test, y_test


def discretize_dataset(dataset, att_to_discretize):
    data = pd.read_csv("../raw/" + dataset + ".dat", header=None)

    stratifier = StratifiedKFold(n_splits=2, random_state=1306, shuffle=True)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    data_folds = []

    i = 0
    for train_index, test_index in stratifier.split(X, y):
        print(i)
        i += 1
        X_train_disc, y_train, X_test_disc, y_test = split_disc(
            data, train_index, test_index, att_to_discretize[dataset]["D"]
        )

        X_train_disc["Classe"] = y_train
        X_test_disc["Classe"] = y_test

        X_train_disc = X_train_disc.astype("int", errors="ignore").astype("str")
        X_test_disc = X_test_disc.astype("int", errors="ignore").astype("str")

        treino = X_train_disc.to_numpy()
        teste = X_test_disc.to_numpy()

        for coluna in range(treino.shape[1]):
            treino_itens = list(np.unique(treino[:, coluna]))
            treino_itens.extend(np.unique(teste[:, coluna]))

            map_itens = {x[1]: x[0] for x in enumerate(np.unique(treino_itens))}

            treino[:, coluna] = vec_translate(treino[:, coluna], map_itens)
            teste[:, coluna] = vec_translate(teste[:, coluna], map_itens)

        keep_columns = np.where(
            [len(np.unique(treino[:, c])) > 1 for c in range(treino.shape[1])]
        )[0]

        treino = treino[:, keep_columns]
        teste = teste[:, keep_columns]

        data_folds.append([treino, teste])

    pickle.dump(data_folds, open("../processed/" + dataset + ".pkl", "wb"))

def get_info_datasets():
    datas = {}

    for dataset in datasets:
        data = pd.read_csv(f"../raw/{dataset[:-4]}.dat", header=None)
        data_disc = pickle.load(open(f"../processed/{dataset}", "rb"))

        freq_classe = data.iloc[:, -1].value_counts().values

        aux = {
            "#T": len(data),
            "#F": len(data.columns),
            "#CL": len(data.iloc[:, -1].unique()),
            "#IR": "1:" + str(round(max(freq_classe) / min(freq_classe), 1)),
        }

        media_itens_unicos = []
        for treino, teste in data_disc:
            itens_unicos = 0
            for coluna in range(treino.shape[1] - 1):
                conjunto_treino = list(np.unique(treino[:, coluna]))
                conjunto_teste = list(np.unique(teste[:, coluna]))
                conjunto_treino.extend(conjunto_teste)
                conjunto = set(conjunto_treino)

                itens_unicos += len(conjunto)
            media_itens_unicos.append(itens_unicos)

        aux["#DI"] = round(np.mean(media_itens_unicos))

        datas[dataset[:-4].capitalize()] = aux

    datas = pd.DataFrame(datas)
    datas = datas.T
    datas.insert(0, "Datasets", datas.index.values)
    datas.index = list(range(1, len(datas) + 1))
    datas[["Datasets", "#T", "#F", "#DI", "#CL", "#IR"]].to_csv(
        "../data/InfoForPaper.csv"
    )


class Dataset:
    def __init__(self, name, path, balance='balanced', randon_state=1306, max_unique_value=10):
        self.data_folds = None
        self.name = name
        self.path = path
        self.data = pd.read_csv(path, header=None)
        self.max_unique_value = max_unique_value
        self.random_state = randon_state
        self.balance = balance

        self.att_to_discretize = {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def set_attributes_to_discretize(self):
        """
        Set the attributes to discretize with the MDLP algorithm
        Attributes with less than 10 unique values are not discretized

        :return:
        """
        attribute_to_discretize = dict()

        for column in range(self.data.shape[1] - 1):
            if self.data[column].dtype == "int64":
                if len(np.unique(self.data[column])) >= self.max_unique_value:
                    self._set_att_to_discretize(column, 'int64')

            elif self.data[column].dtype == "float64":
                self._set_att_to_discretize(column, self.data[column].dtype)

    def _set_att_to_discretize(self, column, dtype):
        try:
            self.att_to_discretize[dtype].append(column)
        except:
            self.att_to_discretize[dtype] = [column]

    def get_attributes_to_discretize(self):
        attributes = []

        for key in self.att_to_discretize.keys():
            attributes.extend(self.att_to_discretize[key])

        return attributes

    def split(self, k_folds):
        strat = StratifiedKFold(n_splits=k_folds, random_state=self.random_state, shuffle=True)
        x, y = self.data.iloc[:, :-1], self.data.iloc[:, -1]

        data_folds = []

        for train_index, test_index in strat.split(x, y):
            x_train = self.data.iloc[train_index, :-1].to_numpy()
            y_train = self.data.iloc[train_index, -1].to_numpy()

            x_test = self.data.iloc[test_index, :-1].to_numpy()
            y_test = self.data.iloc[test_index, -1].to_numpy()

            data_folds.append([x_train, y_train, x_test, y_test])

        self.data_folds = data_folds

    def _transform_to_sequencial(self, x_train, y_train, x_test, y_test):
        for column in range(x_train.shape[1]):
            unique_values = list(np.unique(x_train[:, column]))
            unique_values.extend(np.unique(x_test[:, column]))

            map_itens = {x[1]: x[0] for x in enumerate(np.unique(unique_values))}

            x_train[:, column] = vec_translate(x_train[:, column], map_itens)
            x_test[:, column] = vec_translate(x_test[:, column], map_itens)

        keep_columns = np.where(
            [len(np.unique(x_train[:, c])) > 1 for c in range(x_train.shape[1])]
        )[0]

        x_train = x_train[:, keep_columns]
        x_test = x_test[:, keep_columns]

        x_train = x_train.astype("int").astype("str")
        x_test = x_test.astype("int", ).astype("str")
        y_train = y_train.astype("int").astype("str")
        y_test = y_test.astype("int").astype("str")

        return x_train, y_train, x_test, y_test


    def discretize(self):
        """
        Discretize the dataset using the MDLP algorithm in the attributes set to discretize
        :return:
        """

        # Check if doesn't have any attribute to discretize
        if not self.get_attributes_to_discretize():
            for i, fold in enumerate(self.data_folds):

                x_train, y_train, x_test, y_test = fold

                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

                x_train, y_train, x_test, y_test = self._transform_to_sequencial(x_train, y_train, x_test, y_test)
                self.data_folds[i] = [x_train, y_train, x_test, y_test]

        else:
            for i, fold in enumerate(self.data_folds):

                x_train, y_train, x_test, y_test = fold

                disct = MDLP(random_state=self.random_state, min_depth=1)
                le = LabelEncoder()

                x_train_discr = x_train[:, self.get_attributes_to_discretize()]
                x_test_discr = x_test[:, self.get_attributes_to_discretize()]

                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

                x_train_disc = disct.fit_transform(x_train_discr, y_train)
                x_test_disc = disct.transform(x_test_discr)

                x_train[:, self.get_attributes_to_discretize()] = x_train_disc
                x_test[:, self.get_attributes_to_discretize()] = x_test_disc

                x_train, y_train, x_test, y_test = self._transform_to_sequencial(x_train, y_train, x_test, y_test)
                self.data_folds[i] = [x_train, y_train, x_test, y_test]

    def save(self):
        save = {}
        for i, (x_train, y_train, x_test, y_test) in enumerate(self.data_folds):
            save.update({f'fold{i}-x_train': x_train,
                         f'fold{i}-y_train': y_train,
                         f'fold{i}-x_test': x_test,
                         f'fold{i}-y_test': y_test})

        np.savez_compressed(file=f'keel_ds/data/{self.balance}/processed/{self.name}.npz', **save)
        # pickle.dump(self.data_folds, open(f"keel_ds/data/imbalanced/processed/{self.name}.pkl", "wb"))


    def _get_ir(self):
        """
        Return the Imbalance Ratio of the dataset
        :return:
        """
        classe, count = np.unique(self.data.iloc[:, -1], return_counts=True)
        return np.max(count)/np.min(count)

    def get_info(self):
        aux = {
            "Dataset": self.name,
            "#T": len(self.data),
            "#F": self.data.shape[1],
            "#CL": len(np.unique(self.data.iloc[:, -1])),
            "#IR": "1:" + str(round(self._get_ir(), 1)),
        }

        media_itens_unicos = []
        for x_train, y_train, x_test, y_test in self.data_folds:
            itens_unicos = 0
            for coluna in range(x_train.shape[1]):
                conjunto_treino = list(np.unique(x_train[:, coluna]))
                conjunto_teste = list(np.unique(x_test[:, coluna]))
                conjunto_treino.extend(conjunto_teste)
                conjunto = set(conjunto_treino)

                itens_unicos += len(conjunto)
            media_itens_unicos.append(itens_unicos)

        aux["#DI"] = round(np.mean(media_itens_unicos))

        return aux


if __name__ == "__main__":

    balance = 'balanced'
    k_folds = 10

    datasets = sorted(os.listdir(f"keel_ds/data/{balance}/raw/"))

    info = []

    # datasets = ['car-good.dat']
    for dataset in datasets:
        # try:
        ds = Dataset(dataset[:-4], f"keel_ds/data/{balance}/raw/" + dataset, balance=balance)
        ds.set_attributes_to_discretize()
        ds.split(k_folds=k_folds)
        ds.discretize()

        info.append(ds.get_info())
        ds.save()

        # except:
        #     print(dataset)

    # info = pd.DataFrame(info)
    # info.to_csv('InfoImbalanced.csv')
