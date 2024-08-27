from keel_ds import load_data, list_data,
import numpy as np
from catboost import CatBoostClassifier


if __name__ == '__main__':
    file_name = 'iris'
    folds = load_data(file_name)

    evaluations = []
    for x_train, y_train, x_test, y_test in folds:
        model = CatBoostClassifier(verbose=False)
        model.fit(x_train, y_train)
        evaluation = model.score(x_test, y_test)
        evaluations.append(evaluation)

    print(np.mean(evaluations))
