# KeelDS


## KeelDS: A Python package for loading datasets from KEEL repository

KEEL [https://sci2s.ugr.es/keel/datasets.php] is a popular repository for machine learning datasets. This package provides a simple way to load datasets from the KEEL repository. The package also provides a simple way to load the datasets in a cross-validation setting.

This is a simple package that provides a simple way to load datasets from the KEEL repository. The package also provides a simple way to load the datasets in a cross-validation setting.

### Features

- Load datasets available in the KEEL repository with a single line of code.
- Load datasets already split into train and test sets and with discretization (with Fayyad algorithm - MDLP[https://github.com/hlin117/mdlp-discretization]) applied.

### Installation
----------------

Dependencies

- Python (>= 3.6)
- Pandas (>= 1.0.0)

You can install KeelDS using pip:

```bash
pip install keel-ds
```

### Usage


```python
from keel_ds import load_data
import numpy as np
from catboost import CatBoostClassifier

file_name = 'iris'
folds = load_data(file_name)

evaluations = []
for x_train, y_train, x_test, y_test in folds:
    model = CatBoostClassifier(verbose=False)
    model.fit(x_train, y_train)
    evaluation = model.score(x_test, y_test)
    evaluations.append(evaluation)
    
print(np.mean(evaluations)) # Output = 0.933333333333

```