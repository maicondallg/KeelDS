# KeelDS

## KeelDS: A Python package for loading datasets from KEEL repository

KeelDS is a Python package that provides easy access to datasets from the [KEEL repository](https://sci2s.ugr.es/keel/datasets.php), a popular source for machine learning datasets. This package simplifies the process of loading KEEL datasets, offering options for cross-validation and discretization.

### Features

- Load KEEL datasets with a single line of code
- Access datasets pre-split into train and test sets
- Discretization option using the Fayyad algorithm (MDLP)
- Support for both balanced and imbalanced datasets
- Easy integration with machine learning workflows

### Installation

#### Dependencies

- Python (>= 3.12)
- pandas (>= 2.2.2)

You can install KeelDS using pip:

```bash
pip install keel-ds
```

### Usage

Here's a simple example of how to use KeelDS with a machine learning model:

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

print(np.mean(evaluations))  # Output: 0.933333333333
```
