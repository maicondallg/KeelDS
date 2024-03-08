# KeelDS


## KeelDS is a Python library for data scientists, data engineers, and software developers who are interested in building end-to-end machine learning pipelines. It is designed to be simple and easy to use, and it is built on top of popular libraries such as Pandas, Scikit-learn, and TensorFlow. KeelDS provides a high-level API for building machine learning pipelines, and it takes care of many of the low-level details such as data preprocessing, feature engineering, model training, and model evaluation. KeelDS is designed to be flexible and extensible, and it provides a number of built-in components that can be easily customized and extended to meet the needs of different use cases. KeelDS is also designed to be scalable, and it provides support for distributed computing and parallel processing. KeelDS is open source and it is available under the Apache 2.0 license.

### Installation
----------------

Dependencies
~~~~~~~~~~~~

- Python (>= 3.6)
- Pandas (>= 1.0.0)

You can install KeelDS using pip:

```bash
pip install keel-ds
```

### Usage
```bash

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
    
print(np.mean(evaluations)) # Output = 0,.933333333333

```