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

### API Reference

#### `load_data(data, imbalanced=False, raw=False)`

Load a dataset from the KEEL repository.

- `data` (str): Name of the dataset to load
- `imbalanced` (bool): If True, load from imbalanced datasets. Default is False.
- `raw` (bool): If True, return the raw dataset. Default is False.

Returns a list of tuples (x_train, y_train, x_test, y_test) for each fold.

#### `list_data()`

List all available datasets.

Returns a dictionary with two keys: 'balanced' and 'imbalanced', each containing a list of available dataset names.

### Contributing

Contributions to KeelDS are welcome! Please feel free to submit a Pull Request.

### License

[Add license information here]

### Contact

For any queries or issues, please open an issue on the [GitHub repository](https://github.com/maicondallg/KeelDS).
```

This updated README provides a more comprehensive overview of the KeelDS package, including:

1. A clearer introduction and feature list
2. Updated installation instructions
3. A more detailed usage example
4. API reference for the main functions
5. Information about contributing and contact

You may want to add more sections or details based on your specific needs, such as a more detailed API reference, troubleshooting tips, or information about the dataset preprocessing steps. Also, don't forget to add the appropriate license information.