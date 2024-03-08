from keel_ds import load_data


file_name = 'australian'

folds = load_data(file_name)

x_train, y_train, x_test, y_test = folds[0]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)