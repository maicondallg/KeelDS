from keel_ds import load_data

if __name__ == '__main__':

    file_name = 'vehicle1'

    folds = load_data(file_name, imbalanced=True) # Load the imbalanced dataset with 10 folds

    x_train, y_train, x_test, y_test = folds[0] # 0 is the first fold
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (546, 18) (546,) (61, 18) (61,)