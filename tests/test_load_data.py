

def test_load_data():
    """
    Test the load_data function from keel_ds.load
    :return:
    """

    from keel_ds.load import load_data

    # Test if the function raises an error when the file is not found
    # try:
    #     load_data("not_found")
    # except FileNotFoundError as e:
    #     assert str(e) == "File not_found.pkl not found"

    # Test if the function returns a pandas DataFrame
    df = load_data("iris")
    assert len(df) == 10