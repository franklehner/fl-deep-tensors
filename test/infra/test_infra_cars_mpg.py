"""Test cases for cars
"""
from deep_tensors.infra import cars_mpg


def test_get_numeric_columns():
    "Test output of get_numeric_columns"
    expected_columns = [
        "cylinders",
        "cubic",
        "horsepower",
        "weight",
        "acceleration",
    ]
    columns = cars_mpg.get_numeric_columns()

    for column in columns:
        assert column in expected_columns


def test_get_interval_columns():
    "Test output of get_interval_columns"
    expected_columns = [
        "model_year",
    ]

    columns = cars_mpg.get_interval_columns()

    for column in columns:
        assert column in expected_columns


def test_get_categorical_columns():
    "Test output of get_categorical_columns"
    expected_columns = [
        "origin",
    ]

    columns = cars_mpg.get_categorical_columns()

    for column in columns:
        assert column in expected_columns


def test_load():
    "Test loading data"
    data = cars_mpg.load()
    numeric_columns = cars_mpg.get_numeric_columns()
    interval_columns = cars_mpg.get_interval_columns()
    categorical_columns = cars_mpg.get_categorical_columns()
    all_columns = numeric_columns + interval_columns + categorical_columns

    for column in all_columns:
        assert column in data.columns

    assert len(all_columns) == len(data.columns) - 1
    all_cols = set(all_columns)
    data_columns = set(data.columns.to_list())

    assert data_columns.difference(all_cols) == {"mpg"}
