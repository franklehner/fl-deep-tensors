"""Test cases for preparing car dataset
"""
import decimal
import pytest

from deep_tensors.app import prepare_car_dataset as pcd


def test_load_cars():
    "Test load car dataset"
    data = pcd.load_cars()

    assert isinstance(data, pcd.pd.DataFrame)
    assert not data.empty


@pytest.mark.parametrize(
    "test_size", [
        0.1,
        0.2,
    ],
)
def test_split_cars(test_size):
    "Test splitting data into train and test sizes"
    df = pcd.load_cars()
    df_train, df_test = pcd.split_cars(df=df, test_size=test_size)
    ratio = len(df_test) / len(df_train)
    expected_split = float(
        decimal.Decimal(str(ratio)).quantize(
            decimal.Decimal(".1"),
            rounding=decimal.ROUND_FLOOR,
        ),
    )

    assert expected_split == test_size


def test_normalize_numeric_columns():
    "Test case for normalizing numeric columns"
    df = pcd.load_cars()
    df_train, df_test = pcd.split_cars(df=df, test_size=0.2)
    df_train_norm, df_test_norm = pcd.normalize_numeric_columns(
        train=df_train,
        test=df_test,
    )
    numeric_columns = pcd.cars_mpg.get_numeric_columns()

    for column in numeric_columns:
        assert round(df_train_norm[column].mean()) == 0
        assert round(df_test_norm[column].mean()) == 0
        assert round(df_train_norm[column].std()) == 1
        assert round(df_test_norm[column].std()) == 1


def test_get_numeric_feature_list():
    "Get numeric feature list"
    numeric_features = pcd.get_numeric_feature_list()
    numeric_columns = pcd.cars_mpg.get_numeric_columns()

    assert isinstance(numeric_features, list)
    assert len(numeric_columns) == len(numeric_features)
    for feature in numeric_features:
        assert feature.key in numeric_columns


def test_get_bucket_feature_list():
    "Test case for bucket feature list"
    bucket_features = pcd.get_bucket_feature_list()
    bucket_columns = pcd.cars_mpg.get_interval_columns()

    assert isinstance(bucket_features, list)
    assert len(bucket_columns) == len(bucket_features)

    for feature in bucket_features:
        assert feature.source_column.key in bucket_columns


def test_get_origin_feature_list():
    "Test case for origin feature list"
    origin_features = pcd.get_origin_feature_list()
    origi_columns = pcd.cars_mpg.get_categorical_columns()

    assert isinstance(origin_features, list)
    assert len(origi_columns) == len(origin_features)

    for feature in origin_features:
        assert feature.categorical_column.key in origi_columns


def test_train_input_fn():
    "Test case for train_input_fn"
    df = pcd.load_cars()
    df_train, df_test = pcd.split_cars(df=df)
    df_train_norm, _ = pcd.normalize_numeric_columns(
        train=df_train,
        test=df_test,
    )
    dataset = pcd.train_input_fn(df_train=df_train_norm)

    batch = next(iter(dataset))
    assert isinstance(batch[0], dict)
    assert len(batch[0]) == len(df.columns) - 1
    assert batch[0]["cylinders"].numpy().shape[0] == 8


def test_eval_input_fn():
    "Test case for eval_input_fn"
    df = pcd.load_cars()
    df_train, df_test = pcd.split_cars(df=df)
    _, df_test_norm = pcd.normalize_numeric_columns(
        train=df_train,
        test=df_test,
    )
    dataset = pcd.eval_input_fn(
        df_test=df_test_norm,
    )
    batch = next(iter(dataset))
    assert isinstance(batch[0], dict)
    assert len(batch[0]) == len(df.columns) - 1
    assert batch[0]["cylinders"].numpy().shape[0] == 8


def test_get_all_feature_columns():
    "Test case for get_all_feature_columns"
    all_features = pcd.get_all_feature_columns()

    assert len(all_features) == 7
