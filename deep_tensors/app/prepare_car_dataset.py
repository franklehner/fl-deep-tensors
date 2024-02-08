"""Prepare dataset for car mpg data
"""
from typing import List, Tuple, Union

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import python as tf_python  # pylint: disable=no-name-in-module

from deep_tensors.infra import cars_mpg

FrameSet = Tuple[pd.DataFrame, pd.DataFrame]
NumericColumns = List[tf_python.feature_column.feature_column_v2.NumericColumn]
BucketColumns = List[tf_python.feature_column.feature_column_v2.BucketizedColumn]
CategoricalColumns = List[tf_python.feature_column.feature_column_v2.CategoricalColumn]
AllColumns = List[Union[NumericColumns, BucketColumns, CategoricalColumns]]


def load_cars() -> pd.DataFrame:
    "Load data of cars"
    return cars_mpg.load()


def split_cars(df: pd.DataFrame, test_size: float = 0.2) -> FrameSet:
    "split cars"
    return train_test_split(
        df,
        test_size=test_size,
    )


def normalize_numeric_columns(train: pd.DataFrame, test: pd.DataFrame) -> FrameSet:
    "normalize the numeric columns of the car dataset"
    numeric_columns = cars_mpg.get_numeric_columns()
    train_stats = train.describe().transpose()
    train_norm, test_norm = train.copy(), test.copy()
    for col_name in numeric_columns:
        mean = train_stats.loc[col_name, "mean"]
        std = train_stats.loc[col_name, "std"]
        train_norm[col_name] = (train_norm[col_name] - mean) / std
        test_norm[col_name] = (test_norm[col_name] - mean) / std

    return train_norm, test_norm


def get_numeric_feature_list() -> NumericColumns:
    "get numeric feature list"
    numeric_features = []
    for col_name in cars_mpg.get_numeric_columns():
        numeric_features.append(
            tf.feature_column.numeric_column(key=col_name)
        )

    return numeric_features


def get_bucket_feature_list() -> BucketColumns:
    "Get features with intervals"
    bucket_sized_feature = []
    feature_year = tf.feature_column.numeric_column(key="model_year")
    bucket_sized_feature.append(
        tf.feature_column.bucketized_column(
            source_column=feature_year,
            boundaries=[73, 76, 79],
        ),
    )
    return bucket_sized_feature


def get_origin_feature_list() -> CategoricalColumns:
    "get origin feature list"
    categorical_indicator_features = []
    feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
        key="origin",
        vocabulary_list=[1, 2, 3],
    )
    categorical_indicator_features.append(
        tf.feature_column.indicator_column(feature_origin),
    )

    return categorical_indicator_features


def train_input_fn(df_train: pd.DataFrame, batch_size: int = 8) -> tf.data.Dataset:
    "Train input function"
    df = df_train.copy()
    train_x, train_y = df, df.pop("mpg")
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_x), train_y),
    )

    return dataset.shuffle(1000).repeat().batch(batch_size=batch_size)


def eval_input_fn(df_test: pd.DataFrame, batch_size: int = 8) -> tf.data.Dataset:
    "Function for eval input"
    df = df_test.copy()
    test_x, test_y = df, df.pop("mpg")
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(test_x), test_y),
    )

    return dataset.batch(batch_size=batch_size)


def get_all_feature_columns() -> AllColumns:
    "Get all feature columns"
    numeric_columns = get_numeric_feature_list()
    bucket_columns = get_bucket_feature_list()
    categorical_columns = get_origin_feature_list()
    all_list = numeric_columns + bucket_columns + categorical_columns

    return all_list
