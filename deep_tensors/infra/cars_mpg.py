"""Connection layer of the car mpg data
"""
from typing import List

import pandas as pd

DATASET_PATH = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLUMN_NAMES = [
    "mpg",
    "cylinders",
    "cubic",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]


def get_numeric_columns() -> List[str]:
    "Get the names of the numeric columns"
    return [
        "cylinders",
        "cubic",
        "horsepower",
        "weight",
        "acceleration",
    ]


def get_interval_columns() -> List[str]:
    "Get the names of the columns with intervals"
    return ["model_year"]


def get_categorical_columns() -> List[str]:
    "Get the names of the categorical columns"
    return ["origin"]


def load() -> pd.DataFrame:
    "Load car data"
    data = pd.read_csv(
        DATASET_PATH,
        sep=" ",
        names=COLUMN_NAMES,
        na_values="?",
        comment="\t",
        skipinitialspace=True,
    )
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data
