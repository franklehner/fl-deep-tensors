"""Usecase for car mpg regression
"""
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from deep_tensors.app import prepare_car_dataset


TrainTest = Tuple[pd.DataFrame, pd.DataFrame]
Regressor = Literal["DNN", "Tree"]


@dataclass
class CarRegressor:
    "Regressor for cars"

    hidden_units: List[int]
    model_dir: str
    epochs: int = 1000
    batch_size: int = 8
    regressor_str: Regressor = "DNN"

    def get_datasets(self) -> TrainTest:
        "Get car datasets"
        cars = prepare_car_dataset.load_cars()
        df_train, df_test = prepare_car_dataset.split_cars(df=cars)
        df_train_norm, df_test_norm = prepare_car_dataset.normalize_numeric_columns(
            train=df_train, test=df_test,
        )

        return df_train_norm, df_test_norm

    def build_dnn_regressor(self, warm_start: bool = False) -> tf.estimator.DNNRegressor:
        "Build dnn regressor"
        all_feature_columns = prepare_car_dataset.get_all_feature_columns()
        if warm_start:
            regressor = tf.estimator.DNNRegressor(
                hidden_units=self.hidden_units,
                feature_columns=all_feature_columns,
                warm_start_from=self.model_dir,
                model_dir=self.model_dir,
            )
        else:
            regressor = tf.estimator.DNNRegressor(
                hidden_units=self.hidden_units,
                feature_columns=all_feature_columns,
                model_dir=self.model_dir,
            )

        return regressor

    def train(self) -> None:
        "Train and evaluate estimator"
        df_train, _ = self.get_datasets()
        total_steps = self.epochs * int(np.ceil(len(df_train) / self.batch_size))
        match self.regressor_str:
            case "DNN":
                regressor = self.build_dnn_regressor()

            case _:
                raise ValueError(f"No valid regressor {self.regressor_str}")

        regressor.train(
            input_fn=lambda:prepare_car_dataset.train_input_fn(
                df_train=df_train,
                batch_size=self.batch_size,
            ),
            steps=total_steps,
        )

    def evaluate(self) -> None:
        "Evaluate"
        _, df_test = self.get_datasets()
        match self.regressor_str:
            case "DNN":
                reloaded_regressor = self.build_dnn_regressor(warm_start=True)
            case _:
                raise ValueError(f"No valid regressor {self.regressor_str}")

        eval_results = reloaded_regressor.evaluate(
            input_fn=lambda: prepare_car_dataset.eval_input_fn(
                df_test=df_test,
                batch_size=self.batch_size,
            ),
        )
        average_loss = eval_results["average_loss"]
        print(f"Average loss: {average_loss}")
