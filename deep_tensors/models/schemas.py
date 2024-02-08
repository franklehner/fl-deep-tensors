"""Check schemas of models
"""
from typing import List, Literal, Tuple

from pydantic import BaseModel, PositiveInt, validator, field_validator


class DenseChecker(BaseModel):
    "Check Dense Parameters"
    units: PositiveInt
    activation: Literal["relu", "sigmoid", "tanh", "softmax"]
    name: str


class Hiddens(BaseModel):
    "Check Hidden Layers"
    hiddens: List[Tuple[PositiveInt, str, str]]


    @validator("hiddens")
    @classmethod
    def validate_hiddens(cls, value):
        "validate hiddens"
        for v in value:
            if v[1] not in ["relu", "sigmoid", "tanh", "softmax"]:
                raise ValueError(f"\n\nUnkown activation {v[1]}\n\n")

        return value


class BaseModelCompiler(BaseModel):
    "Validation class for model compiler"
    optimizer: Literal["adam", "rmsprop", "sgd"]
    loss: Literal["binary_crossentropy", "categorical_crossentropy", "mae", "mse"]
    metrics: List[Literal["accuracy", "binary_accuracy", "f1", "r2"]]


class MnistPrepareData(BaseModel):
    "Check MnistData class"
    buffer_size: PositiveInt
    batch_size: PositiveInt
    valid_size: PositiveInt

    @field_validator("valid_size")
    @classmethod
    def check_valid_size(cls, value):
        "check the length of the valid field"
        if value >= 60000:
            raise ValueError("The size of the valid_data must not be bigger than train size")

        return value

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, value):
        "Check the length of the batch_size"
        if value >= 60000:
            raise ValueError("The batch_size must not be bigger than train_size")

        return value


class TensorflowDataSet(BaseModel):
    "Check Tensorflow DataSet"
    name: Literal["mnist"]
