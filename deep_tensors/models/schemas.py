"""Check schemas of models
"""
from typing import List, Literal, Tuple

from pydantic import BaseModel, PositiveInt, validator


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
