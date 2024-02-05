"""Multi Layer Deep Neural Network
"""
from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Tuple, Union

import keras

from deep_tensors.models.schemas import DenseChecker


Layer = Union[
    keras.layers.Dense,
    keras.layers.Conv2D,
]
Layers = List[Layer]
InputShape = Tuple[Optional[int], ...]
Activation = Literal[
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
]


@dataclass
class DenseParameters:
    "Parameters for dense layers"

    units: int
    activation: Activation
    name: str

    def __post_init__(self):
        DenseChecker(**asdict(self))

    def get_layer(self) -> keras.layers.Dense:
        "Get the params of the layer"
        return keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            name=self.name,
        )


class NeuralModel(keras.Model):
    "Neural Model"

    def __init__(self, hiddens: Layers):
        super().__init__()
        self.hiddens = hiddens

    def call(
        self,
        inputs: keras.layers.Input,
        training: Optional[bool] = None,
        mask: Optional[List[bool]] = None,
    ):
        "call method"
        hidden = self.hiddens[0](inputs)
        for layer in self.hiddens[1:]:
            hidden = layer(hidden)

        return hidden
