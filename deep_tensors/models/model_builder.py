"""Build models
"""
from dataclasses import dataclass
from typing import List, Tuple

from deep_tensors.models import neuralnet


@dataclass
class Builder:
    "Build model"

    hiddens: List[Tuple[int, neuralnet.Activation, str]]

    def build_model(self, input_shape: neuralnet.InputShape) -> neuralnet.NeuralModel:
        "build model"
        hiddens = [
            neuralnet.DenseParameters(
                units=hidden[0],
                activation=hidden[1],
                name=hidden[2],
            ).get_layer() for hidden in self.hiddens
        ]
        model = neuralnet.NeuralModel(hiddens=hiddens)
        model.build(input_shape=input_shape)

        return model
