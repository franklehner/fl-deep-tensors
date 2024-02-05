"""Test cases for model builder
"""
import pytest
from pydantic_core._pydantic_core import ValidationError  # pylint: disable=no-name-in-module

from deep_tensors.models import model_builder as mb
from deep_tensors.models import neuralnet as nn


def test_initialize_model_builder():
    "Test the initializing of the model builder"
    hiddens = [
        (4, "relu", "hidden_1"),
        (4, "relu", "hidden_2"),
        (2, "softmax", "output"),
    ]
    builder = mb.Builder(hiddens=hiddens)

    assert isinstance(builder, mb.Builder)
    assert builder.hiddens == hiddens


def test_build_model():
    "Test building a model"
    hiddens = [
        (4, "relu", "hidden_1"),
        (4, "relu", "hidden_2"),
        (2, "softmax", "output"),
    ]
    builder = mb.Builder(hiddens=hiddens)
    input_shape = (None, 4)

    model = builder.build_model(input_shape=input_shape)

    assert isinstance(model, nn.NeuralModel)


@pytest.mark.parametrize(
    "hiddens", [
        [
            (-1, "relu", "hidden_1"),
            (4, "relu", "hidden_2"),
        ],
        [
            (1, "relu", "hidden_1"),
            (-4, "relu", "hidden_2"),
        ],
        [
            (1, "relu", "hidden_1"),
            (4, "relus", "hidden_2"),
        ],
        [
            (1, "relu", "hidden_1"),
            (4, "relu", 2),
        ],
    ],
)
def test_build_models_with_validation_errors(hiddens):
    "Test build models with wrong parameters"
    builder = mb.Builder(hiddens=hiddens)
    input_shape = (None, 4)
    with pytest.raises(ValidationError):
        builder.build_model(input_shape=input_shape)
