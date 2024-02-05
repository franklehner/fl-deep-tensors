"""Test cases for the neuralnet models
"""
import pytest
from pydantic_core._pydantic_core import ValidationError  # pylint: disable=no-name-in-module

from deep_tensors.models import neuralnet


@pytest.mark.parametrize(
    ("units", "activation", "name"), [
        (1, "relu", "foo"),
        (2, "sigmoid", "foo"),
        (2, "softmax", "foo"),
        (2, "tanh", "foo"),
    ],
)
def test_initialize_dense_parameters(units, activation, name):
    "Test the initializing of the dense parameters"
    dense = neuralnet.DenseParameters(
        units=units,
        activation=activation,
        name=name,
    )

    assert dense.units == units
    assert dense.activation == activation
    assert dense.name == name


@pytest.mark.parametrize(
    "units", [
        -1,
        0,
    ],
)
def test_initialize_dense_parameters_with_negative_number_unit(units):
    "Test units with not positive intgers"
    activation = "relu"
    name = "foo"
    with pytest.raises(ValidationError):
        neuralnet.DenseParameters(
            units=units,
            activation=activation,
            name=name,
        )


@pytest.mark.parametrize(
    "activation", [
        "relus",
        "foo",
        "own",
    ],
)
def test_initialize_dense_parameters_with_wrong_activation(activation):
    "Test activations of the dense params"
    units = 1
    name = "foo"
    with pytest.raises(ValidationError):
        neuralnet.DenseParameters(
            units=units,
            activation=activation,
            name=name,
        )


@pytest.mark.parametrize(
    ["units", "activation", "name"], [
        (1, "relu", "foo"),
        (2, "sigmoid", "foo"),
        (2, "softmax", "foo"),
        (2, "tanh", "foo"),
    ],
)
def test_dense_params_get_layer(units, activation, name):
    "Test the output of the get_layer method"
    params = neuralnet.DenseParameters(
        units=units,
        activation=activation,
        name=name,
    )

    dense = params.get_layer()

    assert isinstance(dense, neuralnet.keras.layers.Dense)
    assert dense.activation.__name__ == activation
    assert dense.name == name
    assert dense.units == units


def test_neural_model():
    "Test the initializing of the neural model class"
    hidden_1 = neuralnet.DenseParameters(
        units=4,
        activation="relu",
        name="hidden_1",
    )
    hidden_2 = neuralnet.DenseParameters(
        units=4,
        activation="softmax",
        name="output",
    )
    hiddens = [hidden_1.get_layer(), hidden_2.get_layer()]
    model = neuralnet.NeuralModel(hiddens=hiddens)
    model.build(input_shape=(None, 4))

    assert len(model.layers) == len(hiddens)
    assert model.layers == hiddens


def test_neural_model_with_wrong_layer_class():
    "Test the initializing of the neural model class with wrong layers"
    hidden_1 = neuralnet.DenseParameters(
        units=4,
        activation="relu",
        name="hidden_1",
    )
    hidden_2 = neuralnet.DenseParameters(
        units=4,
        activation="softmax",
        name="output",
    )
    hiddens = [hidden_1, hidden_2]
    with pytest.raises(ValueError):
        model = neuralnet.NeuralModel(hiddens=hiddens)
        model.build(input_shape=(None, 4))
