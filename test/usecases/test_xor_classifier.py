"""Test cases for xor_classifier
"""
import pytest
from pydantic_core._pydantic_core import ValidationError  # pylint: disable=no-name-in-module
import numpy as np

from deep_tensors.models import schemas
from deep_tensors.models import model_builder as mb
from deep_tensors.xor.xor_classifier import XOR


@pytest.fixture(name="xor_model")
def xor_model_fixture():
    "base xor model"
    return XOR(
        units=(16, 16),
        activations=("relu", "softmax"),
        train_count=100,
        epochs=1000,
        batch_size=32,
    )


@pytest.mark.parametrize(
    ["units", "activations", "train_count", "epochs", "batch_size"], [
        (
            (1,),
            ("relu"),
            100,
            1000,
            32,
        ),
        (
            (1, 1),
            ("relu"),
            100,
            1000,
            32,
        ),
        (
            (1,),
            ("relu", "relu"),
            100,
            1000,
            32,
        ),
    ],
)
def test_xor_initializer(units, activations, train_count, epochs, batch_size):
    "test the constructor of the XOR class"
    xor = XOR(
        units=units,
        activations=activations,
        train_count=train_count,
        epochs=epochs,
        batch_size=batch_size,
    )

    assert isinstance(xor, XOR)
    assert xor.units == units
    assert xor.activations == activations
    assert xor.train_count == train_count
    assert xor.epochs == epochs
    assert xor.batch_size == batch_size


@pytest.mark.parametrize(
    ["units", "activations"], [
        (
            (16, 16),
            ("relu", "softmax"),
        ),
        (
            (16, 16),
            ("relu", "softmax"),
        ),
        (
            (16, 16),
            ("relu", "softmax"),
        ),
    ],
)
def test_xor_check_hidden_layers(units, activations):
    "Test the validation of the hidden layers"
    epochs = 1000
    train_count = 100
    batch_size = 32

    xor = XOR(
        units=units,
        activations=activations,
        train_count=train_count,
        epochs=epochs,
        batch_size=batch_size,
    )

    hidden_layers = xor.check_hidden_layers()

    assert isinstance(hidden_layers, schemas.Hiddens)
    assert len(hidden_layers.hiddens) == min(len(units), len(activations))


@pytest.mark.parametrize(
    ["units", "activations"], [
        (
            (16,),
            ("relu", "softmax"),
        ),
        (
            (16, 16, 16),
            ("relu", "softmax"),
        ),
        (
            (16, 16),
            ("relu", "relu", "softmax"),
        ),
    ],
)
def test_xor_check_hidden_layers_with_different_sizes(units, activations):
    "Test the validation of the hidden layers"
    epochs = 1000
    train_count = 100
    batch_size = 32

    xor = XOR(
        units=units,
        activations=activations,
        train_count=train_count,
        epochs=epochs,
        batch_size=batch_size,
    )

    with pytest.raises(ValueError):
        xor.check_hidden_layers()


@pytest.mark.parametrize(
    ["units", "activations"], [
        (
            (-1, 16),
            ("relu", "softmax"),
        ),
        (
            (1, 16),
            ("rule", "softmax"),
        ),
        (
            (1, 16),
            (5, "softmax"),
        ),
        (
            ("relu", 16),
            ("relu", "softmax"),
        ),
    ],
)
def test_xor_check_hidden_layers_with_validation_error(units, activations):
    "test the validation of the hidden layers"
    epochs = 1000
    train_count = 100
    batch_size = 32

    xor = XOR(
        units=units,
        activations=activations,
        train_count=train_count,
        epochs=epochs,
        batch_size=batch_size,
    )

    with pytest.raises(ValidationError):
        xor.check_hidden_layers()


def test_xor_create_data(xor_model):
    "Test case for data creation"
    train, test = xor_model.create_data()

    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)


@pytest.mark.parametrize(
    "train_count", [
        10,
        20,
        50,
        100,
        150,
    ],
)
def test_xor_split_data(train_count, xor_model):
    "Test case for splitting data"
    xor_model.train_count = train_count
    train, test = xor_model.create_data()

    x_train, y_train, x_valid, y_valid = xor_model.split_data(
        train=train,
        test=test,
    )

    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_valid, np.ndarray)
    assert isinstance(y_valid, np.ndarray)
    assert len(x_train) == train_count
    assert len(x_valid) == 200 - train_count


def test_xor_prepare_builder(xor_model):
    "Test case for prepare_builder method"
    expected_hiddens = [
        (16, "relu", "hidden_1"),
        (16, "softmax", "hidden_2"),
    ]

    builder = xor_model.prepare_builder()

    assert isinstance(builder, mb.Builder)
    assert builder.hiddens == expected_hiddens


@pytest.mark.parametrize(
    ["units", "activations", "error"], [
        (
            (16, -1),
            ("relu", "softmax"),
            ValidationError,
        ),
        (
            (16, 10),
            ("relu", "sotmax"), # wrong written softmax
            ValidationError,
        ),
        (
            (16, 16, 10),
            ("relu", "softmax"),
            ValueError,
        ),
    ],
)
def test_xor_prepare_builder_raises_error(units, activations, error):
    "Test case for failed builder"
    epochs = 1000
    batch_size = 32
    train_count = 100

    xor = XOR(
        units=units,
        activations=activations,
        train_count=train_count,
        epochs=epochs,
        batch_size=batch_size,
    )

    with pytest.raises(error):
        xor.prepare_builder()


@pytest.mark.parametrize(
    ["optimizer", "loss", "metrics"], [
        (
            "adam",
            "categorical_crossentropy",
            ["accuracy"],
        ),
        (
            "rmsprop",
            "categorical_crossentropy",
            ["accuracy"],
        ),
        (
            "sgd",
            "categorical_crossentropy",
            ["accuracy"],
        ),
        (
            "sgd",
            "binary_crossentropy",
            ["accuracy"],
        ),
        (
            "sgd",
            "mae",
            ["accuracy"],
        ),
        (
            "sgd",
            "mse",
            ["accuracy"],
        ),
        (
            "sgd",
            "mse",
            ["binary_accuracy"],
        ),
        (
            "sgd",
            "mse",
            ["f1"],
        ),
        (
            "sgd",
            "mse",
            ["r2"],
        ),
        (
            "sgd",
            "mse",
            ["r2", "accuracy", "f1", "binary_accuracy"],
        ),
    ],
)
def test_xor_check_model_compiler(optimizer, loss, metrics, xor_model):
    "Test case for check_model_compiler_method"
    model_compiler = xor_model.check_model_compiler(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    assert isinstance(model_compiler, schemas.BaseModelCompiler)


@pytest.mark.parametrize(
    "optimizer", [
        "sga",
        "rms",
        "atam",
    ],
)
def test_xor_check_model_compiler_with_wrong_optimizer(optimizer, xor_model):
    "Test case for wrong optimizer"
    with pytest.raises(ValidationError):
        xor_model.check_model_compiler(
            optimizer=optimizer,
            loss="mse",
            metrics=["accuracy"],
        )


@pytest.mark.parametrize(
    "loss", [
        "foo",
        1,
        "mean",
    ],
)
def test_xor_check_model_compiler_with_wrong_loss(loss, xor_model):
    "Test case for wrong optimizer"
    with pytest.raises(ValidationError):
        xor_model.check_model_compiler(
            optimizer="sgd",
            loss=loss,
            metrics=["accuracy"],
        )


@pytest.mark.parametrize(
    "metrics", [
        ["foo"],
        1,
        "mean",
    ],
)
def test_xor_check_model_compiler_with_wrong_metrics(metrics, xor_model):
    "Test case for wrong optimizer"
    with pytest.raises(ValidationError):
        xor_model.check_model_compiler(
            optimizer="sgd",
            loss="mse",
            metrics=metrics,
        )
