"XOR Network"
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple
import keras
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

from deep_tensors.models import neuralnet as nn
from deep_tensors.models import model_builder as mb
from deep_tensors.models import schemas


TrainValidDataSets = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
Optimizer = Literal["adam", "rmsprop", "sgd"]
Loss = Literal[
    "binary_crossentropy",
    "categorical_crossentropy",
    "mae",
    "mse",
]
Metrics = List[
    Literal[
        "accuracy",
        "binary_accuracy",
        "f1",
        "r2",
    ],
]


@dataclass
class XOR:
    "XOR net"

    units: Tuple[int, ...]
    activations: Tuple[nn.Activation, ...]
    train_count: int
    epochs: int
    batch_size: int

    def check_hidden_layers(self) -> schemas.Hiddens:
        "Check the hidden layers"
        if len(self.units) != len(self.activations):
            raise ValueError("Activations and units should have the same size")
        hiddens: List[Tuple[int, str, str]] = [
            (value[0], value[1], f"hidden_{idx}")
            for idx, value in enumerate(zip(self.units, self.activations), 1)
        ]

        hidden_layers = schemas.Hiddens(hiddens=hiddens)

        return hidden_layers

    def create_data(self) -> Tuple[np.ndarray, np.ndarray]:
        "Create dataset for xor classification"
        x_tensor = np.random.uniform(low=-1, high=1, size=(200, 2))
        y_tensor = np.ones(len(x_tensor))
        y_tensor[x_tensor[:, 0] * x_tensor[:, 1] < 0] = 0

        return x_tensor, y_tensor

    def split_data(self, train: np.ndarray, test: np.ndarray) -> TrainValidDataSets:
        "Split data in train and valid dataset"
        x_train = train[: self.train_count, :]
        y_train = test[: self.train_count]
        x_valid = train[self.train_count :, :]
        y_valid = test[self.train_count :]

        return x_train, y_train, x_valid, y_valid

    def prepare_builder(self) -> mb.Builder:
        "Prepare builder"
        hiddens = self.check_hidden_layers()
        builder = mb.Builder(hiddens.dict()["hiddens"])

        return builder

    def check_model_compiler(
        self,
        optimizer: Optimizer,
        loss: Loss,
        metrics: Metrics,
    ) -> schemas.BaseModelCompiler:
        "Check_model_compiler"
        model_compiler = schemas.BaseModelCompiler(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        return model_compiler

    def train(self) -> None:
        "Train the model"
        train, test = self.create_data()
        x_train, y_train, x_valid, y_valid = self.split_data(
            train=train,
            test=test,
        )
        builder = self.prepare_builder()
        model = builder.build_model(input_shape=(None, 2))
        print(model.summary())
        model.compile(
            optimizer=keras.optimizers.SGD(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()],
        )
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )
        self.plot(
            history=history.history,
            x_valid=x_valid,
            y_valid=y_valid,
            model=model,
        )

    def plot(
        self,
        history: Dict[str, List[float]],
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        model: keras.Model,
    ) -> None:
        "plot results"
        plt.plot(history["loss"], lw=4)
        plt.plot(history["val_loss"], lw=4)
        plt.legend(["Loss training", "loss validation"], fontsize=15)
        plt.xlabel("Epochs", size=15)
        plt.ylabel("Loss", size=15)
        plt.show()
        plt.plot(history["binary_accuracy"], lw=4)
        plt.plot(history["val_binary_accuracy"], lw=4)
        plt.legend(["accuracy training", "accuracy validation"], fontsize=15)
        plt.xlabel("Epochs", size=15)
        plt.ylabel("Accuracy", size=15)
        plt.show()
        plot_decision_regions(
            X=x_valid,
            y=y_valid.astype(np.int32),
            clf=model,
        )
        plt.xlabel(r"$x_1$", size=15)
        plt.ylabel(r"$x_2$", size=15)
        plt.show()
