"Get Tensorflow datasets"
from typing import Literal, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds

from deep_tensors.models.schemas import TensorflowDataSet


TrainTest = Tuple[tf.data.Dataset, tf.data.Dataset]
Name = Literal["mnist"]


def get_dataset(name: Name) -> TrainTest:
    "Get tensorflow dataset"
    TensorflowDataSet(name=name)
    builder: tfds.core.DatasetBuilder = tfds.builder(name)
    builder.download_and_prepare()
    datasets: tf.data.Dataset = builder.as_dataset(shuffle_files=False)
    train_orig = datasets["train"]
    test_orig = datasets["test"]

    return train_orig, test_orig
