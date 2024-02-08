"Get Tensorflow datasets"
from typing import Tuple
import tensorflow as tf
import tensorflow_datasets as tfds


TrainTest = Tuple[tf.data.Dataset, tf.data.Dataset]


def get_dataset(name: str) -> TrainTest:
    "Get tensorflow dataset"
    builder: tfds.core.DatasetBuilder = tfds.builder(name)
    builder.download_and_prepare()
    datasets: tf.data.Dataset = builder.as_dataset(shuffle_files=False)
    train_orig = datasets["train"]
    test_orig = datasets["test"]

    return train_orig, test_orig
