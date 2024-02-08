"Test cases for getting Tensorflow datasets"
import pytest

from deep_tensors.infra import tensorflow_datasets as td


@pytest.mark.parametrize(
    ["dataset", "shape", "length"], [
        ("mnist", (28, 28, 1), 60000),
    ],
)
def test_get_dataset(dataset, shape, length):
    "Test case for getting datasets"
    train_orig, test_orig = td.get_dataset(
        name=dataset,
    )

    assert tuple(train_orig.element_spec["image"].shape) == shape
    assert tuple(test_orig.element_spec["image"].shape) == shape
    assert len(train_orig) == length
