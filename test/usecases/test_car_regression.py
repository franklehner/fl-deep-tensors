"Test cases for car regression usecase"
import pytest

from deep_tensors.cars import car_regression as cr


@pytest.fixture(name="cars")
def cars_fixture():
    "Base parameter for cars fixture"
    return {
        "hidden_units": [16, 4],
        "epochs": 10,
        "batch_size": 8,
        "regressor_str": "DNN",
        "model_dir": "data/models/autompg_dnnregressor",
    }


def test_initialize_car_regressor(cars):
    "Test initialization of the car regression class"
    car_regressor = cr.CarRegressor(**cars)


    assert car_regressor.hidden_units == cars["hidden_units"]
    assert car_regressor.batch_size == cars["batch_size"]
    assert car_regressor.epochs == cars["epochs"]
    assert car_regressor.regressor_str == cars["regressor_str"]
    assert car_regressor.model_dir == cars["model_dir"]


def test_car_regressor_get_dataset(cars):
    "Test case for get dataset"
    car_regressor = cr.CarRegressor(**cars)
    df_train_norm, df_test_norm = car_regressor.get_datasets()

    assert round(df_train_norm["cylinders"].mean()) == 0
    assert round(df_train_norm["cylinders"].std()) == 1
    assert round(df_test_norm["cylinders"].mean()) == 0
    assert round(df_test_norm["cylinders"].std()) == 1
