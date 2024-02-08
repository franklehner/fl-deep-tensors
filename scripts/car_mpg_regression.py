#!/usr/bin/env python
"""This script trains a regression for car mpg
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import logging
import click

from deep_tensors.cars import car_regression as cr


logging.basicConfig(level=logging.INFO)


@dataclass
class EstimatorContext:
    "Context for Estimator"
    model_dir: str
    test_size: float
    regressor: Optional[cr.Regressor] = None


@click.group(chain=True)
@click.pass_context
@click.option(
    "--model-dir",
    type=click.STRING,
    default="data/models/autompg_",
    show_default=True,
    help="Where should the model be?",
)
@click.option(
    "--test-size",
    type=click.FLOAT,
    default=0.2,
    show_default=True,
    help="Test size",
)
def cli(
    ctx,
    model_dir: str,
    test_size: float,
):
    """Client
    """
    ctx.obj = EstimatorContext(
        model_dir=model_dir,
        test_size=test_size,
    )


@cli.command("train-dnn")
@click.pass_obj
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose outputs."
)
@click.option(
    "--hidden",
    "-h",
    type=click.INT,
    multiple=True,
    default=(32, 10),
    show_default=True,
    help="Give the count of hidden units",
)
@click.option(
    "--epochs",
    "-e",
    type=click.INT,
    default=1000,
    show_default=True,
    help="Number of epochs",
)
def train_dnn(
    ctx: EstimatorContext,
    verbose: bool,
    hidden: Tuple[int, ...],
    epochs: int,
):
    "Train dnn regressor"
    ctx.regressor = "DNN"
    if ctx.model_dir.endswith("autompg_"):
        ctx.model_dir += "dnnregressor"
    else:
        raise ValueError(f"Please check model_dir {ctx.model_dir}")

    if verbose:
        logging.info("Initialize DNN Regressor")

    car_regressor = cr.CarRegressor(
        hidden_units=list(hidden),
        epochs=epochs,
        batch_size=8,
        regressor_str=ctx.regressor,
        model_dir=ctx.model_dir,
    )
    car_regressor.train()


@cli.command("eval-dnn")
@click.pass_obj
@click.option(
    "--hidden",
    "-h",
    type=click.INT,
    multiple=True,
    default=(32, 10),
    show_default=True,
    help="Give the count of hidden units",
)
def evaluate_dnn(
    ctx: EstimatorContext,
    hidden: Tuple[int, ...],
):
    "evaluate dnn regressor"
    ctx.regressor = "DNN"
    if ctx.model_dir.endswith("autompg_"):
        ctx.model_dir += "dnnregressor"
    else:
        raise ValueError(f"Please check model_dir {ctx.model_dir}")

    car_regressor = cr.CarRegressor(
        hidden_units=list(hidden),
        regressor_str=ctx.regressor,
        model_dir=ctx.model_dir,
    )

    car_regressor.evaluate()


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
