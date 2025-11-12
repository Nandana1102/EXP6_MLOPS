import os, joblib
from train_pipeline import train_model

def test_training_random_forest():
    metric = train_model("random_forest")
    assert metric > 0.5
    assert os.path.exists("random_forest_model.pkl")

def test_training_linear_regression():
    metric = train_model("linear_regression")
    assert metric > 0
    assert os.path.exists("linear_regression_model.pkl")
