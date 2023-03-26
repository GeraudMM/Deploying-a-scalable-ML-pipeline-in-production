"""the goal of this file is to unit test our function"""
import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)
from ml.data import process_data


@pytest.fixture
def input_data():
    X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
    input_data = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2', 'feature_3'])
    input_data['feature_1'] = pd.qcut(input_data['feature_1'], 4, labels=['a', 'b', 'c', 'd'])
    input_data['feature_3'] = pd.qcut(input_data['feature_3'], 4, labels=['x', 'y', 'z', 'w'])
    input_data['label'] = y
    return input_data


@pytest.fixture
def trained_model(input_data):
    X, y, encoder, lb = process_data(
        input_data, categorical_features=[
            'feature_1', 'feature_3'], label='label', training=True)
    model = LogisticRegression()
    model.fit(X, y)
    return model


@pytest.fixture
def encoder_lb(input_data):
    X, y, encoder, lb = process_data(
        input_data, categorical_features=[
            'feature_1', 'feature_3'], label='label', training=True)
    return encoder, lb


def test_train_model():
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


@pytest.mark.parametrize("y, preds, expected_precision, expected_recall, expected_fbeta", [
    ([1, 0, 1], [1, 0, 1], 1, 1, 1),
    ([1, 0, 1], [0, 1, 0], 0.0, 0.0, 0.0),
    ([1, 0, 1], [1, 1, 1], 0.67, 1, 0.8)
])
def test_compute_model_metrics(y, preds, expected_precision, expected_recall, expected_fbeta):
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert np.round(precision, 2) == expected_precision
    assert np.round(recall, 2) == expected_recall
    assert np.round(fbeta, 2) == expected_fbeta


@pytest.mark.parametrize("feature, value, expected_precision, expected_recall, expected_fbeta", [
    ('feature_1', 'a', 0.98, 1.0, 0.99),
    ('feature_1', 'b', 0.82, 0.94, 0.88),
    ('feature_1', 'c', 0.73, 0.58, 0.65),
    ('feature_1', 'd', 0.96, 0.46, 0.62),
    ('feature_3', 'x', 0.98, 1.0, 0.99),
    ('feature_3', 'y', 0.78, 0.95, 0.86),
    ('feature_3', 'z', 0.92, 0.36, 0.52),
])
def test_compute_metric_on_slice_of_data(
        trained_model,
        encoder_lb,
        input_data,
        feature,
        value,
        expected_precision,
        expected_recall,
        expected_fbeta):
    encoder, lb = encoder_lb
    slice_df = input_data[input_data[feature] == value]
    slice_X, slice_y, _, _ = process_data(
        slice_df, categorical_features=[
            'feature_1', 'feature_3'], label='label', encoder=encoder, lb=lb, training=False)
    preds = inference(trained_model, slice_X)
    precision, recall, fbeta = compute_model_metrics(slice_y, preds)

    assert np.isclose(precision, expected_precision, rtol=1e-2)
    assert np.isclose(recall, expected_recall, rtol=1e-2)
    assert np.isclose(fbeta, expected_fbeta, rtol=1e-2)
