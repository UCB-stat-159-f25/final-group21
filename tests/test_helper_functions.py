import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import pytest

from helper_functions import (
    summarize_df, group_data_reliability, show_full_names,
    sort_on_time_rate, on_time_summary_table, group_data_clustering,
    preprocessor, train_models)

############## summarize_df ##############
def test_summarize_df_shape():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = summarize_df(df)
    assert out.shape[0] == 2

def test_summarize_df_columns_exist():
    df = pd.DataFrame({"a": [1, 2]})
    out = summarize_df(df)
    assert "mean" in out.columns


############## group_data_reliability ##############
def test_group_data_reliability_runs():
    df = pd.DataFrame({
        "airline": ["A", "A", "B"],
        "DEPARTURE_DELAY": [1, 2, 3],
        "ARRIVAL_DELAY": [2, 3, 4],
        "dep_on_time": [1, 0, 1],
        "arr_on_time": [1, 1, 0]
    })
    out = group_data_reliability(df, "airline")
    assert isinstance(out, pd.DataFrame)

def test_group_data_reliability_has_two_groups():
    df = pd.DataFrame({
        "airline": ["A", "B"],
        "DEPARTURE_DELAY": [1, 2],
        "ARRIVAL_DELAY": [1, 2],
        "dep_on_time": [1, 1],
        "arr_on_time": [1, 0]
    })
    out = group_data_reliability(df, "airline")
    assert len(out) == 2


############## show_full_names ##############
def test_show_full_names_runs():
    base = pd.DataFrame({"airline": ["AA"], "n_flights": [10]})
    ext = pd.DataFrame({"IATA_CODE": ["AA"], "FULL": ["American"]})
    out = show_full_names(base, ext, "airline", [])
    assert isinstance(out, pd.DataFrame)

def test_show_full_names_drops_column():
    base = pd.DataFrame({"airline": ["AA"], "x": [1]})
    ext = pd.DataFrame({"IATA_CODE": ["AA"]})
    out = show_full_names(base, ext, "airline", ["x"])
    assert "x" not in out.columns


############## sort_on_time_rate ##############
def test_sort_on_time_rate_runs():
    df = pd.DataFrame({
        "arr_on_time_rate": [0.5, 0.9],
        "dep_on_time_rate": [0.4, 0.8],
        "mean_arr_delay": [10, 5],
        "mean_dep_delay": [8, 3]
    })
    out = sort_on_time_rate(df)
    assert isinstance(out, pd.DataFrame)

def test_sort_on_time_rate_same_length():
    df = pd.DataFrame({
        "arr_on_time_rate": [0.1, 0.2],
        "dep_on_time_rate": [0.1, 0.2],
        "mean_arr_delay": [1, 2],
        "mean_dep_delay": [1, 2]
    })
    out = sort_on_time_rate(df)
    assert len(out) == 2


############## on_time_summary_table ##############
def test_on_time_summary_table_runs():
    df = pd.DataFrame({
        "airline": ["A"],
        "n_flights": [10],
        "arr_on_time_rate": [0.9],
        "dep_on_time_rate": [0.8],
        "mean_arr_delay": [5],
        "mean_dep_delay": [4]
    })
    out = on_time_summary_table(df, "airline", "airline", "Airline")
    assert isinstance(out, pd.DataFrame)

def test_on_time_summary_table_rows():
    df = pd.DataFrame({
        "airline": ["A", "B"],
        "n_flights": [1, 2],
        "arr_on_time_rate": [0.1, 0.2],
        "dep_on_time_rate": [0.1, 0.2],
        "mean_arr_delay": [1, 2],
        "mean_dep_delay": [1, 2]
    })
    out = on_time_summary_table(df, "airline", "airline", "Airline")
    assert len(out) == 2


############## group_data_clustering ##############
def test_group_data_clustering_runs():
    df = pd.DataFrame({"g": ["A", "A"], "x": [1, 2]})
    out = group_data_clustering(df, "g", ["x"])
    assert isinstance(out, pd.DataFrame)

def test_group_data_clustering_not_empty():
    df = pd.DataFrame({"g": ["A", "B"], "x": [1, 2]})
    out = group_data_clustering(df, "g", ["x"])
    assert len(out) == 2


############## preprocessor ##############
def test_preprocessor_runs():
    X = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = preprocessor(X, ["b"])
    assert p is not None

def test_preprocessor_transform_shape():
    X = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = preprocessor(X, ["b"])
    Xt = p.fit_transform(X)
    assert Xt.shape[0] == 2


############## train_models ##############
def test_train_models_regression_runs():
    X = pd.DataFrame({"X": [1, 2, 3, 4]})
    y = pd.Series([1, 2, 3, 4])
    models = {"Linear Regression": LinearRegression()}
    p = preprocessor(X, [])
    out = train_models("regression", models, p, X, y, X, y)
    assert "Linear Regression" in out

def test_train_models_classification_runs():
    X = pd.DataFrame({"X": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    models = {"Logistic Regression": LogisticRegression()}
    p = preprocessor(X, [])
    out = train_models("classification", models, p, X, y, X, y)
    assert "Logistic Regression" in out
