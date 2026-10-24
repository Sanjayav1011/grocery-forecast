"""
metrics.py
-----------
Forecast accuracy metrics used for model selection and reporting.

WAPE is the headline metric: it is volume-weighted, so it doesn't explode
on low-volume days the way MAPE does. MAPE is kept only for comparison
with older runs.
"""

import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def wape(y_true, y_pred):
    """sum|y - p| / sum|y| x 100."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum() * 100)


def bias_pct(y_true, y_pred):
    """(sum p - sum y) / sum y x 100; positive = over-forecast."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_pred.sum() - y_true.sum()) / y_true.sum() * 100)


def seasonal_naive_scale(train_df, target, group_cols, m=7):
    """
    MASE denominator: mean |y_t - y_{t-m}| of the seasonal-naive forecast on
    the TRAINING data, computed within each series so differences never
    cross from one city/SKU into another.
    """
    ordered = train_df.sort_values(group_cols + ["date"])
    diffs = ordered.groupby(group_cols)[target].diff(m).abs()
    return float(diffs.mean())


def mase(y_true, y_pred, scale):
    """MAE scaled by the training seasonal-naive MAE; < 1 beats seasonal naive."""
    return mae(y_true, y_pred) / scale


def mape(y_true, y_pred):
    """Legacy: mean |y - p| / y over days with y > 0. Kept for comparison only."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, scale):
    """Standard metric block: MAE, RMSE, WAPE, MASE, Bias%."""
    return {
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "WAPE_%": round(wape(y_true, y_pred), 2),
        "MASE": round(mase(y_true, y_pred, scale), 3),
        "Bias_%": round(bias_pct(y_true, y_pred), 2),
    }
