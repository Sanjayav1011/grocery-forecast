"""
inventory.py
--------------
Shared inventory logic, so the synthetic generator, the model risk report
(train_model.py) and the catalog-wide regional report
(build_bigbasket_regional_risk.py) all use the same rules:

  - simulate_inventory: periodic-review replenishment. Each series is
    delivered every R = min(DELIVERY_CYCLE_DAYS, shelf life) days, ordered up
    to trailing demand x (R + SAFETY_STOCK_DAYS); perishable stock is written
    off when it reaches its shelf life; some deliveries are over-orders.
  - classify_risk: stockout / overstock / normal plus a reorder quantity
    from a daily demand forecast and the morning's opening stock.
"""

import math

import numpy as np

from config import DELIVERY_CYCLE_DAYS, LEAD_TIME_DAYS, SAFETY_STOCK_DAYS, SHELF_LIFE_DAYS

OVERORDER_PROBABILITY = 0.05  # chance a delivery is a planner's over-order


def review_days_for(categories):
    """Delivery cycle per category: weekly, or every shelf-life days if shorter."""
    return np.array([min(DELIVERY_CYCLE_DAYS, SHELF_LIFE_DAYS[c]) for c in categories])


def trailing_mean(values, window, first):
    """
    Mean of the previous `window` days (fewer at the start) for each row of
    a (n_series, n_days) array; day 0 has no history and gets `first`.
    Vectorised with cumulative sums so it scales to many series.
    """
    n_series, n_days = values.shape
    cumulative = np.concatenate([np.zeros((n_series, 1)), np.cumsum(values, axis=1)], axis=1)
    days = np.arange(n_days)
    start = np.maximum(days - window, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (cumulative[:, days] - cumulative[:, start]) / (days - start)
    means[:, 0] = first
    return means


def simulate_inventory(true_demand, base, categories, rng, initial_stock=None, phase=None, history=None):
    """
    Runs the replenishment policy for many series at once.

    true_demand: (n_series, n_days) daily demand; base: (n_series,) typical
    daily demand, used before any history exists; categories: model category
    per series. Returns opening_stock, units_sold (float) and
    days_until_delivery, each (n_series, n_days).

    To continue a series that already ran (generate_synthetic_data.extend),
    pass the stock left at the end of the previous day (initial_stock), the
    delivery phase for day 0 (deliveries on days t with (t - phase) % R == 0)
    and the previous days' demand (history, (n_series, n_past)) for the
    trailing average. Left as None, everything behaves as before: a random
    phase and stock that covers the days until the first delivery.
    """
    true_demand = np.atleast_2d(np.asarray(true_demand, dtype=float))
    n_series, n_days = true_demand.shape
    base = np.broadcast_to(np.asarray(base, dtype=float), (n_series,))
    shelf_life = np.array([SHELF_LIFE_DAYS[c] for c in categories])
    review_days = review_days_for(categories)

    if phase is None:
        phase = rng.integers(0, review_days)  # stores don't all receive on the same weekday
    phase = np.asarray(phase)
    if history is None:
        trailing_avg = trailing_mean(true_demand, 14, base)
    else:
        history = np.atleast_2d(np.asarray(history, dtype=float))
        trailing_avg = trailing_mean(np.concatenate([history, true_demand], axis=1), 14, base)[:, history.shape[1]:]
    overorder = rng.random((n_series, n_days)) < OVERORDER_PROBABILITY
    overorder_mult = rng.uniform(3.0, 5.0, (n_series, n_days))

    opening = np.empty((n_series, n_days))
    sold = np.empty((n_series, n_days))
    stock = base * phase if initial_stock is None else np.asarray(initial_stock, dtype=float)  # enough to reach the first delivery
    expires = shelf_life <= review_days  # the previous batch is spoiled by the next delivery
    for t in range(n_days):
        delivery = (t - phase) % review_days == 0
        stock = np.where(delivery & expires, 0.0, stock)
        target = trailing_avg[:, t] * (review_days + SAFETY_STOCK_DAYS) * np.where(overorder[:, t], overorder_mult[:, t], 1.0)
        stock = np.where(delivery, np.maximum(stock, target), stock)
        opening[:, t] = stock
        sold[:, t] = np.minimum(true_demand[:, t], stock)
        stock = stock - sold[:, t]
    days_until_delivery = review_days[:, None] - (np.arange(n_days)[None, :] - phase[:, None]) % review_days[:, None]
    return opening, sold, days_until_delivery


def classify_risk(df, horizon=LEAD_TIME_DAYS, category_col="category"):
    """
    Adds horizon_demand, days_of_cover, shelf_life_days, risk_status and
    reorder_qty from avg_daily_pred (forecast units/day over the horizon),
    opening_stock and the model category in `category_col`. Stockout = stock won't last until the next
    delivery (days_until_delivery, or `horizon` when unknown); overstock =
    more cover than horizon + 2 + shelf life.
    """
    df = df.copy()
    df["horizon_demand"] = df["avg_daily_pred"] * horizon
    df["days_of_cover"] = df["opening_stock"] / df["avg_daily_pred"].clip(lower=0.1)
    df["shelf_life_days"] = df[category_col].map(SHELF_LIFE_DAYS)
    if "days_until_delivery" not in df.columns:
        df["days_until_delivery"] = horizon
    df["risk_status"] = np.select(
        [df["days_of_cover"] < df["days_until_delivery"], df["days_of_cover"] > horizon + 2 + df["shelf_life_days"]],
        ["stockout", "overstock"],
        default="normal",
    )
    shortfall = (df["horizon_demand"] + 2 * df["avg_daily_pred"] - df["opening_stock"]).clip(lower=0)
    df["reorder_qty"] = shortfall.map(math.ceil).astype(int)
    return df
