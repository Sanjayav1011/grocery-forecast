"""
favorita_data.py
------------------
Loads, cleans and featurises a laptop-sized subset of the Corporación
Favorita grocery sales data (Ecuador, Kaggle) for the real-data validation.

  subset    10 stores (the busiest store, by transactions, in each of the 10
            busiest cities) and dates START..END. train.csv is read in chunks
            with explicit dtypes and saved once as parquet in data/favorita/.
  cleaning  negative unit_sales (returns) clipped to 0; missing onpromotion
            = False. train.csv only has rows with sales, so every store x item
            series is filled with 0-sale days from the item's first sale in
            that store. Days a store was closed (no rows at all, e.g. 25 Dec)
            stay in the grid for the lags (as 0) but are never trained on or
            scored. Items: the TOP_ITEMS_PER_STORE best sellers per store,
            ranked on the training period only.
  features  the main project's leak-free lags and rolling stats
            (build_features.add_lag_and_rolling_features, shifted >= 7
            days), calendar, promotion, holidays (holiday_features), oil price
            (forward-filled, lagged 7 days) and an earthquake flag.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import FAVORITA_DATA_DIR, LEAD_TIME_DAYS  # noqa: E402
from build_features import (add_lag_and_rolling_features, apply_categorical_mappings,  # noqa: E402
                            build_categorical_mappings)

START, END = "2016-01-01", "2017-08-15"
TRAIN_END = "2017-03-31"
TEST = ("2017-04-01", "2017-08-15")
N_STORES = 10
TOP_ITEMS_PER_STORE = 1500
EARTHQUAKE = ("2016-04-16", "2016-05-16")  # Manabí earthquake and the month of relief buying after it
HOLIDAY_DISTANCE_CAP = 60  # days to / after a holiday, capped
CHUNK_ROWS = 5_000_000
TRAIN_DTYPES = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32", "onpromotion": "object"}

SERIES_COLS = ["store_nbr", "item_nbr"]
TARGET = "demand"  # units sold; Favorita has no stock data, so sales are not un-censored
CATEGORICAL_COLS = ["store_nbr", "item_nbr", "family", "city"]
CATEGORICAL_FEATURES = [f"{col}_code" for col in CATEGORICAL_COLS]
LAG_COLS = [f"lag_{LEAD_TIME_DAYS}", f"lag_{2 * LEAD_TIME_DAYS}", f"lag_{3 * LEAD_TIME_DAYS}", "lag_364",
            f"roll_mean_7_lag{LEAD_TIME_DAYS}", f"roll_mean_14_lag{LEAD_TIME_DAYS}",
            f"roll_mean_28_lag{LEAD_TIME_DAYS}", f"roll_std_14_lag{LEAD_TIME_DAYS}"]
HOLIDAY_COLS = ["is_holiday", "is_national_holiday", "is_local_holiday", "is_event", "is_work_day",
                "days_until_next_holiday", "days_since_last_holiday"]
FEATURE_COLS = (["on_promotion", "day_of_week", "is_weekend", "month", "week_of_year", "is_month_start",
                 "is_month_end"] + HOLIDAY_COLS + ["earthquake", "oil_price_lag7"] + LAG_COLS + CATEGORICAL_FEATURES)


def raw_path(name):
    return FAVORITA_DATA_DIR / name


def data_available():
    return all(raw_path(name).exists() for name in
               ("train.csv", "items.csv", "stores.csv", "holidays_events.csv", "oil.csv", "transactions.csv"))


def select_stores(n=N_STORES):
    """The busiest store (total transactions START..END) in each of the n busiest cities."""
    tx = pd.read_csv(raw_path("transactions.csv"), parse_dates=["date"])
    tx = tx[(tx["date"] >= START) & (tx["date"] <= END)]
    per_store = (tx.groupby("store_nbr")["transactions"].sum().rename("transactions").reset_index()
                 .merge(pd.read_csv(raw_path("stores.csv")), on="store_nbr"))
    cities = per_store.groupby("city")["transactions"].sum().sort_values(ascending=False).head(n)
    stores = (per_store[per_store["city"].isin(cities.index)].sort_values("transactions", ascending=False)
              .drop_duplicates("city").set_index("city").loc[cities.index].reset_index())
    stores["city_rank"] = np.arange(1, len(stores) + 1)
    stores["reason"] = [f"busiest store in {city}, #{rank} city by transactions {START}..{END}"
                        for city, rank in zip(stores["city"], stores["city_rank"])]
    return stores[["store_nbr", "city", "state", "type", "cluster", "transactions", "city_rank", "reason"]]


def subset_path(stores):
    return raw_path(f"train_subset_{START}_{END}_stores_{'-'.join(map(str, sorted(stores)))}.parquet")


def load_subset(stores):
    """train.csv rows for `stores` and START..END, cached as parquet (first run reads 4.7 GB in chunks)."""
    path = subset_path(stores)
    if path.exists():
        return pd.read_parquet(path)
    started, kept, keep = time.perf_counter(), [], set(stores)
    reader = pd.read_csv(raw_path("train.csv"), usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
                         dtype={"date": "object", **TRAIN_DTYPES}, chunksize=CHUNK_ROWS)
    for number, chunk in enumerate(reader, start=1):
        first, last = chunk["date"].iloc[0], chunk["date"].iloc[-1]  # train.csv is sorted by date
        if last >= START:
            chunk = chunk[(chunk["date"] >= START) & (chunk["date"] <= END) & chunk["store_nbr"].isin(keep)]
            kept.append(chunk)
        print(f"  chunk {number}: {first}..{last}, kept {sum(len(c) for c in kept):,} rows "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        if first > END:
            break
    rows = pd.concat(kept, ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    rows.to_parquet(path, index=False)
    print(f"Saved {len(rows):,} rows to {path} in {time.perf_counter() - started:.0f}s")
    return rows


def clean(rows):
    """Clips returns to 0 and fills missing onpromotion with False; returns (rows, stats)."""
    rows = rows.copy()
    promo = rows["onpromotion"]
    stats = {"raw_rows": len(rows), "negative_sales_clipped": int((rows["unit_sales"] < 0).sum()),
             "onpromotion_missing_filled": int(promo.isna().sum())}
    rows["unit_sales"] = rows["unit_sales"].clip(lower=0)
    if promo.dtype != bool:
        promo = promo.map({True: True, False: False, "True": True, "False": False})
    rows["onpromotion"] = promo.fillna(False).astype(bool)
    return rows, stats


def top_items(rows, n=TOP_ITEMS_PER_STORE, ranked_until=TRAIN_END):
    """
    (store_nbr, item_nbr) pairs of the n best-selling items per store, ranked on
    sales up to `ranked_until` only (the test window never picks the items),
    and the share of total sales they cover.
    """
    ranking = rows[rows["date"] <= ranked_until].groupby(SERIES_COLS)["unit_sales"].sum()
    ranking = ranking.sort_values(ascending=False).groupby(level="store_nbr").head(n)
    keep = ranking.index.to_frame(index=False)
    total = rows.groupby("store_nbr")["unit_sales"].sum()
    covered = rows.merge(keep, on=SERIES_COLS).groupby("store_nbr")["unit_sales"].sum()
    coverage = {
        "items_per_store_sold_at_least_once": {int(s): int(c) for s, c in rows.groupby("store_nbr")["item_nbr"].nunique().items()},
        "items_kept_per_store": {int(s): int(c) for s, c in keep.groupby("store_nbr").size().items()},
        "unique_items_kept": int(keep["item_nbr"].nunique()),
        "sales_share_covered_%": round(float(covered.sum() / total.sum() * 100), 2),
        "sales_share_covered_by_store_%": {int(s): round(float(covered[s] / total[s] * 100), 2) for s in total.index},
    }
    return keep, coverage


def build_grid(rows, keep):
    """
    One row per kept store x item x calendar day, from the item's first sale
    in that store to END; days without a sale get 0. store_open = 0 on days
    the store has no sales at all (closed).
    """
    rows = rows.merge(keep, on=SERIES_COLS)
    first_sale = rows.groupby(SERIES_COLS)["date"].min().rename("first_sale").reset_index()
    dates = pd.date_range(START, END, freq="D")
    days = (pd.Timestamp(END) - first_sale["first_sale"]).dt.days.to_numpy() + 1
    grid = pd.DataFrame({
        "store_nbr": np.repeat(first_sale["store_nbr"].to_numpy(), days),
        "item_nbr": np.repeat(first_sale["item_nbr"].to_numpy(), days),
        "date": np.concatenate([dates[len(dates) - d:] for d in days]),
    })
    grid = grid.merge(rows[SERIES_COLS + ["date", "unit_sales", "onpromotion"]], on=SERIES_COLS + ["date"], how="left")
    filled = int(grid["unit_sales"].isna().sum())
    grid["unit_sales"] = grid["unit_sales"].fillna(0).astype("float32")
    grid["onpromotion"] = grid["onpromotion"].fillna(False).astype(bool)
    open_days = rows[["store_nbr", "date"]].drop_duplicates().assign(store_open=1)
    grid = grid.merge(open_days, on=["store_nbr", "date"], how="left")
    grid["store_open"] = grid["store_open"].fillna(0).astype("int8")
    return grid, {"grid_rows": len(grid), "zero_sale_days_filled": filled,
                  "closed_store_days": int((grid["store_open"] == 0).sum())}


def holiday_table():
    """
    holidays_events.csv as dated flags per locale:
      Holiday / Transfer / Additional / Bridge  a day off (is_holiday), except
                 rows with transferred=True: the holiday was moved to another
                 day (its Transfer row), so the original date is a normal day
      Work Day   a weekend day worked to make up a Bridge: not a holiday
      Event      national events (Black Friday, Mother's Day, World Cup, the
                 earthquake days): is_event, not a holiday
    locale National applies to every store, Regional to the store's state
    (locale_name), Local to the store's city.
    """
    h = pd.read_csv(raw_path("holidays_events.csv"), parse_dates=["date"])
    h["transferred"] = h["transferred"].astype(str).str.lower() == "true"
    h["holiday"] = h["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"]) & ~h["transferred"]
    h["event"] = h["type"].eq("Event")
    h["work_day"] = h["type"].eq("Work Day")
    return h


def holiday_features(stores, dates=None):
    """HOLIDAY_COLS per store_nbr x date (stores: store_nbr, city, state)."""
    h = holiday_table()
    dates = pd.date_range(START, END, freq="D") if dates is None else pd.DatetimeIndex(dates)
    frames = []
    for store in stores.itertuples():
        applies = ((h["locale"] == "National") | ((h["locale"] == "Regional") & (h["locale_name"] == store.state))
                   | ((h["locale"] == "Local") & (h["locale_name"] == store.city)))
        mine = h[applies]
        day = lambda mask: pd.DatetimeIndex(mine.loc[mask, "date"].unique())
        national = day(mine["holiday"] & (mine["locale"] == "National"))
        local = day(mine["holiday"] & (mine["locale"] != "National"))
        holidays = np.sort(day(mine["holiday"]).to_numpy())
        frame = pd.DataFrame({"store_nbr": store.store_nbr, "date": dates})
        frame["is_national_holiday"] = frame["date"].isin(national).astype("int8")
        frame["is_local_holiday"] = frame["date"].isin(local).astype("int8")
        frame["is_holiday"] = (frame["is_national_holiday"] | frame["is_local_holiday"]).astype("int8")
        frame["is_event"] = frame["date"].isin(day(mine["event"])).astype("int8")
        frame["is_work_day"] = frame["date"].isin(day(mine["work_day"])).astype("int8")
        values = frame["date"].to_numpy()
        after = np.searchsorted(holidays, values, side="left")  # next holiday on or after the day
        before = np.searchsorted(holidays, values, side="right") - 1  # last holiday on or before
        big = np.timedelta64(HOLIDAY_DISTANCE_CAP, "D")
        until = np.where(after < len(holidays), holidays[np.minimum(after, len(holidays) - 1)] - values, big)
        since = np.where(before >= 0, values - holidays[np.maximum(before, 0)], big)
        frame["days_until_next_holiday"] = np.minimum(until, big).astype("timedelta64[D]").astype(int)
        frame["days_since_last_holiday"] = np.minimum(since, big).astype("timedelta64[D]").astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def oil_feature():
    """Daily WTI oil price, gaps forward-filled, shifted LEAD_TIME_DAYS days (known when the forecast is made)."""
    oil = pd.read_csv(raw_path("oil.csv"), parse_dates=["date"]).set_index("date")["dcoilwtico"]
    days = pd.date_range(pd.Timestamp(START) - pd.Timedelta(days=LEAD_TIME_DAYS + 7), END, freq="D")
    oil = oil.reindex(days).ffill().bfill()
    return oil.shift(LEAD_TIME_DAYS).rename("oil_price_lag7").rename_axis("date").reset_index()


def add_features(grid, stores):
    """Calendar, promotion, holiday, oil, earthquake, item and leak-free lag features, plus categorical codes."""
    items = pd.read_csv(raw_path("items.csv"))
    df = grid.merge(items[["item_nbr", "family", "perishable"]], on="item_nbr", how="left")
    df = df.merge(stores[["store_nbr", "city"]], on="store_nbr", how="left")
    df[TARGET] = df["unit_sales"].astype("float32")
    df["on_promotion"] = df["onpromotion"].astype("int8")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int8")
    df["is_month_start"] = df["date"].dt.is_month_start.astype("int8")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("int8")
    df = df.merge(holiday_features(stores), on=["store_nbr", "date"], how="left")
    df["earthquake"] = df["date"].between(*EARTHQUAKE).astype("int8")
    df = df.merge(oil_feature(), on="date", how="left")
    df = add_lag_and_rolling_features(df, group_cols=SERIES_COLS, target_col=TARGET)
    df[LAG_COLS] = df[LAG_COLS].astype("float32")
    df = apply_categorical_mappings(df, build_categorical_mappings(df, CATEGORICAL_COLS))
    return df.reset_index(drop=True)


def features_path(stores):
    return raw_path(f"features_{START}_{END}_top{TOP_ITEMS_PER_STORE}_stores_{'-'.join(map(str, sorted(stores)))}.parquet")


def build_dataset():
    """
    (features, info): the modelling frame (open store-days with full
    short-lag history) and the subset / cleaning / coverage facts for
    results.json. The frame is cached as parquet in data/favorita/.
    """
    stores = select_stores()
    print("Stores:\n" + stores[["store_nbr", "city", "state", "type", "transactions", "reason"]].to_string(index=False))
    rows, cleaning = clean(load_subset(stores["store_nbr"].tolist()))
    keep, coverage = top_items(rows)
    info = {"stores": stores.to_dict(orient="records"), "cleaning": cleaning, "items": coverage}
    path = features_path(stores["store_nbr"].tolist())
    if path.exists():
        df = pd.read_parquet(path)
        info["cleaning"].update(_grid_stats(path))
    else:
        grid, grid_stats = build_grid(rows, keep)
        info["cleaning"].update(grid_stats)
        df = add_features(grid, stores)
        # modelling rows: open store-days with the short lags available (as build_features.build)
        df = df[(df["store_open"] == 1)].dropna(subset=[f"lag_{3 * LEAD_TIME_DAYS}", f"roll_std_14_lag{LEAD_TIME_DAYS}"])
        df = df.reset_index(drop=True)
        df.to_parquet(path, index=False)
        _grid_stats(path, grid_stats)
    info["cleaning"]["modelling_rows"] = len(df)
    info["cleaning"]["negative_sales_after_cleaning"] = int((df[TARGET] < 0).sum())
    return df, info


def _grid_stats(features_file, stats=None):
    """Grid counts saved next to the cached features (written once, read on later runs)."""
    import json
    side = features_file.with_suffix(".grid.json")
    if stats is not None:
        side.write_text(json.dumps(stats))
        return stats
    return json.loads(side.read_text()) if side.exists() else {}
