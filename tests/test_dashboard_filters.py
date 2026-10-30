"""
Stockout views on the admin dashboard (src/stockout_view.py and the
/api/stockout routes in dashboard.py), on a small hand-made data set:

  Delhi, tracked (sales history, 28 days before the planning day vs the 28 before):
    T1 Amul       10/day after 5/day   -> 280 units, rising +100%
    T2 Haldiram's 20/day flat          -> 560 units, trend 0
    T3 Haldiram's 15/day after 30/day  -> 420 units, falling -50%
    T4 Parle       2/day flat          ->  56 units, trend 0
  Delhi, analog (no history): A1 Amul, A2 Haldiram's, A3 Parle, plus A4, a
  second SKU of A1's product that must be shown once.
  Mumbai: one tracked product, so a city filter is visible.
"""

import csv
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

import dashboard
import festival_uplift
import product_detail
import stockout_view

PLANNING_DAY = pd.Timestamp("2026-08-26")
DIRECT, ANALOG = "lightgbm_direct", "lightgbm_analog_k5"
# sku, city, name, brand, category, price, forecast_daily, cover, source, risk
PRODUCTS = [
    ("T1", "Delhi", "Amul Lassi 200 ml", "Amul", "dairy", 25.0, 12.0, 1.5, DIRECT, "stockout"),
    ("T2", "Delhi", "Haldiram's Bhujia 400 g", "Haldiram's", "snacks", 110.0, 20.0, 3.0, DIRECT, "normal"),
    ("T3", "Delhi", "Haldiram's Soan Papdi 500 g", "Haldiram's", "sweets", 200.0, 14.0, 6.0, DIRECT, "normal"),
    ("T4", "Delhi", "Parle-G Gold 1 kg", "Parle", "snacks", 150.0, 2.0, 1.8, DIRECT, "stockout"),
    ("A1", "Delhi", "Amul Kool Cafe 200 ml", "Amul", "beverages", 40.0, 30.0, 2.0, ANALOG, "normal"),
    ("A2", "Delhi", "Haldiram's Kaju Katli 1 kg", "Haldiram's", "sweets", 900.0, 6.0, 9.0, ANALOG, "normal"),
    ("A3", "Delhi", "Parle Monaco 200 g", "Parle", "snacks", 50.0, 8.0, 4.0, ANALOG, "normal"),
    ("A4", "Delhi", "AMUL Kool Cafe, 200 ml", "Amul", "beverages", 40.0, 99.0, 0.5, ANALOG, "stockout"),
    ("T1", "Mumbai", "Amul Lassi 200 ml", "Amul", "dairy", 25.0, 9.0, 4.0, DIRECT, "normal"),
]
# per-day shape of the tracked forecasts (mean 1): T2 falls, the rest rise
RISING = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
FALLING = RISING[::-1]
DAILY_SALES = {("Delhi", "T1"): (10, 5), ("Delhi", "T2"): (20, 20), ("Delhi", "T3"): (15, 30),
               ("Delhi", "T4"): (2, 2), ("Mumbai", "T1"): (7, 7)}  # (last 28 days, the 28 before)


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    inventory = pd.DataFrame([{
        "region": city, "sku_id": sku, "product_name": name, "brand": brand, "pack_size_value": "200.0",
        "pack_size_unit": "ml", "category": category, "subcategory": "", "price": price, "forecast_daily": daily,
        "forecast_units": daily, "forecast_horizon": daily * 7, "model_category": category, "forecast_source": source, "days_of_cover": cover, "days_until_delivery": 2,
        "reorder_qty": int(daily * 9), "risk_status": risk, "opening_stock": round(daily * cover),
        "analog_ids": "" if source == DIRECT else "T1;T2;T3;T4;T1",
        "analog_weights": "" if source == DIRECT else "0.4;0.3;0.1;0.1;0.1",
        "analog_price_adj": "" if source == DIRECT else "1.0;1.2;0.8;2.0;0.5",
    } for sku, city, name, brand, category, price, daily, cover, source, risk in PRODUCTS])
    inventory.to_csv(tmp_path / "bigbasket_regional_inventory.csv", index=False)
    pd.DataFrame({"forecast_start": [PLANNING_DAY.date().isoformat()]}).to_csv(
        tmp_path / "bigbasket_regional_forecast.csv", index=False)
    pd.DataFrame([{"sku_id": sku, "city": city, "date": (PLANNING_DAY + pd.Timedelta(days=d)).date().isoformat(),
                   "prediction": daily * (FALLING if sku == "T2" else RISING)[d]}
                  for sku, city, _, _, _, _, daily, _, source, _ in PRODUCTS if source == DIRECT for d in range(7)]).to_csv(
        tmp_path / "bigbasket_tracked_daily_forecast.csv", index=False)
    pd.DataFrame([("Diwali", city, category, uplift, 5, 2021, 2025) for city in ["Delhi", "Mumbai"]
                  for category, uplift in [("sweets", 3.0), ("snacks", 1.5), ("dairy", 2.0)]],
                 columns=["festival", "city", "category", "uplift", "occurrences", "first_year", "last_year"]).to_csv(
        tmp_path / "festival_uplift.csv", index=False)
    pos = [{"date": (PLANNING_DAY - pd.Timedelta(days=age)).date().isoformat(), "city": city, "sku_id": sku,
            "units_sold": recent if age <= 28 else previous, "true_demand": (recent if age <= 28 else previous) + 0.5,
            "stockout_flag": int(age == 10)}
           for (city, sku), (recent, previous) in DAILY_SALES.items() for age in range(0, 70)]
    pd.DataFrame(pos).to_csv(tmp_path / "pos_synthetic.csv", index=False)
    pd.DataFrame({"date": ["2099-01-01"], "next_festival": ["Diwali"], "days_until_next_festival": ["0"]}).to_csv(
        tmp_path / "festival_calendar_5yr.csv", index=False)
    for module in (stockout_view, dashboard, product_detail, festival_uplift):
        monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "OUTPUTS_DIR", tmp_path)
    festival_uplift._UPLIFTS.clear()
    monkeypatch.setattr(stockout_view, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "REGIONAL_INTELLIGENCE_CACHE", None)
    stockout_view.clear_cache()
    product_detail.clear_cache()
    yield tmp_path
    stockout_view.clear_cache()
    product_detail.clear_cache()
    festival_uplift._UPLIFTS.clear()


@pytest.fixture
def server(fake_data):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(base, path, **params):
    query = urllib.parse.urlencode({"scope": "regional", "city": "Delhi", **params})
    with urllib.request.urlopen(f"{base}{path}?{query}", timeout=30) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if response.headers["Content-Type"] == "application/json" else body


def skus(payload):
    return [row["sku_id"] for row in payload["rows"]]


def test_history_columns_and_one_row_per_product(server):
    payload = get(server, "/api/stockout")
    assert payload["total"] == payload["products_in_city"] == 7  # A4 duplicates A1 (direct-less: lowest SKU id kept)
    rows = {row["sku_id"]: row for row in payload["rows"]}
    assert "A4" not in rows
    t1 = rows["T1"]
    assert (t1["units_7d"], t1["units_28d"], t1["avg_daily_sales"], t1["trend_pct"]) == (70, 280, 10.0, 100.0)
    a1 = rows["A1"]  # analog: no history, the forecast stands in
    assert a1["has_history"] is False and a1["units_28d"] is None and a1["trend_pct"] is None
    assert a1["forecast_daily"] == 30.0


def test_brand_filter_is_case_insensitive_and_exact(server):
    assert sorted(skus(get(server, "/api/stockout", brand="haldiram's"))) == ["A2", "T2", "T3"]
    assert get(server, "/api/stockout", brand="Haldiram")["total"] == 0


def test_price_range_is_inclusive_and_combines_with_other_filters(server):
    assert sorted(skus(get(server, "/api/stockout", min_price="50", max_price="200"))) == ["A3", "T2", "T3", "T4"]
    assert skus(get(server, "/api/stockout", min_price="500.01")) == ["A2"]
    assert sorted(skus(get(server, "/api/stockout", min_price="50", max_price="200", category="snacks"))) == ["A3", "T2", "T4"]


def test_max_cover(server):
    assert sorted(skus(get(server, "/api/stockout", max_cover="2"))) == ["A1", "T1", "T4"]


def test_top_sellers_by_units_sold_in_28_days(server):
    # 4 Delhi products have history: T2 560, T3 420, T1 280, T4 56 units
    assert skus(get(server, "/api/stockout", top="25")) == ["T2"]
    assert sorted(skus(get(server, "/api/stockout", top="50"))) == ["T2", "T3"]
    assert skus(get(server, "/api/stockout", top="25", city="Mumbai")) == ["T1"]  # ranked within the city


def test_sort_order(server):
    # default: lowest cover first, then fastest daily rate
    assert skus(get(server, "/api/stockout")) == ["T1", "T4", "A1", "T2", "A3", "T3", "A2"]
    # analog products have no units sold, so they go last either way
    assert skus(get(server, "/api/stockout", sort="units_28d"))[:4] == ["T2", "T3", "T1", "T4"]
    assert skus(get(server, "/api/stockout", sort="units_28d", order="asc"))[:4] == ["T4", "T1", "T3", "T2"]
    assert skus(get(server, "/api/stockout", sort="price", order="desc")) == ["A2", "T3", "T4", "T2", "A3", "A1", "T1"]
    assert skus(get(server, "/api/stockout", sort="cover", limit="2", offset="1")) == ["T4", "A1"]


def test_trend_filter(server):
    assert skus(get(server, "/api/stockout", trend="rising")) == ["T1"]
    assert skus(get(server, "/api/stockout", trend="falling")) == ["T3"]
    assert skus(get(server, "/api/stockout", sort="trend"))[:2] == ["T1", "T4"]  # T2 and T4 tie at 0%: lower cover first


def test_source_and_text_search(server):
    assert sorted(skus(get(server, "/api/stockout", source="analog"))) == ["A1", "A2", "A3"]
    assert skus(get(server, "/api/stockout", q="kool CAFE")) == ["A1"]


def test_csv_has_exactly_the_filtered_sorted_rows(server):
    params = {"brand": "Haldiram's", "sort": "price", "order": "asc", "limit": "1"}
    payload = get(server, "/api/stockout", **params)
    rows = list(csv.DictReader(io.StringIO(get(server, "/api/stockout", format="csv", **params))))
    assert len(rows) == payload["total"] == 3  # every row, not just the page
    assert [row["sku_id"] for row in rows] == ["T2", "T3", "A2"]
    assert set(rows[0]) == set(stockout_view.EXPORT_COLUMNS)


def test_unknown_city_and_bad_parameters_are_json_errors(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        get(server, "/api/stockout", city="Atlantis")
    assert error.value.code == 404
    assert error.value.headers["Content-Type"] == "application/json"
    assert json.loads(error.value.read()) == {"error": "Unknown city", "name": "Atlantis"}
    for params in [{"scope": "everything"}, {"sort": "name"}, {"trend": "up"}, {"min_price": "cheap"}]:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(server, "/api/stockout", **params)
        assert error.value.code == 400


def test_brand_view(server):
    payload = get(server, "/api/stockout/brands", brand="Parle")  # the brand filter itself is ignored here
    brands = {row["brand"]: row for row in payload["rows"]}
    assert [row["brand"] for row in payload["rows"]] == ["Haldiram's", "Amul", "Parle"]  # by units
    haldiram = brands["Haldiram's"]
    # 560 + 420 sold, plus A2's 28 x 6 estimated; revenue = units x MRP
    assert haldiram["units_28d"] == 560 + 420 + 168
    assert haldiram["revenue_28d"] == 560 * 110 + 420 * 200 + 168 * 900
    assert (haldiram["products"], haldiram["estimated_products"], haldiram["stockout_risk"]) == (3, 1, 0)
    assert brands["Amul"]["stockout_risk"] == 1 and brands["Amul"]["avg_cover"] == 1.75
    assert {row["category"] for row in payload["categories"]} == {"dairy", "snacks", "sweets", "beverages"}
    assert get(server, "/api/stockout/brands", category="sweets")["total"] == 1


def test_occasion_list_ranks_by_units_or_revenue(fake_data):
    [delhi, _] = dashboard.regional_intelligence()
    by_units = [row["sku_id"] for row in delhi["demand"]]
    by_revenue = [row["sku_id"] for row in delhi["demand_by_revenue"]]
    # ranked by the plain 7-day forecast (no category boost); A4 (99/day) is a
    # second SKU of A1, so it never shows
    assert by_units[:3] == ["A1", "T2", "T3"] and "A4" not in by_units + by_revenue
    assert by_revenue[:4] == ["A2", "T3", "T2", "A1"]  # 7 x (6 x ₹900, 14 x ₹200, 20 x ₹110, 30 x ₹40)
    assert all(row["revenue_estimate"] == round(row["forecast_7d"] * float(row["price"]), 2)
               for row in delhi["demand_by_revenue"])
    assert delhi["demand"][0]["forecast_7d"] == 210.0
    assert len(by_revenue) == len({dashboard.product_key(row) for row in delhi["demand_by_revenue"]})


def write_serving_stock(fake_data, rows):
    """Serving opening stock (city, sku, stock, days until delivery) for the planning day, plus the product master."""
    pd.DataFrame([{"date": PLANNING_DAY.date().isoformat(), "city": city, "sku_id": sku, "opening_stock": stock,
                   "days_until_delivery": until} for city, sku, stock, until in rows]).to_csv(
        fake_data / "serving_opening_stock.csv", index=False)
    pd.DataFrame([("T1", "Amul Lassi 200 ml", "Amul", "dairy", 25.0, "200.0", "ml"),
                  ("T2", "Haldiram's Bhujia 400 g", "Haldiram's", "snacks", 110.0, "400.0", "g")],
                 columns=["sku_id", "product_name", "brand", "category", "mrp", "pack_size_value", "pack_size_unit"]).to_csv(
        fake_data / "sampled_products.csv", index=False)


def test_model_scope_uses_the_shared_risk_rule(fake_data):
    # serving: stock counted on the planning day vs the mean of its 7 daily forecasts (12 and 20 a day)
    write_serving_stock(fake_data, [("Delhi", "T1", 15, 3), ("Delhi", "T2", 100, 3)])
    table = stockout_view.table_for("model")
    assert table.attrs["planning_day"] == PLANNING_DAY.date().isoformat()
    rows = table.set_index("sku_id")
    assert rows.loc["T1", "risk_status"] == "stockout" and rows.loc["T1", "days_of_cover"] == 1.25  # 15 / 12 < 3 days
    assert rows.loc["T2", "risk_status"] == "normal" and rows.loc["T2", "reorder_qty"] == 80  # 7x20 + 2x20 - 100
    assert rows.loc["T1", "units_28d"] == 280 and rows.loc["T1", "price"] == 25.0
