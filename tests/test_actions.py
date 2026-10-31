"""
Overstock actions (src/actions.py): discount math and its 40% cap, transfer
pairing and quantities, no transfer when no other city is short, the
reduce-order fallback, the vendor email's overstock column, and checks on
the generated data.
"""

import numpy as np
import pandas as pd
import pytest

import actions
import alerts
from config import DATA_DIR, PRICE_ELASTICITY
from test_alerts import fake_project  # noqa: F401  (pytest fixture)


def rows(*records):
    """
    Risk rows: (city, sku_id, category, opening_stock, forecast_daily,
    risk_status, reorder_qty[, days_until_delivery = 3]). Dairy and sweets are
    delivered every 3 / 4 days, so their stock keeps days_until_delivery more
    days (shelf left = days_until_delivery - cycle + shelf life).
    """
    records = [record + (3,) * (len(record) == 7) for record in records]
    return pd.DataFrame(records, columns=["city", "sku_id", "category", "opening_stock", "forecast_daily",
                                          "risk_status", "reorder_qty", "days_until_delivery"])


def readiness(*records):
    """Readiness rows: (city, sku_id, status, shortfall)."""
    return pd.DataFrame(records, columns=["city", "sku_id", "status", "shortfall"])


def suggest(risk, ready=None, week_start=2):
    """Actions, with the festival week starting `week_start` days after the stock count."""
    return actions.overstock_actions(risk, actions.shortfalls(risk, ready, "Diwali", week_start)).set_index(["city", "sku_id"])


def test_sell_window_is_shelf_life_for_perishables_and_the_overstock_threshold_otherwise():
    assert actions.sell_window("dairy") == 3 and actions.sell_window("sweets") == 4
    assert actions.sell_window("snacks") == 7 + 2 + 180  # classify_risk: horizon + 2 + shelf life
    assert actions.excess_units(100, 10, 3) == 70 and actions.excess_units(20, 10, 3) == 0


def test_discount_is_the_smallest_5pct_step_that_sells_the_stock_in_time():
    # stock 50, 10/day, 4 days: needs demand x1.25; 25% off gives x1.22, 30% off x1.28
    assert actions.demand_lift(0.25) < 1.25 < actions.demand_lift(0.30)
    assert actions.demand_lift(0.30) == pytest.approx(0.7 ** -PRICE_ELASTICITY)
    discount, days = actions.discount_plan(50, 10, 4)
    assert discount == 0.30
    assert days == 4  # 50 / (10 x 1.28) = 3.9 days, rounded up
    assert actions.discount_plan(41, 10, 4) == (0.05, 4)  # x1.04 is enough
    assert actions.discount_plan(50, 0, 4) is None


def test_discount_is_capped_at_40pct():
    assert actions.demand_lift(0.40) == pytest.approx(1.43, abs=0.01)
    assert actions.discount_plan(57, 10, 4) == (0.40, 4)  # needs x1.425: exactly the cap
    assert actions.discount_plan(60, 10, 4) is None  # needs x1.5


def test_transfer_pairs_largest_excess_with_largest_shortfall_and_never_exceeds_the_excess():
    risk = rows(("Delhi", "A", "dairy", 100, 10, "overstock", 0),      # excess 70, keeps 3 days
                ("Mumbai", "A", "dairy", 60, 10, "overstock", 0),      # excess 30
                ("Chennai", "A", "dairy", 2, 10, "stockout", 50, 1),   # short 50, next delivery in 1 day
                ("Bengaluru", "A", "dairy", 40, 10, "normal", 0))
    ready = readiness(("Bengaluru", "A", "Order now", 40.0), ("Chennai", "A", "At risk", 500.0))
    out = suggest(risk, ready)
    delhi, mumbai = out.loc[("Delhi", "A")], out.loc[("Mumbai", "A")]
    assert (delhi["action_type"], delhi["action"], delhi["transfer_units"]) == ("transfer", "Move 50 units to Chennai", 50)
    assert "50 of 70 excess units" in delhi["action_detail"] and "stockout risk" in delhi["action_detail"]
    # Chennai's shortfall is filled; Mumbai's 30 go to Bengaluru's 40 ("Order now"), capped at its excess
    assert (mumbai["action"], mumbai["transfer_to"], mumbai["transfer_units"]) == ("Move 30 units to Bengaluru", "Bengaluru", 30)
    assert "Diwali “Order now”" in mumbai["action_detail"]

    big = suggest(rows(("Delhi", "B", "dairy", 100, 10, "overstock", 0), ("Chennai", "B", "dairy", 1, 10, "stockout", 500, 1)))
    assert big.loc[("Delhi", "B"), "transfer_units"] == 70  # min(excess 70, shortfall 500)


def test_freshness_blocks_a_stockout_transfer_the_stock_would_not_survive():
    # stock keeps 2 more days (sellable today and tomorrow); Chennai needs it until its delivery in 1 day,
    # after 1 day in transit: shelf left 2 > 1 + 1 fails (needs to be strictly more)
    blocked = suggest(rows(("Delhi", "A", "dairy", 100, 10, "overstock", 0, 2),
                           ("Chennai", "A", "dairy", 1, 10, "stockout", 50, 1))).loc[("Delhi", "A")]
    assert blocked["action_type"] == "write_off" and pd.isna(blocked["transfer_to"])
    assert "Transfer ruled out: stock here is sellable until day 1; Chennai (stockout risk) needs it sellable on day 2" in blocked["action_detail"]
    allowed = suggest(rows(("Delhi", "A", "dairy", 100, 10, "overstock", 0, 3),
                           ("Chennai", "A", "dairy", 1, 10, "stockout", 50, 1))).loc[("Delhi", "A")]
    assert allowed["action"] == "Move 50 units to Chennai"  # 3 > 1 + 1


def test_freshness_blocks_order_now_when_stock_expires_before_the_festival_week():
    risk = rows(("Delhi", "A", "sweets", 200, 10, "overstock", 0, 3),  # keeps 3 days: days 0-2
                ("Mumbai", "A", "sweets", 200, 10, "overstock", 0, 4),  # keeps 4 days: days 0-3
                ("Chennai", "A", "sweets", 20, 10, "normal", 0))
    ready = readiness(("Chennai", "A", "Order now", 500.0))
    out = suggest(risk, ready, week_start=3)  # festival week starts on day 3
    assert out.loc[("Mumbai", "A"), "action"] == "Move 160 units to Chennai"  # excess 200 - 10 x 4
    delhi = out.loc[("Delhi", "A")]
    assert delhi["action_type"] == "write_off"
    assert "Chennai (Diwali “Order now”) needs it sellable on day 3" in delhi["action_detail"]
    # a fresh, smaller shortfall elsewhere is used instead of the stale, larger one
    risk = rows(("Delhi", "A", "sweets", 200, 10, "overstock", 0, 3),
                ("Mumbai", "A", "sweets", 1, 10, "stockout", 20, 1))
    assert suggest(risk, ready, week_start=3).loc[("Delhi", "A"), "action"] == "Move 20 units to Mumbai"


def test_no_transfer_when_the_other_city_is_not_short():
    risk = rows(("Delhi", "A", "snacks", 2100, 10, "overstock", 0),
                ("Chennai", "A", "snacks", 50, 10, "normal", 0),
                ("Mumbai", "B", "snacks", 1, 10, "stockout", 30))  # short, but a different product
    ready = readiness(("Chennai", "A", "At risk", 80.0), ("Chennai", "A", "Ready", 0.0), ("Delhi", "A", "Order now", 90.0))
    row = suggest(risk, ready).loc[("Delhi", "A")]
    assert row["action_type"] == "discount" and pd.isna(row["transfer_to"])
    # excess 2,100 - 10 x 189 = 210: needs x1.11 -> 15% off (x1.12), 2,100 / 11.2 = 188 days
    assert (row["action"], row["discount_pct"], row["discount_days"], row["excess_units"]) == ("15% off for 188 days", 15, 188, 210)


def test_perishable_excess_is_an_expected_write_off_with_the_right_order_size():
    risk = rows(("Delhi", "A", "dairy", 100, 10, "overstock", 0)).assign(recent_avg_order=80.0)
    row = suggest(risk).loc[("Delhi", "A")]
    # excess 100 - 10 x 3 = 70 expires; right order = 10/day x (3-day cycle + 2 safety days) = 50,
    # as the reorder rule orders; recent deliveries averaged 80
    assert (row["action_type"], row["reduce_units"], row["normal_order"], row["recent_avg_order"]) == ("write_off", 70, 50, 80)
    assert row["action"] == "Expected write-off ≈70 units (expires before it can sell) · order ~50 per delivery instead of ~80 from now on"
    assert row["action_detail"] == ("Mark down or donate before expiry · Discount alone won't clear it: "
                                    "needs ~3.3× normal sales before expiry; 40% off gives ~1.4×")
    assert pd.isna(row["skip_orders"])  # never "skip orders" for perishables
    # without an order history (no recent_avg_order), only the right size
    sweets = suggest(rows(("Delhi", "A", "sweets", 100, 10, "overstock", 0))).loc[("Delhi", "A")]
    assert sweets["action"] == "Expected write-off ≈60 units (expires before it can sell) · order ~60 per delivery from now on"  # 10 x (4 + 2)
    assert actions.write_off_text(0, 60).startswith("Expected write-off ≈0 units")


def test_non_perishables_skip_whole_orders_when_40pct_off_is_not_enough():
    # snacks: sell window 7 + 2 + 180 = 189 days; 3,000 units at 10/day needs x1.59 (> x1.43)
    row = suggest(rows(("Delhi", "A", "snacks", 3000, 10, "overstock", 0))).loc[("Delhi", "A")]
    assert (row["action_type"], row["reduce_units"], row["skip_orders"]) == ("reduce_order", 1110, 12)
    # a normal order is 10/day x (7-day cycle + 2 safety days) = 90; 1,110 / 90 = 12.3 whole orders
    assert row["action"] == "Skip the next 12 orders (≈1,110 units excess; a normal order is ~90)"
    assert row["action_detail"] == "Discount alone won't clear it: needs ~1.6× normal sales within 189 days; 40% off gives ~1.4×"
    assert actions.reduce_text(95, 70) == ("Skip the next order (≈95 units excess; a normal order is ~70)", 1)
    assert actions.reduce_text(20, 70) == ("Reduce next order by 20 units", 0)  # less than one normal order


def test_recent_average_order_is_the_stock_on_delivery_mornings(tmp_path, monkeypatch):
    import config
    from build_bigbasket_regional_risk import recent_avg_order

    # regional simulation: dairy delivered every 3 days (days_until_delivery == 3 on delivery mornings)
    opening = np.array([[90.0, 60, 30, 120, 80, 40], [500, 400, 300, 200, 100, 50]])
    until = np.array([[3, 2, 1, 3, 2, 1], [7, 6, 5, 4, 3, 2]])
    avg = recent_avg_order(opening, until, pd.Series(["dairy", "snacks"]))
    assert avg[0] == 105.0 and np.isnan(avg[1])  # (90 + 120) / 2; non-perishables: none

    # model scope: POS history + this morning's stock, last 28 days only
    counted = pd.Timestamp("2026-10-30")
    pos = pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-10-26", "2026-10-27", "2026-10-29"]),
                        "city": "Delhi", "sku_id": "A", "category": "dairy",
                        "opening_stock": [999, 60, 40, 90], "days_until_delivery": [3, 3, 2, 3]})
    pos.to_csv(tmp_path / "pos_synthetic.csv", index=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    today = pd.DataFrame({"date": [counted], "city": "Delhi", "sku_id": "A", "category": "dairy",
                          "opening_stock": [300], "days_until_delivery": [1]})
    orders = actions.model_recent_orders(today, counted)
    assert orders.to_dict("records") == [{"city": "Delhi", "sku_id": "A", "recent_avg_order": 75.0}]  # (60 + 90) / 2


def test_vendor_email_shows_the_action_in_the_overstock_section(fake_project, monkeypatch):
    monkeypatch.setattr(alerts, "load_actions", lambda city: {"SKUBeda019": "Move 12 units to Chennai · 12 of 20 excess units"})
    alert = alerts.build_alert("Bengaluru")
    overstock_html = alert["html"].split("OVERSTOCK")[1]
    assert "Suggested action (Estimate · simulated stock)" in overstock_html
    assert "Move 12 units to Chennai · 12 of 20 excess units" in overstock_html
    assert "Move 12 units to Chennai" in alert["text"] and "Estimate · simulated stock" in alert["text"]
    attachment = pd.read_csv(pd.io.common.StringIO(alert["csv"]))
    assert attachment.loc[attachment["sku_id"] == "SKUBeda019", "suggested_action"].iloc[0].startswith("Move 12 units")


@pytest.mark.parametrize("scope", ["model", "regional"])
def test_generated_actions_cover_every_overstock_and_transfers_fit(scope):
    """On the real data: one action per overstock row; transfers go to a short city and fit both sides."""
    import stockout_view

    if not (DATA_DIR / "bigbasket_regional_inventory.csv").exists():
        pytest.skip("pipeline outputs not generated; run run_pipeline.py first")
    stockout_view.clear_cache()
    risk, _ = stockout_view.raw_rows(scope)
    out = actions.scope_actions(scope)
    assert len(out) == (risk["risk_status"] == "overstock").sum()
    assert set(out["action_type"]) <= set(actions.TYPES)
    transfers = out[out["action_type"] == "transfer"]
    assert (transfers["transfer_units"] <= transfers["excess_units"]).all()
    assert (transfers["transfer_to"] != transfers["city"]).all()
    statuses = risk.set_index(["city", "sku_id"])["risk_status"]
    for _, row in transfers.iterrows():
        assert statuses.get((row["transfer_to"], row["sku_id"])) != "overstock"
