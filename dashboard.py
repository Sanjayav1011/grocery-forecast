import csv
import io
import json
import re
import sys
import threading
import zipfile
from datetime import date, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import PROJECT_DIR, ARCHIVE_PATH, ARCHIVE_CSV, DATA_DIR, OUTPUTS_DIR, MODELS_DIR, LEAD_TIME_DAYS
import explain_forecast
import festival_readiness
import festival_uplift
import product_detail
import stockout_view
from stockout_view import product_key, unique_products  # noqa: F401  (product_key kept for callers)
CATALOG_CACHE = None
STOREFRONT_CATALOG_CACHE = None
REGIONAL_INTELLIGENCE_CACHE = None
NOT_FORECAST_FILE = "bigbasket_not_forecast.csv"  # written by src/forecast_regional.py
# the model risk report for the serving date (src/serving_risk_report.py); the evaluation
# reports (stockout_risk_report.csv, 1 Apr - 25 Aug 2026) only feed accuracy figures
SERVING_REPORTS = {"stockout": "serving_stockout_risk_report.csv", "overstock": "serving_overstock_risk_report.csv"}


def dataset_paths():
    paths = {}
    for folder in (DATA_DIR, OUTPUTS_DIR):
        for path in folder.glob("*.csv"):
            paths[path.name] = path
    return paths


def dataset_info():
    info = []
    for name, path in sorted(dataset_paths().items()):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            columns = next(reader, [])
            rows = sum(1 for _ in reader)
        info.append({"name": name, "rows": rows, "columns": columns, "download": f"/download/{quote(name)}"})
    return info


def dataset_preview(name, offset, limit):
    path = dataset_paths().get(name)
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for index, row in enumerate(reader):
            if index < offset:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def data_date():
    """
    The last day of sales data (the day the regional forecast was made from).
    Every date on the admin page counts from it, never the computer's clock.
    """
    path = DATA_DIR / "bigbasket_regional_forecast.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            first = next(csv.DictReader(file), None)
        if first:
            return date.fromisoformat(first["forecast_start"]) - timedelta(days=1)
    return None


def next_festival(occasions, today):
    """The first festival on or after `today` (rows with days_until_next_festival == 0 are festival days)."""
    if today is None:
        return None
    upcoming = [row for row in occasions if row["date"] >= today.isoformat() and row["days_until_next_festival"] == "0"]
    festival = min(upcoming, key=lambda row: row["date"]) if upcoming else None
    return festival and {"name": festival["next_festival"], "date": festival["date"],
                         "days_until": (date.fromisoformat(festival["date"]) - today).days}


def not_forecast_count():
    path = DATA_DIR / NOT_FORECAST_FILE
    return len(read_csv_from_path(path)) if path.exists() else None


def regional_intelligence():
    global REGIONAL_INTELLIGENCE_CACHE
    if REGIONAL_INTELLIGENCE_CACHE is not None:
        return REGIONAL_INTELLIGENCE_CACHE
    rows = read_csv_from_path(DATA_DIR / "bigbasket_regional_inventory.csv")
    today = data_date()
    festival = next_festival(read_csv_from_path(DATA_DIR / "festival_calendar_5yr.csv"), today)
    window = [(today + timedelta(days=offset)).isoformat() if today else None for offset in (1, LEAD_TIME_DAYS)]
    result = []
    excluded = not_forecast_count()
    for region in sorted({row["region"] for row in rows}):
        region_rows = [row for row in rows if row["region"] == region]
        stockout = sorted((row for row in region_rows if row["risk_status"] == "stockout"), key=lambda row: float(row["days_of_cover"]))[:10]
        overstock = sorted((row for row in region_rows if row["risk_status"] == "overstock"), key=lambda row: float(row["days_of_cover"]), reverse=True)[:10]
        for row in region_rows:
            # the next 7 days of forecast, as forecast_regional saved it (forecast_daily x 7)
            row["forecast_7d"] = float(row["forecast_horizon"])
            price = float(row["price"]) if row.get("price") else None
            row["revenue_estimate"] = round(row["forecast_7d"] * price, 2) if price else None
        products = unique_products(region_rows)
        demand = sorted(products, key=lambda row: row["forecast_7d"], reverse=True)[:10]
        # revenue = 7-day forecast units x MRP; products without an MRP can't rank
        demand_by_revenue = sorted((row for row in products if row["revenue_estimate"] is not None),
                                   key=lambda row: row["revenue_estimate"], reverse=True)[:10]
        for row in stockout:
            row["recommended_restock_units"] = int(row["reorder_qty"])  # shared rule, inventory.classify_risk
        result.append({
            "region": region,
            "sku_count": len(region_rows),
            "stockout_count": sum(row["risk_status"] == "stockout" for row in region_rows),
            "overstock_count": sum(row["risk_status"] == "overstock" for row in region_rows),
            "data_date": today.isoformat() if today else None,
            "window_start": window[0],
            "window_end": window[1],
            "next_festival": festival,
            "stockout": stockout,
            "overstock": overstock,
            "demand": demand,
            "demand_by_revenue": demand_by_revenue,
            "not_forecast_count": excluded,
            "not_forecast_download": f"/download/{quote(NOT_FORECAST_FILE)}",
        })
    REGIONAL_INTELLIGENCE_CACHE = result
    return result


def dashboard_payload(city):
    """
    /api/dashboard: the serving model risk rows (all cities or one), the next
    festival and the KPIs. Accuracy (metrics.json) is from the evaluation
    period, a separate, earlier window: evaluation_period says which.
    """
    city = city.strip().casefold()
    stockout = read_csv(SERVING_REPORTS["stockout"])
    overstock = read_csv(SERVING_REPORTS["overstock"])
    latest = max((row["date"] for row in stockout + overstock), default=None)  # the day stock is counted
    stockout = [row for row in stockout if row["date"] == latest]
    overstock = [row for row in overstock if row["date"] == latest]
    if city and city != "all":
        stockout = [row for row in stockout if row["city"].casefold() == city]
        overstock = [row for row in overstock if row["city"].casefold() == city]
    today = data_date()
    metrics = read_json("metrics.json")
    return {
        "metrics": metrics, "stockout": stockout, "overstock": overstock,
        "latest": latest, "data_date": today.isoformat() if today else None,
        "evaluation_period": metrics.get("final_test_window"),
        "next_festival": next_festival(read_csv_from_path(DATA_DIR / "festival_calendar_5yr.csv"), today),
        "city": city or "all", "tracked_products": tracked_product_count(),
    }


def read_csv(filename):
    with (OUTPUTS_DIR / filename).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_csv_from_path(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(filename):
    with (OUTPUTS_DIR / filename).open(encoding="utf-8") as file:
        return json.load(file)


def tracked_product_count():
    # Products the forecast model covers, from the encodings saved by build_features.py
    path = MODELS_DIR / "categorical_mappings.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return len(json.load(file).get("sku_id", {}))


def read_catalog():
    global CATALOG_CACHE
    if CATALOG_CACHE is not None:
        return CATALOG_CACHE

    # Preferred source: the extracted BigBasket catalog (~57k products).
    if ARCHIVE_CSV.exists():
        with ARCHIVE_CSV.open(newline="", encoding="utf-8-sig", errors="replace") as file:
            rows = csv.DictReader(file)
            CATALOG_CACHE = []
            for row in rows:
                name = str(row.get("SKU Name", "")).strip()
                if not name or name == "nan":
                    continue
                category = str(row.get("Category", "")).title()
                subcategory = str(row.get("Sub-Category", "")).title()
                if category == "Kitchen, Garden & Pets" and subcategory == "Electronics & Devices":
                    category = "Electronics"
                CATALOG_CACHE.append({
                    "category": category,
                    "subcategory": subcategory,
                    "subsubcategory": str(row.get("Sub-sub-Category", "")).title(),
                    "brand": str(row.get("Brand", "")).strip(),
                    "name": name,
                    "size": str(row.get("SKU Size", "")).strip(),
                    "mrp": str(row.get("MRP", "")).strip(),
                    "ean": str(row.get("EAN Code", "")).strip(),
                    "description": str(row.get("About the Product", "")).strip(),
                    "image": str(row.get("Image Link", "")).strip(),
                    "link": str(row.get("Product Link", "")).strip(),
                })
        return CATALOG_CACHE

    if not ARCHIVE_PATH.exists():
        # Last resort: synthetic SKUs, so the dashboard still works without a real catalog.
        pos_path = DATA_DIR / "pos_synthetic.csv"
        CATALOG_CACHE = []
        if pos_path.exists():
            with pos_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            seen = set()
            for row in rows:
                sku_id = row["sku_id"]
                if sku_id in seen:
                    continue
                seen.add(sku_id)
                CATALOG_CACHE.append({
                    "category": row["category"].title(),
                    "subcategory": "",
                    "brand": "Synthetic catalog",
                    "name": f"{row['category'].title()} {sku_id}",
                    "size": "",
                    "mrp": row["unit_price"],
                    "image": "",
                    "link": "",
                })
        return CATALOG_CACHE

    products = []
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        files = [name for name in archive.namelist() if name.lower().endswith("product_data.xlsx")]
        import pandas as pd
        for name in files:
            frame = pd.read_excel(io.BytesIO(archive.read(name)))
            for _, row in frame.iterrows():
                products.append({
                    "category": str(row.get("Category", "")).title(),
                    "subcategory": str(row.get("Sub-Category", "")).title(),
                    "subsubcategory": str(row.get("Sub-sub-Category", "")).title(),
                    "brand": str(row.get("Brand", "")).strip(),
                    "name": str(row.get("SKU Name", "")).strip(),
                    "size": str(row.get("SKU Size", "")).strip(),
                    "mrp": str(row.get("MRP", "")).strip(),
                    "ean": str(row.get("EAN Code", "")).strip(),
                    "description": str(row.get("About the Product", "")).strip(),
                    "image": str(row.get("Image Link", "")).strip(),
                    "link": str(row.get("Product Link", "")).strip(),
                })
    CATALOG_CACHE = [product for product in products if product["name"] and product["name"] != "nan"]
    synthetic_path = DATA_DIR / "pos_synthetic.csv"
    if synthetic_path.exists():
        with synthetic_path.open(newline="", encoding="utf-8") as file:
            synthetic_rows = csv.DictReader(file)
            seen_skus = set()
            for row in synthetic_rows:
                if row["category"] not in {"fruits", "vegetables"} or row["sku_id"] in seen_skus:
                    continue
                seen_skus.add(row["sku_id"])
                CATALOG_CACHE.append({
                    "category": row["category"].title(),
                    "subcategory": "Fresh produce",
                    "subsubcategory": "",
                    "brand": "Demo fresh produce",
                    "name": f"Fresh {row['category'].title()} {row['sku_id']}",
                    "size": "",
                    "mrp": row["unit_price"],
                    "ean": row["sku_id"],
                    "description": "Synthetic fresh-produce catalog item",
                    "image": "",
                    "link": "",
                })
    return CATALOG_CACHE


def catalog_page(offset=0, limit=24, search="", category=""):
    global STOREFRONT_CATALOG_CACHE
    terms = search.casefold().split()
    category_target = " ".join(re.findall(r"[a-z0-9]+", category.casefold()))
    products = []
    total = 0
    catalog = read_catalog()
    if not terms and not category_target:
        if STOREFRONT_CATALOG_CACHE is None:
            category_order = (
                "foodgrains", "bakery", "beverages", "snacks", "eggs",
                "gourmet", "fruits", "baby", "cleaning", "beauty",
                "pharmacy", "kitchen", "food court", "fashion", "electronics",
            )

            def store_order(product):
                product_category = str(product.get("category", "")).casefold()
                return next(
                    (index for index, name in enumerate(category_order) if name in product_category),
                    len(category_order),
                )

            STOREFRONT_CATALOG_CACHE = sorted(catalog, key=store_order)
        return {
            "products": STOREFRONT_CATALOG_CACHE[offset:offset + limit],
            "total": len(STOREFRONT_CATALOG_CACHE),
        }

    for product in catalog:
        search_text = " ".join(
            str(product.get(field, "")) for field in
            ("name", "category", "subcategory", "subsubcategory", "brand", "ean")
        ).casefold()
        if not all(term in search_text for term in terms):
            continue
        if category_target:
            category_values = [
                " ".join(re.findall(r"[a-z0-9]+", str(product.get(field, "")).casefold()))
                for field in ("category", "subcategory", "subsubcategory", "name", "brand")
            ]
            if not any(
                value and (
                    value == category_target
                    or value in category_target
                    or category_target in value
                    or all(term in value for term in category_target.split())
                )
                for value in category_values
            ):
                continue
        if offset <= total < offset + limit:
            products.append(product)
        total += 1
    return {"products": products, "total": total}


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fresh Basket | Groceries delivered to your door</title>
  <style>
    :root { --bg:#f7faf7; --panel:#ffffff; --panel2:#f0f6f0; --line:#dce8dc; --text:#173b28; --muted:#66806f; --red:#d94b4b; --amber:#d38b00; --green:#1f8f4d; --orange:#f39a1e; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--text); background:linear-gradient(180deg, #eaf6eb 0, var(--bg) 260px); font-family:Arial, Helvetica, sans-serif; font-size:13px; }
    .shell { width:min(980px, 100%); margin:auto; padding:28px 26px 56px; }
    header { display:flex; justify-content:space-between; align-items:flex-start; gap:22px; margin-bottom:34px; }
    h1 { font-size:22px; font-weight:700; margin:0; letter-spacing:0; color:#126b36; }
    .subhead { color:var(--muted); font-size:11px; margin-top:7px; }
    .live { color:var(--muted); font-size:11px; white-space:nowrap; } .live i { display:inline-block; width:8px; height:8px; background:var(--green); border-radius:50%; margin-right:6px; }
    .summary { display:grid; grid-template-columns:repeat(4,1fr); gap:34px; margin:0 12px 30px; }
    .metric span { color:#aaa8a1; font-size:10px; display:block; margin-bottom:7px; } .metric strong { font-size:19px; font-weight:500; } .metric .red { color:var(--red); } .metric .amber { color:var(--amber); } .metric-note { display:block; color:var(--muted); font-size:10px; margin-top:6px; }
    .filters { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:18px; } .filter { border:1px solid #b9d5be; background:#fff; color:#28633d; border-radius:18px; padding:8px 14px; cursor:pointer; font-size:11px; } .filter.active { background:var(--green); color:#fff; border-color:var(--green); }
    .table-wrap { border:1px solid var(--line); border-radius:9px; overflow:hidden; background:var(--panel); } table { width:100%; border-collapse:collapse; } th { color:var(--muted); font-size:10px; font-weight:600; text-align:left; padding:9px 12px; background:#f0f6f0; } td { padding:11px 12px; border-top:1px solid var(--line); font-size:11px; } tr:hover { background:#f5fbf5; } .sku { font-weight:600; } .cover { color:var(--red); background:#ffe6e6; border-radius:10px; padding:4px 8px; white-space:nowrap; } .cover.high { color:var(--amber); background:#fff1cc; }
    .below { display:grid; grid-template-columns:1fr 300px; gap:32px; align-items:start; margin-top:14px; } .note { color:#dedcd6; font-family:Georgia, serif; line-height:1.55; font-size:14px; } .note code { color:#e9665d; font-family:monospace; font-size:12px; }
    .product { border:1px solid #363636; border-radius:9px; padding:12px; background:#171717; } .product-image { height:108px; display:grid; place-items:center; background:#111; border-radius:5px; color:#999; font-size:34px; margin-bottom:12px; } .product h2 { font:600 13px Arial, sans-serif; margin:0 0 4px; } .price { color:#aaa; font-size:11px; } .product .warning { color:var(--red); font-size:11px; margin:12px 0 9px; } .dot { display:inline-block; width:5px; height:5px; background:var(--red); border-radius:50%; vertical-align:middle; margin-right:4px; } .product button { width:100%; border:0; border-radius:6px; padding:7px; background:#f1f0ed; color:#111; font-size:11px; cursor:pointer; }
    .catalog-head { display:flex; justify-content:space-between; align-items:center; gap:14px; margin:34px 0 14px; } .catalog-head h2 { margin:0; font-size:20px; font-weight:700; color:#173b28; } .search { width:260px; border:1px solid #b9d5be; border-radius:6px; background:#fff; color:var(--text); padding:10px 12px; font-size:11px; } .catalog { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; } .card { min-width:0; border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; box-shadow:0 2px 8px rgba(32,83,46,.06); } .card-image { height:170px; display:grid; place-items:center; background:#f7faf7; border-radius:5px; overflow:hidden; color:#8ba391; font-size:25px; } .card-image a { display:block; width:100%; height:100%; } .card-image img { width:100%; height:100%; object-fit:contain; } .card h3 { font-size:12px; line-height:1.35; font-weight:600; margin:10px 0 8px; height:32px; overflow:hidden; } .card h3 a { color:var(--text); text-decoration:none; } .card p { color:var(--muted); font-size:10px; margin:0 0 7px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .card .buy { display:flex; justify-content:space-between; align-items:center; gap:6px; color:#173b28; font-size:13px; } .card .buy a { color:#fff; background:var(--green); border-radius:5px; padding:7px 10px; text-decoration:none; font-size:10px; } .empty { color:var(--muted); padding:22px; text-align:center; grid-column:1/-1; }
    .below, .summary, main > .table-wrap, .filters { display:none; } .live { display:none; }
    .category-menu { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; margin:24px 0 30px; } .category-panel { border:1px solid var(--line); border-radius:12px; background:linear-gradient(180deg, #ffffff 0%, #f5faf6 100%); padding:12px; box-shadow:0 4px 12px rgba(27,72,48,.05); min-height:132px; } .category-panel-header { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:10px; } .category-panel-title { font-size:12px; font-weight:700; color:#176b37; letter-spacing:.02em; } .category-mini { font-size:10px; color:var(--muted); } .category-links { display:flex; flex-wrap:wrap; gap:6px; } .category-pill { display:inline-block; padding:7px 9px; border-radius:999px; border:1px solid #d4e5d7; background:#fff; color:#245e39; font-size:10px; text-decoration:none; line-height:1.2; } .category-pill:hover { border-color:var(--green); color:var(--green); }
    .store-actions { display:flex; align-items:center; gap:8px; } .basket-button { border:0; border-radius:6px; background:var(--green); color:#fff; padding:10px 14px; cursor:pointer; font-size:11px; } .basket-count { display:inline-block; min-width:18px; padding:2px 5px; border-radius:10px; background:#fff; color:var(--green); margin-left:4px; }
    .basket-panel { position:fixed; z-index:5; top:0; right:0; width:min(380px,100%); height:100vh; overflow:auto; padding:22px; background:#fff; box-shadow:-8px 0 24px rgba(20,60,35,.18); transform:translateX(105%); transition:transform .2s ease; } .basket-panel.open { transform:translateX(0); } .basket-head { display:flex; justify-content:space-between; align-items:center; } .basket-head button, .payment button.close { border:0; background:transparent; font-size:20px; cursor:pointer; color:var(--muted); } .basket-item { display:grid; grid-template-columns:1fr auto; gap:8px; border-bottom:1px solid var(--line); padding:12px 0; font-size:11px; } .basket-item button { border:1px solid #b9d5be; background:#fff; border-radius:4px; cursor:pointer; } .basket-total { display:flex; justify-content:space-between; margin:18px 0; font-weight:700; } .checkout { width:100%; border:0; border-radius:6px; padding:11px; background:var(--green); color:#fff; cursor:pointer; } .payment { display:none; margin-top:16px; padding-top:16px; border-top:1px solid var(--line); } .payment.open { display:block; } .payment label { display:block; color:var(--muted); font-size:10px; margin:10px 0 4px; } .payment input { width:100%; border:1px solid #b9d5be; border-radius:5px; padding:9px; }
    .regional-intelligence, .model-risk { margin:0 0 30px; } .region-tabs { display:flex; flex-wrap:wrap; gap:7px; margin:14px 0; } .region-tab { border:1px solid #b9d5be; border-radius:18px; padding:8px 14px; background:#fff; color:#28633d; cursor:pointer; font-size:11px; } .region-tab.active { background:var(--green); color:#fff; } .region-summary { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:12px; } .region-metric { border:1px solid var(--line); border-radius:7px; background:#fff; padding:12px; } .region-metric span { display:block; color:var(--muted); font-size:10px; margin-bottom:5px; } .region-metric strong { font-size:16px; color:var(--text); } .region-metric.alert strong { color:var(--red); } .region-metric.warn strong { color:var(--amber); } .occasion { border:1px solid #cfe5d2; border-radius:7px; background:#eff9f0; padding:12px; margin-bottom:12px; color:#28633d; font-size:12px; } .regional-columns { display:grid; grid-template-columns:1fr 1fr; gap:12px; } .regional-panel { border:1px solid var(--line); border-radius:7px; background:#fff; overflow:hidden; } .regional-panel h3 { margin:0; padding:11px 12px; font-size:12px; color:var(--text); background:#f0f6f0; } .regional-panel table { min-width:0; } .regional-panel td, .regional-panel th { padding:8px 10px; } .regional-panel td { font-size:10px; } .regional-panel .risk { color:var(--red); } .regional-panel .safe { color:var(--green); }
    .stockout-view .view-tabs { display:flex; gap:6px; padding:10px 12px 0; } .stockout-view .filter { padding:6px 11px; font-size:10px; } .view-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:6px; padding:10px 12px; } .view-toolbar input, .view-toolbar select, .view-toolbar button, .view-toolbar a { border:1px solid #b9d5be; border-radius:6px; background:#fff; color:var(--text); padding:6px 8px; font-size:10px; text-decoration:none; cursor:pointer; } .view-toolbar input[type=number] { width:68px; } .view-toolbar [data-key=brand], .view-toolbar [data-key=q] { width:150px; } .view-count { margin-left:auto; color:var(--muted); font-size:10px; } .view-caption { padding:0 12px 8px; color:var(--muted); font-size:10px; line-height:1.4; } .stockout-view .table-wrap { overflow:auto; max-height:560px; border-width:1px 0; border-radius:0; } .stockout-view th { position:sticky; top:0; z-index:1; } .stockout-view table { min-width:940px; } th.sortable { cursor:pointer; white-space:nowrap; } th.sortable:hover { color:var(--green); } tr.brand-row { cursor:pointer; } .view-pager { display:flex; align-items:center; gap:10px; padding:8px 12px; color:var(--muted); font-size:10px; } .view-pager button { border:1px solid #b9d5be; border-radius:6px; background:#fff; color:var(--text); padding:5px 9px; cursor:pointer; font-size:10px; } .view-pager button:disabled { opacity:.5; cursor:default; } .rank-toggle { float:right; color:var(--muted); font-weight:400; font-size:10px; } .rank-toggle .filter { padding:3px 9px; font-size:10px; margin-left:4px; }
    .product-link { color:inherit; text-decoration:none; } .product-link:hover, .product-link:focus { color:var(--green); text-decoration:underline; }
    .product-panel { position:fixed; z-index:6; top:0; right:0; width:min(860px,100%); height:100vh; overflow:auto; padding:22px; background:#fff; box-shadow:-8px 0 24px rgba(20,60,35,.18); transform:translateX(105%); visibility:hidden; transition:transform .2s ease, visibility .2s; } .product-panel.open { transform:translateX(0); visibility:visible; } .product-panel-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; } .product-panel-head h2 { margin:0; font-size:16px; color:var(--text); } .product-panel-head button { border:0; background:transparent; font-size:22px; line-height:1; cursor:pointer; color:var(--muted); } .product-panel h3 { margin:18px 0 8px; font-size:12px; color:var(--text); } .product-panel .risk { color:var(--red); } .product-panel td { font-size:10px; padding:7px 10px; }
    .product-layout { display:grid; grid-template-columns:minmax(0,1fr) 220px; gap:18px; margin-top:12px; } .chart-box { position:relative; } .chart-box svg { display:block; width:100%; height:auto; } .chart-legend { display:flex; flex-wrap:wrap; gap:4px 12px; margin-bottom:6px; color:var(--muted); font-size:10px; } .key { display:inline-block; width:16px; height:0; border-top:2px solid; margin-right:5px; vertical-align:middle; } .key.dotted { border-top-style:dotted; } .key.dashed { border-top-style:dashed; } .swatch { display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; opacity:.3; vertical-align:middle; } .chart-tooltip { position:absolute; top:40px; min-width:170px; padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--muted); font-size:10px; line-height:1.6; box-shadow:0 4px 12px rgba(20,60,35,.12); pointer-events:none; } .chart-tooltip strong { color:var(--text); font-size:11px; } .tip-date { color:var(--text); font-weight:600; } .product-facts { display:grid; grid-template-columns:auto 1fr; align-content:start; gap:7px 10px; margin:0; font-size:11px; } .product-facts dt { color:var(--muted); font-size:10px; } .product-facts dd { margin:0; color:var(--text); } .chart-table { margin-top:14px; font-size:11px; } .chart-table summary { color:var(--green); cursor:pointer; } .chart-table .table-wrap { max-height:300px; overflow:auto; margin-top:8px; }
    @media (max-width:700px) { .product-layout { grid-template-columns:1fr; } }
    .explain { margin-top:14px; } .explain svg { display:block; width:100%; max-width:620px; height:auto; } .explain .swatch { opacity:1; } .explain-summary { color:var(--text); font-size:11px; margin:0 0 6px; } .drivers-chart { padding:8px 12px 4px; } .drivers-chart svg { display:block; width:100%; height:auto; }
    .datasets { margin-top:34px; } .dataset-toolbar { display:flex; align-items:center; gap:10px; margin:14px 0; } .dataset-toolbar select, .dataset-toolbar a { border:1px solid #b9d5be; border-radius:7px; background:#fff; color:var(--text); padding:8px 10px; font-size:11px; } .dataset-toolbar a { text-decoration:none; } .dataset-meta { color:var(--muted); font-size:11px; } .dataset-table { max-height:390px; overflow:auto; } .dataset-table table { min-width:720px; } .dataset-table th { position:sticky; top:0; } .dataset-page { display:flex; align-items:center; gap:10px; margin-top:10px; color:var(--muted); font-size:11px; } .dataset-page button { border:1px solid #b9d5be; border-radius:6px; background:#fff; color:var(--text); padding:6px 10px; cursor:pointer; }
    .admin-only { display:none !important; } .page-nav { display:flex; align-items:center; gap:8px; } .page-nav a { border:1px solid #b9d5be; border-radius:6px; background:#fff; color:#28633d; padding:9px 12px; text-decoration:none; font-size:11px; } .page-nav a[aria-current="page"] { background:var(--green); border-color:var(--green); color:#fff; } .header-actions { display:flex; align-items:center; gap:10px; }
    body.admin-page .admin-only { display:block !important; } body.admin-page .summary { display:grid !important; } body.admin-page .filters { display:flex !important; } body.admin-page main > .table-wrap.admin-only { display:block !important; } body.admin-page .customer-only { display:none !important; }
    .alert-actions { display:flex; gap:8px; } .alert-actions button { border:1px solid #b9d5be; border-radius:6px; background:#fff; color:#28633d; padding:8px 12px; cursor:pointer; font-size:11px; } .alert-actions button.send { background:var(--green); border-color:var(--green); color:#fff; } .alert-actions button:disabled { opacity:.5; cursor:default; }
    .toast { position:fixed; z-index:10; right:20px; bottom:20px; max-width:420px; padding:12px 16px; border-radius:8px; background:#173b28; color:#fff; font-size:12px; line-height:1.45; box-shadow:0 8px 24px rgba(20,60,35,.25); opacity:0; transform:translateY(10px); transition:opacity .2s, transform .2s; pointer-events:none; } .toast.show { opacity:1; transform:none; } .toast.error { background:var(--red); }
    .admin-products { margin:28px 0; } .admin-product-search { width:260px; border:1px solid #b9d5be; border-radius:6px; background:#fff; color:var(--text); padding:10px 12px; font-size:11px; } .product-admin-table { overflow:auto; max-height:600px; } .product-admin-table table { min-width:1300px; } .product-admin-table td { max-width:300px; overflow-wrap:anywhere; white-space:normal; vertical-align:top; }
    @media (max-width:700px) { .shell{padding:22px 14px} header{margin-bottom:28px; flex-wrap:wrap} .summary{grid-template-columns:1fr 1fr; gap:20px 10px; margin:0 0 25px} .below{grid-template-columns:1fr} .catalog-head{align-items:stretch; flex-direction:column} .search, .admin-product-search{width:100%} .catalog{grid-template-columns:repeat(2,1fr)} .table-wrap{overflow-x:auto} table{min-width:570px} .category-menu{grid-template-columns:repeat(2,minmax(0,1fr))} .region-summary{grid-template-columns:repeat(2,1fr)} .regional-columns{grid-template-columns:1fr} }
    body:not(.admin-page) { background:#f6f7f3; color:#2e342f; }
    body:not(.admin-page) .shell { width:min(1280px,100%); padding:0 30px 60px; }
    .customer-topline { margin:0 -30px; padding:8px 30px; background:#174f35; color:#fff; text-align:center; font-size:11px; letter-spacing:.03em; }
    body:not(.admin-page) header { display:grid; grid-template-columns:auto minmax(260px,1fr) auto; align-items:center; gap:28px; margin:0 -30px; padding:18px 30px; background:#fff; border-bottom:1px solid #e8ece7; }
    .brand-area { display:flex; align-items:center; gap:22px; }
    .brand-mark { color:#168345; text-decoration:none; font-size:27px; font-weight:800; letter-spacing:-1.5px; white-space:nowrap; }
    .brand-mark span { color:var(--orange); }
    .delivery-label { max-width:150px; color:#616b63; font-size:11px; line-height:1.4; }
    .delivery-label strong { display:block; color:#29352c; font-size:12px; }
    .store-search { display:flex; width:100%; max-width:620px; }
    .store-search input { width:100%; border:1px solid #d7ded7; border-radius:6px 0 0 6px; padding:12px 14px; font-size:13px; outline:none; }
    .store-search input:focus { border-color:#168345; box-shadow:0 0 0 3px #16834520; }
    .store-search button { border:0; border-radius:0 6px 6px 0; padding:0 18px; background:var(--orange); color:#fff; font-size:16px; cursor:pointer; }
    body:not(.admin-page) .header-actions { gap:12px; }
    body:not(.admin-page) .page-nav a { border:0; padding:8px; color:#59635b; }
    body:not(.admin-page) .store-actions { gap:0; }
    body:not(.admin-page) .basket-button { min-height:40px; border-radius:6px; background:#168345; padding:10px 16px; font-size:12px; font-weight:700; white-space:nowrap; }
    .hero-banner { position:relative; display:flex; min-height:270px; align-items:center; justify-content:space-between; overflow:hidden; margin:24px 0 16px; padding:36px 54px; border-radius:14px; background:linear-gradient(110deg,#e1f2d7 0%,#f0f8df 55%,#f9edcf 100%); }
    .hero-copy { position:relative; z-index:1; max-width:560px; }
    .hero-kicker { color:#397446; font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .hero-banner h2 { margin:10px 0; color:#183e2a; font-size:clamp(30px,4vw,48px); letter-spacing:-1.5px; line-height:1.05; }
    .hero-banner p { max-width:430px; margin:0 0 20px; color:#5d6a5e; font-size:15px; line-height:1.5; }
    .hero-cta { display:inline-flex; align-items:center; gap:14px; border-radius:6px; padding:12px 16px; background:var(--orange); color:#fff; font-size:12px; font-weight:700; text-decoration:none; }
    .hero-art { display:grid; width:210px; height:210px; flex:0 0 210px; place-items:center; margin-right:7%; overflow:hidden; border:8px solid #ffffff95; border-radius:50%; background:#ffffff95; box-shadow:0 16px 34px #345f2520; }
    .hero-art img { width:100%; height:100%; object-fit:contain; }
    .service-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0 26px; }
    .service-item { display:flex; align-items:center; gap:11px; padding:14px 16px; border:1px solid #e7ebe5; border-radius:8px; background:white; }
    .service-icon { font-size:23px; }
    .service-item strong { display:block; margin-bottom:3px; color:#25362a; font-size:12px; }
    .service-item span:last-child { color:#758078; font-size:10px; }
    body:not(.admin-page) .category-menu { display:flex; gap:18px; overflow:auto; scrollbar-width:none; margin:0 -30px 24px; padding:16px 30px; background:#fff; border-bottom:1px solid #e8ece7; }
    body:not(.admin-page) .category-menu::-webkit-scrollbar { display:none; }
    body:not(.admin-page) .category-panel { display:flex; min-width:94px; min-height:0; flex:0 0 94px; flex-direction:column; gap:8px; align-items:center; border:0; padding:0; background:transparent; box-shadow:none; }
    body:not(.admin-page) .category-panel-header { display:block; margin:0; text-align:center; }
    body:not(.admin-page) .category-panel-title { font-size:10px; line-height:1.3; }
    body:not(.admin-page) .category-mini { display:none; }
    body:not(.admin-page) .category-links { justify-content:center; max-height:none; overflow:visible; }
    body:not(.admin-page) .category-pill { display:grid; width:64px; height:64px; place-items:center; border:2px solid #f2e4cc; border-radius:50%; padding:7px; color:#d67a13; background:#fff8ec; font-size:24px; text-decoration:none; }
    body:not(.admin-page) .category-pill span { display:none; }
    body:not(.admin-page) .category-pill:hover { border-color:var(--orange); background:#fff1d8; }
    .section-eyebrow { margin:30px 0 8px; color:#788279; font-size:10px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    body:not(.admin-page) .catalog-head { margin:0 0 16px; }
    body:not(.admin-page) .catalog-head h2 { font-size:23px; letter-spacing:-.4px; }
    body:not(.admin-page) .catalog-head .search { display:none; }
    body:not(.admin-page) .catalog { grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; }
    body:not(.admin-page) .card { position:relative; border:1px solid #eaede8; border-radius:10px; padding:12px; box-shadow:0 2px 8px #1f35200b; transition:transform .16s ease,box-shadow .16s ease; }
    body:not(.admin-page) .card:hover { transform:translateY(-3px); box-shadow:0 8px 20px #1f352018; }
    body:not(.admin-page) .card-image { height:170px; margin-bottom:8px; background:#fafbf9; }
    body:not(.admin-page) .card h3 { height:34px; margin:8px 0 12px; font-size:12px; }
    body:not(.admin-page) .card .buy { align-items:center; font-size:14px; }
    body:not(.admin-page) .card .basket-button { min-height:34px; padding:8px 11px; font-size:10px; }
    body:not(.admin-page) .basket-panel { width:min(410px,100%); }
    body:not(.admin-page) .storefront-pagination { display:flex; align-items:center; justify-content:center; gap:18px; margin:24px 0 4px; color:#68736b; font-size:12px; }
    body:not(.admin-page) .storefront-pagination button { border:1px solid #d7ded7; border-radius:5px; padding:9px 14px; background:#fff; color:#285d3b; cursor:pointer; }
    body:not(.admin-page) .storefront-pagination button:disabled { opacity:.45; cursor:default; }
    @media (max-width:900px) {
      body:not(.admin-page) header { grid-template-columns:1fr auto; gap:12px; }
      .brand-area { grid-column:1; }
      .store-search { grid-column:1/-1; grid-row:2; max-width:none; }
      .header-actions { grid-column:2; grid-row:1; }
      body:not(.admin-page) .catalog { grid-template-columns:repeat(4,minmax(0,1fr)); }
      .hero-art { margin-right:0; }
    }
    @media (max-width:620px) {
      body:not(.admin-page) .shell { padding:0 14px 40px; }
      .customer-topline { margin:0 -14px; padding:8px 12px; font-size:10px; }
      body:not(.admin-page) header { margin:0 -14px; padding:14px; }
      .brand-mark { font-size:23px; }
      .delivery-label { display:none; }
      body:not(.admin-page) .page-nav { display:none; }
      .hero-banner { min-height:230px; margin:16px 0 12px; padding:24px; }
      .hero-banner h2 { max-width:250px; font-size:32px; }
      .hero-banner p { max-width:250px; font-size:12px; }
      .hero-art { position:absolute; right:-32px; bottom:12px; width:135px; height:135px; font-size:68px; opacity:.75; }
      .service-strip { grid-template-columns:1fr; gap:8px; margin-bottom:18px; }
      .service-item { padding:10px 12px; }
      body:not(.admin-page) .category-menu { margin:0 -14px 20px; padding:12px 14px; }
      body:not(.admin-page) .category-panel { min-width:150px; }
      body:not(.admin-page) .catalog { grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
      body:not(.admin-page) .card { padding:9px; }
      body:not(.admin-page) .card-image { height:140px; }
      body:not(.admin-page) .catalog-head h2 { font-size:20px; }
    }
  </style>
</head>
<body>
<main class="shell">
    <div class="customer-topline customer-only">Fresh picks, fair prices, delivered to your door</div>
    <header>
      <div class="brand-area customer-only"><a class="brand-mark" href="/customer">fresh<span>basket</span></a><div class="delivery-label">Delivering to<strong>Choose your location</strong></div></div>
      <div class="admin-only"><h1>Admin product and inventory details</h1><div class="subhead" id="status">UPDATING</div></div>
      <form class="store-search customer-only" id="store-search"><input id="search" type="search" placeholder="Search for fruits, vegetables, dairy and more" aria-label="Search products"><button type="submit" aria-label="Search">⌕</button></form>
      <div class="header-actions"><nav class="page-nav" aria-label="Pages"><a href="/customer" class="admin-only">Customer</a><a href="/admin" class="customer-only">Admin</a><a href="/admin" class="admin-only" aria-current="page">Admin</a></nav><div class="store-actions customer-only"><button class="basket-button" id="basket-toggle">🛒 Basket <span class="basket-count" id="basket-count">0</span></button></div></div>
    </header>
    <nav class="category-menu customer-only" id="category-menu" aria-label="Shop by category"></nav>
    <section class="hero-banner customer-only"><div class="hero-copy"><div class="hero-kicker">Good food, right on time</div><h2>Freshness for every day</h2><p>Find your everyday favourites, handpicked fresh and delivered straight to your door.</p><a class="hero-cta" href="#catalog">Shop groceries <span>→</span></a></div><div class="hero-art"><img id="hero-product-image" alt="Featured product from our catalog"></div></section>
    <section class="service-strip customer-only" aria-label="Shopping benefits"><div class="service-item"><span class="service-icon">🌿</span><div><strong>Freshly picked</strong><span>Quality essentials for your home</span></div></div><div class="service-item"><span class="service-icon">🚚</span><div><strong>Convenient delivery</strong><span>Choose what works for you</span></div></div><div class="service-item"><span class="service-icon">🛍️</span><div><strong>Everyday favourites</strong><span>All your grocery needs in one place</span></div></div></section>
    <section class="regional-intelligence admin-only"><div class="catalog-head"><h2>Regional inventory intelligence</h2><span class="subhead" id="regional-caption">Restock decisions by region</span></div><div class="region-tabs" id="region-tabs"></div><div class="region-summary" id="region-summary"></div><div class="occasion" id="occasion"></div><div class="subhead" id="not-forecast" style="margin:-4px 0 12px"></div><div class="regional-panel"><h3>Stockout risk and recommended restock</h3><div class="stockout-view" id="regional-view"></div></div><div class="regional-columns" style="margin-top:12px"><div class="regional-panel" style="grid-column:1/-1"><h3>Overstock risk</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Cover</th><th>Action</th><th>Forecast source</th></tr></thead><tbody id="regional-overstock"></tbody></table></div></div></div><div class="regional-panel" style="margin-top:12px"><h3><span id="demand-title">Highest demand — next 7 days</span> <span class="rank-toggle" id="rank-toggle"></span></h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Forecast (7 days)</th><th>Est. revenue (7 days)</th><th>Category</th><th>Forecast source</th></tr></thead><tbody id="regional-demand"></tbody></table></div></div><div class="regional-panel" style="margin-top:12px"><h3><span id="plan-title">Festival planning</span> <span class="rank-toggle" id="plan-rank-toggle"></span></h3><div class="view-toolbar"><select id="plan-festival" aria-label="Festival"></select><span class="view-count" id="plan-uplifts"></span></div><div class="view-caption" id="plan-caption"></div><div class="table-wrap"><table><thead><tr><th>Product</th><th>7-day forecast</th><th>Uplift (category)</th><th>Festival-week units</th><th>Est. revenue</th><th>Category</th></tr></thead><tbody id="plan-body"></tbody></table></div></div><div class="regional-panel stockout-view" style="margin-top:12px"><h3 id="ready-title">Festival readiness</h3><div class="view-tabs" id="ready-tabs"></div><div class="view-toolbar"><select id="ready-status" aria-label="Status"></select><select id="ready-category" aria-label="Category"></select><input id="ready-brand" list="ready-brand-list" placeholder="Brand / company" aria-label="Brand"><datalist id="ready-brand-list"></datalist><button type="button" id="ready-clear">Clear filters</button><a id="ready-download" href="#">Download CSV</a><span class="view-count" id="ready-count"></span></div><div class="view-caption" id="ready-caption"></div><div class="table-wrap"><table><thead id="ready-head"></thead><tbody id="ready-body"></tbody></table></div><div class="view-pager"><button type="button" id="ready-prev">Previous</button><span id="ready-page"></span><button type="button" id="ready-next">Next</button></div></div></section>
  <div class="summary admin-only" id="summary"></div>
    <section class="model-risk admin-only"><div class="catalog-head"><h2>Model risk report <span id="model-risk-scope" class="subhead"></span></h2><div class="alert-actions"><button type="button" id="alert-preview">Preview alert</button><button type="button" class="send" id="alert-send">Email vendor</button></div></div><div class="regional-panel"><h3>Stockout risk: reorder before stock runs out</h3><div class="stockout-view" id="model-view"></div></div><div class="regional-columns" style="margin-top:12px"><div class="regional-panel" style="grid-column:1/-1"><h3>Overstock risk: more than can sell before expiry</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Brand</th><th>Cover</th><th>Shelf life</th></tr></thead><tbody id="model-overstock"></tbody></table></div></div></div></section>
    <section class="model-risk admin-only" id="drivers-card"><div class="catalog-head"><h2>What drives demand <span id="drivers-scope" class="subhead"></span></h2></div><div class="regional-columns"><div class="regional-panel"><h3>Perishable (dairy, sweets)</h3><div class="drivers-chart" id="drivers-perishable"></div></div><div class="regional-panel"><h3>Non-perishable</h3><div class="drivers-chart" id="drivers-nonperishable"></div></div></div><p class="subhead" id="drivers-caption"></p></section>
    <div class="section-eyebrow customer-only">Picked for your kitchen</div><div class="catalog-head customer-only"><h2>Shop groceries <span id="catalog-count" class="subhead"></span></h2><div class="subhead">Fresh picks for every day</div></div>
  <div class="catalog customer-only" id="catalog"></div>
    <div class="storefront-pagination customer-only"><button id="customer-prev" type="button">Previous</button><span id="customer-page-label"></span><button id="customer-next" type="button">Next</button></div>
    <aside class="basket-panel customer-only" id="basket-panel"><div class="basket-head"><h2>Your basket</h2><button id="basket-close" aria-label="Close basket">&times;</button></div><div id="basket-items"></div><div class="basket-total"><span>Total</span><span id="basket-total">₹0.00</span></div><button class="checkout" id="checkout">Continue to payment</button><div class="payment" id="payment"><h3>Payment</h3><label for="customer-name">Name</label><input id="customer-name" placeholder="Your name"><label for="customer-phone">Phone</label><input id="customer-phone" placeholder="10-digit phone number"><label for="payment-method">Payment method</label><input id="payment-method" value="UPI / Card / Cash on delivery"><button class="checkout" id="place-order">Place demo order</button></div></aside>
    <section class="admin-products admin-only"><div class="catalog-head"><h2>Product catalog details <span id="admin-product-count" class="subhead"></span></h2><input id="admin-search" class="admin-product-search" type="search" placeholder="Search all product details"></div><div class="table-wrap product-admin-table"><table><thead><tr><th>Product</th><th>Category</th><th>Subcategory</th><th>Type</th><th>Brand</th><th>Size</th><th>MRP</th><th>EAN / SKU</th><th>Description</th><th>Image URL</th><th>Product URL</th></tr></thead><tbody id="admin-product-body"></tbody></table></div><div class="dataset-page"><button id="product-prev">Previous</button><span id="product-page-label"></span><button id="product-next">Next</button></div></section>
</main>
<aside class="product-panel admin-only" id="product-panel" role="dialog" aria-modal="true" aria-labelledby="product-panel-title" aria-hidden="true"><div class="product-panel-head"><h2 id="product-panel-title"></h2><button type="button" id="product-panel-close" aria-label="Close product panel">&times;</button></div><div id="product-panel-body"></div></aside>
<div class="toast" id="toast" role="status" aria-live="polite"></div>
<script>
const money = value => Number(value).toFixed(1);
let data;
let catalog = [];
let categories = [];
let regionalData = [];
let selectedRegion = 'Mumbai';
let basket = [];
let selectedCategory = '';
let adminProductPage = 0;
const adminProductPageSize = 30;
let adminSearchTimer;
let customerPage = 0;
let customerPageCount = 1;
let customerSearchTimer;
const customerPageSize = 24;
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character])); }
function productCard(product, index) {
  const rawLink = String(product.link || '').trim();
  const normalizedLink = rawLink.toLowerCase();
  const href = normalizedLink.startsWith('https://') || normalizedLink.startsWith('http://') ? escapeHtml(rawLink) : '';
  const image = product.image && product.image !== 'nan'
    ? `<img src="${escapeHtml(product.image)}" alt="${escapeHtml(product.name)}" loading="lazy" referrerpolicy="no-referrer" onerror="this.style.display='none'">`
    : '▣';
  const linkedImage = href ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${image}</a>` : image;
  const name = escapeHtml(product.name);
  const linkedName = href ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${name}</a>` : name;
  const rawPrice = String(product.mrp || '').trim();
  const price = rawPrice && rawPrice.toLowerCase() !== 'nan'
    ? `₹${escapeHtml(rawPrice.replace(/^₹\\s*/, ''))}`
    : 'Price unavailable';
  const detail = escapeHtml([product.brand, product.size].filter(Boolean).join(' · ') || product.subcategory || 'Everyday essential');
  return `<article class="card"><div class="card-image">${linkedImage}</div><p>${detail}</p><h3>${linkedName}</h3><div class="buy"><b>${price}</b><button class="basket-button" data-product-index="${index}">Add to basket</button></div></article>`;
}
function productPrice(product) { const value = Number(String(product.mrp).replace(/[^0-9.]/g, '')); return Number.isFinite(value) ? value : 0; }
function addToBasket(product) { const item = basket.find(entry => entry.name === product.name); if (item) item.quantity += 1; else basket.push({name: product.name, price: productPrice(product), quantity: 1}); renderBasket(); document.getElementById('basket-panel').classList.add('open'); }
function renderBasket() { const count = basket.reduce((sum, item) => sum + item.quantity, 0); const total = basket.reduce((sum, item) => sum + item.price * item.quantity, 0); document.getElementById('basket-count').textContent = count; document.getElementById('basket-total').textContent = `₹${total.toFixed(2)}`; document.getElementById('basket-items').innerHTML = basket.length ? basket.map((item, index) => `<div class="basket-item"><span>${item.name}<br>₹${item.price.toFixed(2)} × ${item.quantity}</span><span><button data-basket-action="minus" data-index="${index}">−</button> <button data-basket-action="plus" data-index="${index}">+</button></span></div>`).join('') : '<p class="subhead">Your basket is empty.</p>'; document.querySelectorAll('[data-basket-action]').forEach(button => button.addEventListener('click', () => { const index = Number(button.dataset.index); basket[index].quantity += button.dataset.basketAction === 'plus' ? 1 : -1; if (basket[index].quantity <= 0) basket.splice(index, 1); renderBasket(); })); }
function normalizeCategory(value) {
  return String(value || '').toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function categoryMatches(product, category) {
  const target = normalizeCategory(category);
  if (!target) return true;

  const values = [
    product.category,
    product.subcategory,
    product.subsubcategory,
    product.name,
    product.brand,
  ].map(value => normalizeCategory(value));

  return values.some(value => {
    if (!value) return false;
    return value === target ||
      value.includes(target) ||
      target.includes(value) ||
      target.split(' ').every(term => value.includes(term));
  });
}

async function renderCatalog() {
  const query = document.getElementById('search').value.trim();
  const offset = customerPage * customerPageSize;
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(customerPageSize),
    search: query,
    category: selectedCategory,
  });
  const response = await fetch(`/api/catalog?${params}`);
  if (!response.ok) throw new Error('Unable to load products. Check the dashboard data files.');
  const payload = await response.json();
  catalog = payload.products;
  const heroImage = catalog.find(product => product.image && product.image !== 'nan')?.image;
  if (heroImage) document.getElementById('hero-product-image').src = heroImage;
  const pageCount = Math.max(1, Math.ceil(payload.total / customerPageSize));
  customerPageCount = pageCount;
  document.getElementById('catalog-count').textContent = `(${payload.total.toLocaleString()} products)`;
  document.getElementById('catalog').innerHTML = catalog.length
    ? catalog.map((product, index) => productCard(product, index)).join('')
    : '<div class="empty">No products available for this search.</div>';
  document.querySelectorAll('[data-product-index]').forEach(button =>
    button.addEventListener('click', () => addToBasket(catalog[Number(button.dataset.productIndex)]))
  );
  document.getElementById('customer-page-label').textContent = payload.total
    ? `Page ${customerPage + 1} of ${pageCount}`
    : 'No products';
  document.getElementById('customer-prev').disabled = customerPage === 0;
  document.getElementById('customer-next').disabled = customerPage >= pageCount - 1;
}
async function renderAdminProducts() {
  const start = adminProductPage * adminProductPageSize;
  const params = new URLSearchParams({
    offset: String(start),
    limit: String(adminProductPageSize),
    search: document.getElementById('admin-search').value.trim(),
  });
  const response = await fetch(`/api/catalog?${params}`);
  if (!response.ok) throw new Error('Unable to load products. Check the dashboard data files.');
  const payload = await response.json();
  const visible = payload.products;
  const total = payload.total;
  const pageCount = Math.max(1, Math.ceil(total / adminProductPageSize));
  document.getElementById('admin-product-count').textContent = `(${total.toLocaleString()} products)`;
  document.getElementById('admin-product-body').innerHTML = visible.length ? visible.map(product => `<tr>${[
    product.name, product.category, product.subcategory, product.subsubcategory, product.brand,
    product.size, product.mrp, product.ean, product.description, product.image, product.link,
  ].map(value => `<td>${escapeHtml(value)}</td>`).join('')}</tr>`).join('') : '<tr><td class="empty" colspan="11">No products match this search.</td></tr>';
  document.getElementById('product-page-label').textContent = total
    ? `Products ${start + 1}-${Math.min(start + adminProductPageSize, total)} of ${total.toLocaleString()}`
    : 'No products';
  document.getElementById('product-prev').disabled = adminProductPage === 0;
  document.getElementById('product-next').disabled = adminProductPage >= pageCount - 1;
}
function productRelevance(product, categoryFilter = '') {
  const target = normalizeCategory(categoryFilter);
  const text = normalizeCategory(`${product.name} ${product.category} ${product.subcategory} ${product.subsubcategory} ${product.brand}`);
  let score = 0;

  if (target) {
    if (text.includes(target)) score += 100;
    if (normalizeCategory(product.category).includes(target)) score += 50;
    if (normalizeCategory(product.subcategory).includes(target)) score += 35;
    if (normalizeCategory(product.subsubcategory).includes(target)) score += 25;
    if (normalizeCategory(product.name).includes(target)) score += 15;
  } else {
    score += 20;
  }

  const price = Number(String(product.mrp || '').replace(/[^0-9.]/g, ''));
  if (Number.isFinite(price) && price > 0) score += Math.min(price / 70, 20);
  if (product.image && product.image !== 'nan') score += 5;
  if (product.category) score += 2;
  return score;
}

function renderCategories() {
  const categoryOrder = [
    'Fruits & Vegetables', 'Foodgrains Oil & Masala', 'Bakery Cakes & Dairy',
    'Beverages', 'Snacks & Branded Foods', 'Eggs Meat & Fish',
    'Gourmet & World Food', 'Baby Care', 'Beauty & Hygiene',
    'Cleaning & Household', 'Pharmacy & Wellness', 'Kitchen Garden & Pets',
    'Food Court', 'Fashion', 'Electronics',
  ];
  const parentGroups = categories.length
    ? categories.filter(item => Number(item.level) === 1).sort((left, right) =>
      categoryOrder.indexOf(left.category) - categoryOrder.indexOf(right.category))
    : [];
  const fallback = [...new Set((catalog || []).map(item => String(item.category || '').trim()).filter(Boolean))];
  const visibleParents = parentGroups.length ? parentGroups : fallback.map(name => ({ category: name }));
  const categoryIcons = {
    'Fruits & Vegetables': '🥬',
    'Foodgrains Oil & Masala': '🌾',
    'Bakery Cakes & Dairy': '🥛',
    'Beverages': '🥤',
    'Snacks & Branded Foods': '🍪',
    'Eggs Meat & Fish': '🥚',
    'Gourmet & World Food': '🧀',
    'Baby Care': '🍼',
    'Beauty & Hygiene': '🧴',
    'Cleaning & Household': '🧹',
    'Pharmacy & Wellness': '💊',
    'Kitchen Garden & Pets': '🐾',
    'Food Court': '🍽️',
    'Fashion': '👕',
    'Electronics': '🔌',
  };

  document.getElementById('category-menu').innerHTML = visibleParents.map(parent => {
    const icon = categoryIcons[parent.category] || '🛍️';
    const pills = `<a class="category-pill" href="#" data-category="${escapeHtml(parent.category)}" aria-label="${escapeHtml(parent.category)}">${icon}</a>`;
    return `
      <div class="category-panel">
        <div class="category-panel-header">
          <div class="category-panel-title">${escapeHtml(parent.category)}</div>
        </div>
        <div class="category-links">${pills}</div>
      </div>
    `;
  }).join('');

  document.querySelectorAll('#category-menu .category-pill').forEach(link => link.addEventListener('click', event => {
    event.preventDefault();
    selectedCategory = event.currentTarget.dataset.category;
    document.getElementById('search').value = '';
    customerPage = 0;
    renderCatalog();
    document.getElementById('catalog').scrollIntoView({ behavior: 'smooth' });
  }));
}
function renderRegional() {
  const region = regionalData.find(item => item.region === selectedRegion) || regionalData[0];
  if (!region) return;
  selectedRegion = region.region;
  document.getElementById('region-tabs').innerHTML = regionalData.map(item => `<button class="region-tab ${item.region === selectedRegion ? 'active' : ''}" data-region="${item.region}">${item.region}</button>`).join('');
  document.querySelectorAll('.region-tab').forEach(button => button.addEventListener('click', () => { selectedRegion = button.dataset.region; renderRegional(); refreshStockoutViews(); loadDashboard(selectedRegion).catch(error => console.error(error)); }));
  document.getElementById('rank-toggle').innerHTML = `Rank by: ${[['units', 'Units'], ['revenue', 'Revenue']].map(([key, label]) => `<button type="button" class="filter ${rankBy === key ? 'active' : ''}" data-rank="${key}">${label}</button>`).join('')}`;
  document.querySelectorAll('[data-rank]').forEach(button => button.addEventListener('click', () => { rankBy = button.dataset.rank; saveUrlState(); renderRegional(); }));
  const festival = region.next_festival;
  document.getElementById('region-summary').innerHTML = `<div class="region-metric"><span>Region SKUs</span><strong>${region.sku_count.toLocaleString()}</strong></div><div class="region-metric alert"><span>Stockout risk</span><strong>${region.stockout_count.toLocaleString()}</strong></div><div class="region-metric warn"><span>Overstock risk</span><strong>${region.overstock_count.toLocaleString()}</strong></div><div class="region-metric"><span>Next festival</span><strong>${festival ? `${escapeHtml(festival.name)} · ${festival.days_until}d` : 'None scheduled'}</strong></div><div class="region-metric" id="readiness-kpi"><span>Festival readiness</span><strong>–</strong></div>`;
  document.getElementById('regional-caption').textContent = `Forecast: LightGBM (direct for ${data?.tracked_products ?? '–'} tracked, analog for the rest) · Stock: simulated`;
  const windowLabel = `${shortDate(region.window_start)}–${shortDate(region.window_end)}`;
  const inWindow = festival && festival.date <= region.window_end;
  document.getElementById('occasion').textContent = `Data as of ${longDate(region.data_date)}. The forecast covers the next 7 days, ${windowLabel}${inWindow ? `, which include ${festival.name} (${shortDate(festival.date)})` : ''}${festival && !inWindow ? `; the next festival is ${festival.name} on ${shortDate(festival.date)}` : ''}. Festivals further ahead are under Festival planning.`;
  document.getElementById('not-forecast').innerHTML = region.not_forecast_count == null ? '' : `${region.not_forecast_count.toLocaleString()} products not forecast (free samples / no price / packaging) · <a href="${region.not_forecast_download}" style="color:var(--green)">download list</a>`;
  document.getElementById('regional-overstock').innerHTML = region.overstock.map(row => `<tr><td>${productLink(row.product_name, row.sku_id, row.region, 'regional')}</td><td class="risk">${money(row.days_of_cover)} days</td><td class="safe">Reduce orders</td><td>${escapeHtml(row.forecast_source)}</td></tr>`).join('');
  document.getElementById('demand-title').textContent = `Highest demand — next 7 days (${windowLabel})`;
  document.getElementById('regional-demand').innerHTML = (rankBy === 'revenue' ? region.demand_by_revenue : region.demand).map(row => `<tr><td>${productLink(row.product_name, row.sku_id, row.region, 'regional')}</td><td>${fmt(row.forecast_7d)} units</td><td>${row.revenue_estimate == null ? '–' : `₹${fmt(row.revenue_estimate)} (est.)`}</td><td>${escapeHtml(row.category)}</td><td>${escapeHtml(row.forecast_source)}</td></tr>`).join('');
  renderFestivalPlanning().catch(error => console.error(error));
  renderReadiness().catch(error => console.error(error));
}
let planFestival = '';
let planRank = 'units';
let planRequest = 0;
async function renderFestivalPlanning() {
  const request = ++planRequest;
  const response = await fetch(`/api/festival-planning?${new URLSearchParams({city: selectedRegion, festival: planFestival, rank: planRank})}`);
  const plan = await response.json();
  if (request !== planRequest) return;
  const body = document.getElementById('plan-body');
  if (!response.ok) { body.innerHTML = `<tr><td class="empty" colspan="6">${escapeHtml(plan.error || 'Unable to load festival planning')}</td></tr>`; return; }
  document.getElementById('plan-title').textContent = `Festival planning: ${plan.festival} (${shortDate(plan.date)}, in ${plan.days_until} days)`;
  document.getElementById('plan-festival').innerHTML = plan.options.map(option => `<option value="${escapeHtml(option.festival)}" ${option.has_uplift ? '' : 'disabled'} ${option.festival === plan.festival ? 'selected' : ''}>${escapeHtml(option.festival)} · ${shortDate(option.date)}${option.has_uplift ? '' : ' (no past years to measure)'}</option>`).join('');
  document.getElementById('plan-rank-toggle').innerHTML = `Rank by: ${[['units', 'Units'], ['revenue', 'Revenue']].map(([key, label]) => `<button type="button" class="filter ${planRank === key ? 'active' : ''}" data-plan-rank="${key}">${label}</button>`).join('')}`;
  document.querySelectorAll('[data-plan-rank]').forEach(button => button.addEventListener('click', () => { planRank = button.dataset.planRank; saveUrlState(); renderFestivalPlanning(); }));
  const years = plan.uplifts.length ? `${Math.min(...plan.uplifts.map(u => u.first_year))}–${Math.max(...plan.uplifts.map(u => u.last_year))}` : '';
  document.getElementById('plan-uplifts').textContent = plan.uplifts.map(u => `${u.category} ×${u.uplift.toFixed(1)}`).join(' · ') + (plan.uplifts.length ? ` at ${plan.festival} in ${plan.city}` : '');
  document.getElementById('plan-caption').textContent = `Planning estimate, not a model forecast (the model forecasts 7 days ahead). Festival-week units = the product's current 7-day forecast (${shortDate(plan.window_start)}–${shortDate(plan.window_end)}) × its category's uplift: festival-week demand vs the 4 normal weeks before it, in ${plan.city}, averaged over past ${plan.festival}s (${years}, simulated sales history). Revenue = units × MRP (estimated).`;
  body.innerHTML = plan.rows.map(row => `<tr><td>${productLink(row.product_name, row.sku_id, row.city, 'regional', plan.festival)}</td><td>${fmt(row.forecast_7d)}</td><td>${escapeHtml(row.model_category)} ×${row.uplift.toFixed(2)}</td><td>${fmt(row.festival_week_units)} units</td><td>${row.revenue_estimate == null ? '–' : `₹${fmt(row.revenue_estimate)} (est.)`}</td><td>${escapeHtml(row.category)}</td></tr>`).join('') || `<tr><td class="empty" colspan="6">No measured uplift for ${escapeHtml(plan.festival)} in ${escapeHtml(plan.city)}</td></tr>`;
}
const STOCKOUT_KEYS = ['view', 'brand', 'category', 'min_price', 'max_price', 'top', 'trend', 'max_cover', 'source', 'q', 'sort', 'order', 'page'];
const PRICE_BANDS = {lt50: ['', '49.99'], '50-200': ['50', '200'], '200-500': ['200.01', '500'], gt500: ['500.01', '']};
const STOCKOUT_PAGE_SIZE = 50;
const stockoutViews = [];
let rankBy = 'units';
let openProduct = null;
function saveUrlState() {
  const params = new URLSearchParams();
  if (selectedRegion) params.set('city', selectedRegion);
  if (rankBy !== 'units') params.set('rank', rankBy);
  if (openProduct) { params.set('product', openProduct.sku); params.set('product_city', openProduct.city); params.set('product_scope', openProduct.scope); if (openProduct.festival) params.set('product_festival', openProduct.festival); }
  if (planFestival) params.set('plan', planFestival);
  if (planRank !== 'units') params.set('plan_rank', planRank);
  READY_FILTERS.forEach(key => { if (readinessState[key] && !(key === 'page' && readinessState[key] === '0')) params.set(`ready_${key}`, readinessState[key]); });
  stockoutViews.forEach(view => Object.entries(view.state).forEach(([key, value]) => {
    if (value && !(key === 'page' && value === '0') && !(key === 'view' && value === 'table')) params.set(`${view.prefix}_${key}`, value);
  }));
  const query = params.toString();
  history.replaceState(null, '', `${location.pathname}${query ? `?${query}` : ''}`);
}
function readUrlState() {
  const params = new URLSearchParams(location.search);
  if (params.get('city')) selectedRegion = params.get('city');
  if (params.get('rank') === 'revenue') rankBy = 'revenue';
  if (params.get('product')) openProduct = {sku: params.get('product'), city: params.get('product_city') || selectedRegion, scope: params.get('product_scope') || 'regional', festival: params.get('product_festival') || ''};
  planFestival = params.get('plan') || '';
  if (params.get('plan_rank') === 'revenue') planRank = 'revenue';
  READY_FILTERS.forEach(key => { readinessState[key] = params.get(`ready_${key}`) || (key === 'page' ? '0' : ''); });
  return params;
}
const fmt = (value, digits = 0) => value == null ? '–' : Number(value).toLocaleString('en-IN', {minimumFractionDigits: digits, maximumFractionDigits: digits});
function trendCell(value) {
  if (value == null) return '–';
  return `<span class="${value >= 0 ? 'safe' : 'risk'}">${value >= 0 ? '▲' : '▼'} ${Math.abs(value).toFixed(1)}%</span>`;
}
function stockoutRowHtml(row, scope) {
  const daily = row.has_history ? fmt(row.avg_daily_sales, 1) : `${fmt(row.forecast_daily, 1)} <span class="subhead">forecast (no history)</span>`;
  const stockout = row.risk_status === 'stockout';
  return `<tr><td>${productLink(row.product_name, row.sku_id, row.city, scope)}</td><td>${escapeHtml(row.brand)}</td><td>${escapeHtml(row.category)}</td><td>${row.price == null ? '–' : `₹${fmt(row.price, 2)}`}</td><td>${fmt(row.units_7d)}</td><td>${fmt(row.units_28d)}</td><td>${daily}</td><td>${trendCell(row.trend_pct)}</td><td class="${stockout ? 'risk' : ''}">${fmt(row.days_of_cover, 1)} days${stockout ? ' · stockout' : ''}</td><td>${fmt(row.days_until_delivery)} day${row.days_until_delivery === 1 ? '' : 's'}</td><td>${fmt(row.reorder_qty)} units</td><td>${row.forecast_source === 'lightgbm_direct' ? 'direct' : 'analog'}</td></tr>`;
}
const STOCKOUT_COLUMNS = [['Product'], ['Brand'], ['Category'], ['MRP', 'price'], ['Sold 7d'], ['Sold 28d', 'units_28d'], ['Avg daily sales (28d)', 'avg_daily_sales'], ['Trend', 'trend'], ['Days until stockout', 'cover'], ['Next delivery'], ['Reorder qty', 'reorder_qty'], ['Source']];
const BRAND_COLUMNS = ['Brand', 'Units sold 28d', 'Revenue 28d (est.)', 'Products', 'At stockout risk', 'Reorder units', 'Avg cover'];
function createStockoutView(prefix, scope, containerId, caption) {
  const params = new URLSearchParams(location.search);
  const view = {prefix, scope, state: {}, request: 0};
  STOCKOUT_KEYS.forEach(key => { view.state[key] = params.get(`${prefix}_${key}`) || ''; });
  view.state.view = view.state.view || 'table';
  view.state.page = view.state.page || '0';
  const el = document.getElementById(containerId);
  const option = (value, label) => `<option value="${value}">${label}</option>`;
  el.innerHTML = `<div class="view-tabs"><button type="button" class="filter" data-tab="table">Stocks out first</button><button type="button" class="filter" data-tab="brands">By brand</button></div>
    <div class="view-toolbar">
      <input data-key="brand" list="${prefix}-brand-list" placeholder="Brand / company" aria-label="Brand"><datalist id="${prefix}-brand-list"></datalist>
      <select data-key="category" aria-label="Category"><option value="">All categories</option></select>
      <select data-band aria-label="Price band">${option('', 'Any price')}${option('lt50', '&lt;₹50')}${option('50-200', '₹50–200')}${option('200-500', '₹200–500')}${option('gt500', '&gt;₹500')}${option('custom', 'Custom')}</select>
      <input data-key="min_price" type="number" min="0" placeholder="Min ₹" aria-label="Minimum price"><input data-key="max_price" type="number" min="0" placeholder="Max ₹" aria-label="Maximum price">
      <select data-key="top" aria-label="Sales volume">${option('', 'All sellers')}${option('10', 'Top 10% sellers')}${option('25', 'Top 25% sellers')}</select>
      <select data-key="trend" aria-label="Trend">${option('', 'Any trend')}${option('rising', 'Rising')}${option('falling', 'Falling')}</select>
      <select data-key="max_cover" aria-label="Days until stockout">${option('', 'Any cover')}${option('2', '≤2 days')}${option('4', '≤4 days')}${option('7', '≤7 days')}</select>
      <select data-key="source" aria-label="Forecast source">${option('', 'Any source')}${option('direct', 'Direct')}${option('analog', 'Analog')}</select>
      <input data-key="q" type="search" placeholder="Search product name" aria-label="Search product name">
      <button type="button" data-clear>Clear filters</button><a data-download href="#">Download CSV</a><span class="view-count" data-count></span>
    </div><div class="view-caption">${caption.replace('{day}', '…')}</div><div class="table-wrap"><table><thead data-head></thead><tbody data-body></tbody></table></div>
    <div class="view-pager"><button type="button" data-prev>Previous</button><span data-page-label></span><button type="button" data-next>Next</button></div>`;
  const query = (extra = {}) => {
    const out = new URLSearchParams({scope, city: selectedRegion || 'all'});
    STOCKOUT_KEYS.filter(key => !['view', 'page'].includes(key)).forEach(key => { if (view.state[key]) out.set(key, view.state[key]); });
    Object.entries(extra).forEach(([key, value]) => out.set(key, value));
    return out;
  };
  const set = (changes) => { Object.assign(view.state, changes); if (!('page' in changes)) view.state.page = '0'; view.refresh(); };
  let typingTimer;
  el.querySelectorAll('[data-key]').forEach(input => {
    const key = input.dataset.key;
    const typed = ['q', 'min_price', 'max_price'].includes(key);
    input.addEventListener(typed ? 'input' : 'change', () => {
      clearTimeout(typingTimer);
      typingTimer = setTimeout(() => set({[key]: input.value.trim()}), typed ? 300 : 0);
    });
  });
  el.querySelector('[data-band]').addEventListener('change', event => {
    const band = PRICE_BANDS[event.target.value];
    if (band) set({min_price: band[0], max_price: band[1]});
    else if (!event.target.value) set({min_price: '', max_price: ''});
  });
  el.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => set({view: button.dataset.tab})));
  el.querySelector('[data-clear]').addEventListener('click', () => {
    const cleared = {};
    STOCKOUT_KEYS.filter(key => key !== 'view').forEach(key => { cleared[key] = ''; });
    set(cleared);
  });
  el.querySelector('[data-prev]').addEventListener('click', () => set({page: String(Math.max(0, Number(view.state.page) - 1))}));
  el.querySelector('[data-next]').addEventListener('click', () => set({page: String(Number(view.state.page) + 1)}));
  el.querySelector('[data-head]').addEventListener('click', event => {
    const sortKey = event.target.closest('[data-sort]')?.dataset.sort;
    if (!sortKey) return;
    const defaultOrder = ['cover', 'price'].includes(sortKey) ? 'asc' : 'desc';
    const current = view.state.sort === sortKey ? (view.state.order || defaultOrder) : '';
    set({sort: sortKey, order: current ? (current === 'asc' ? 'desc' : 'asc') : defaultOrder});
  });
  el.querySelector('[data-body]').addEventListener('click', event => {
    const brand = event.target.closest('[data-brand]')?.dataset.brand;
    if (brand != null) set({brand, view: 'table'});
  });
  view.syncControls = () => {
    el.querySelectorAll('[data-key]').forEach(input => { if (document.activeElement !== input) input.value = view.state[input.dataset.key]; });
    const band = Object.entries(PRICE_BANDS).find(([, [low, high]]) => low === view.state.min_price && high === view.state.max_price);
    el.querySelector('[data-band]').value = band ? band[0] : (view.state.min_price || view.state.max_price ? 'custom' : '');
    el.querySelectorAll('[data-tab]').forEach(button => button.classList.toggle('active', button.dataset.tab === view.state.view));
    el.querySelector('[data-download]').href = `/api/stockout?${query({format: 'csv', sort: view.state.sort, order: view.state.order})}`;
  };
  view.refresh = async () => {
    saveUrlState();
    view.syncControls();
    const request = ++view.request;
    const offset = Number(view.state.page) * STOCKOUT_PAGE_SIZE;
    const brandsQuery = query();
    const [listResponse, brandsResponse] = await Promise.all([
      fetch(`/api/stockout?${query({offset: String(offset), limit: String(STOCKOUT_PAGE_SIZE)})}`),
      fetch(`/api/stockout/brands?${brandsQuery}`),
    ]);
    if (request !== view.request) return;
    const list = await listResponse.json();
    const brands = await brandsResponse.json();
    const body = el.querySelector('[data-body]');
    if (!listResponse.ok || !brandsResponse.ok) {
      body.innerHTML = `<tr><td class="empty" colspan="12">${escapeHtml(list.error || brands.error || 'Unable to load')}</td></tr>`;
      return;
    }
    el.querySelector(`#${prefix}-brand-list`).innerHTML = brands.rows.map(row => `<option value="${escapeHtml(row.brand)}">${row.products} products</option>`).join('');
    const categorySelect = el.querySelector('[data-key="category"]');
    categorySelect.innerHTML = `<option value="">All categories</option>${brands.categories.map(row => `<option value="${escapeHtml(row.category)}">${escapeHtml(row.category)} (${row.products})</option>`).join('')}`;
    categorySelect.value = view.state.category;
    el.querySelector('[data-count]').textContent = `${list.total.toLocaleString()} of ${list.products_in_city.toLocaleString()} products`;
    const pageCount = Math.max(1, Math.ceil(list.total / STOCKOUT_PAGE_SIZE));
    const head = el.querySelector('[data-head]');
    const pager = el.querySelector('.view-pager');
    const captionText = caption.replace('{day}', escapeHtml(list.planning_day || ''));
    if (view.state.view === 'brands') {
      head.innerHTML = `<tr>${BRAND_COLUMNS.map(label => `<th>${label}</th>`).join('')}</tr>`;
      body.innerHTML = brands.rows.slice(0, 100).map(row => `<tr class="brand-row" data-brand="${escapeHtml(row.brand)}"><td>${escapeHtml(row.brand || '(no brand)')}${row.estimated_products ? ' *' : ''}</td><td>${fmt(row.units_28d)}</td><td>₹${fmt(row.revenue_28d)}</td><td>${fmt(row.products)}</td><td class="${row.stockout_risk ? 'risk' : ''}">${fmt(row.stockout_risk)}</td><td>${fmt(row.reorder_units)}</td><td>${fmt(row.avg_cover, 1)} days</td></tr>`).join('') || `<tr><td class="empty" colspan="7">No brands match these filters</td></tr>`;
      pager.style.display = 'none';
      el.querySelector('.view-caption').innerHTML = `${captionText} By brand: top ${Math.min(100, brands.total)} of ${brands.total.toLocaleString()} brands by units sold; click a brand to filter. * includes products without history, counted as 28 × forecast daily (estimated). Revenue = units × MRP (estimated).`;
      return;
    }
    el.querySelector('.view-caption').innerHTML = captionText;
    head.innerHTML = `<tr>${STOCKOUT_COLUMNS.map(([label, key]) => {
      if (!key) return `<th>${label}</th>`;
      const arrow = view.state.sort === key ? ((view.state.order || (['cover', 'price'].includes(key) ? 'asc' : 'desc')) === 'asc' ? ' ↑' : ' ↓') : '';
      return `<th class="sortable" data-sort="${key}">${label}${arrow}</th>`;
    }).join('')}</tr>`;
    body.innerHTML = list.rows.map(row => stockoutRowHtml(row, scope)).join('') || `<tr><td class="empty" colspan="12">No products match these filters</td></tr>`;
    pager.style.display = '';
    el.querySelector('[data-page-label]').textContent = list.total ? `Products ${offset + 1}-${Math.min(offset + STOCKOUT_PAGE_SIZE, list.total)} of ${list.total.toLocaleString()} · page ${Number(view.state.page) + 1} of ${pageCount}` : 'No products';
    el.querySelector('[data-prev]').disabled = Number(view.state.page) === 0;
    el.querySelector('[data-next]').disabled = Number(view.state.page) >= pageCount - 1;
  };
  stockoutViews.push(view);
  return view;
}
function refreshStockoutViews() { stockoutViews.forEach(view => view.refresh().catch(error => console.error(error))); }
const CHART_COLORS = {sold: '#2a78d6', demand: '#eb6834', stockout: '#d94b4b', grid: '#dce8dc', muted: '#66806f', text: '#173b28', wash: '#f4f8f4'};
let productPanelOpener = null;
const isoDay = iso => new Date(`${iso}T00:00:00`);
const shortDate = iso => isoDay(iso).toLocaleDateString('en-IN', {day: 'numeric', month: 'short'});
const longDate = iso => isoDay(iso).toLocaleDateString('en-IN', {day: 'numeric', month: 'short', year: 'numeric'});
const addDays = (day, count) => { const next = new Date(day.getFullYear(), day.getMonth(), day.getDate() + count); return `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, '0')}-${String(next.getDate()).padStart(2, '0')}`; };
function niceStep(max, count) {
  const raw = max / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map(multiple => multiple * power).find(step => step >= raw);
}
function productLink(name, sku, city, scope, festival = '') {
  return `<a href="#" class="product-link" data-sku="${escapeHtml(sku)}" data-city="${escapeHtml(city)}" data-scope="${scope}" data-festival="${escapeHtml(festival)}">${escapeHtml(name || sku)}</a>`;
}
function productChart(product) {
  const W = 540, H = 280, m = {left: 46, right: 16, top: 40, bottom: 26};
  const history = product.history, forecast = product.forecast;
  const base = isoDay(history.length ? history[0].date : product.today);
  const index = iso => Math.round((isoDay(iso) - base) / 86400000);
  const todayIndex = index(product.today);
  const lastIndex = index(forecast[forecast.length - 1].date);
  const plotW = W - m.left - m.right, plotH = H - m.top - m.bottom;
  const x = i => m.left + (i / Math.max(1, lastIndex)) * plotW;
  const dayW = plotW / Math.max(1, lastIndex);
  const seriesMax = Math.max(1, ...history.map(h => Math.max(h.units_sold, h.true_demand ?? 0)), ...forecast.map(f => f.units ?? 0));
  const stock = product.stock.units;
  const stockOnScale = stock <= seriesMax * 3;
  const top = Math.max(seriesMax, stockOnScale ? stock : 0) * 1.08;
  const step = niceStep(top, 4);
  const yMax = Math.ceil(top / step) * step;
  const y = value => m.top + plotH - (value / yMax) * plotH;
  const compact = value => value >= 1000 ? `${(value / 1000).toFixed(value % 1000 ? 1 : 0)}K` : String(value);
  const path = points => points.map(([i, v], k) => `${k ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const parts = [];
  parts.push(`<rect x="${x(todayIndex)}" y="${m.top}" width="${x(lastIndex) - x(todayIndex) + dayW / 2}" height="${plotH}" fill="${CHART_COLORS.wash}"/>`);
  parts.push(`<text x="${x(lastIndex)}" y="${m.top + 12}" text-anchor="end" font-size="9" fill="${CHART_COLORS.muted}">Forecast</text>`);
  for (let value = 0; value <= yMax + 1e-9; value += step) {
    parts.push(`<line x1="${m.left}" x2="${W - m.right}" y1="${y(value)}" y2="${y(value)}" stroke="${CHART_COLORS.grid}" stroke-width="1"/>`);
    parts.push(`<text x="${m.left - 6}" y="${y(value) + 3}" text-anchor="end" font-size="9" fill="${CHART_COLORS.muted}">${compact(value)}</text>`);
  }
  history.forEach(h => { if (h.stockout) parts.push(`<rect x="${x(index(h.date)) - dayW / 2}" y="${m.top}" width="${dayW}" height="${plotH}" fill="${CHART_COLORS.stockout}" fill-opacity="0.14"><title>Stockout day ${escapeHtml(shortDate(h.date))} (simulated)</title></rect>`); });
  product.festivals.forEach((festival, k) => {
    const fx = x(index(festival.date));
    const labelY = m.top - 8 - (k % 2) * 11;
    const anchor = fx > W - 90 ? 'end' : fx < m.left + 60 ? 'start' : 'middle';
    parts.push(`<line x1="${fx}" x2="${fx}" y1="${labelY + 3}" y2="${m.top + plotH}" stroke="${CHART_COLORS.muted}" stroke-width="1" stroke-opacity="0.55"/>`);
    parts.push(`<text x="${fx}" y="${labelY}" text-anchor="${anchor}" font-size="9" fill="${CHART_COLORS.text}">${escapeHtml(festival.name)}</text>`);
  });
  if (product.festival_after) {
    const after = product.festival_after;
    parts.push(`<text x="${W - m.right}" y="${m.top - 8}" text-anchor="end" font-size="9" fill="${CHART_COLORS.text}">${escapeHtml(after.name)} ${escapeHtml(shortDate(after.date))} (in ${index(after.date) - todayIndex} days) →</text>`);
  }
  const stockY = stockOnScale ? y(stock) : m.top;
  parts.push(`<line x1="${m.left}" x2="${W - m.right}" y1="${stockY}" y2="${stockY}" stroke="${CHART_COLORS.text}" stroke-width="1" stroke-opacity="0.7"/>`);
  parts.push(`<text x="${m.left + 4}" y="${stockY - 4}" font-size="9" fill="${CHART_COLORS.text}">${stockOnScale ? '' : '↑ '}Stock ${stock.toLocaleString('en-IN')} units (simulated)${stockOnScale ? '' : ', off scale'}</text>`);
  if (history.length) {
    parts.push(`<path d="${path(history.filter(h => h.true_demand != null).map(h => [index(h.date), h.true_demand]))}" fill="none" stroke="${CHART_COLORS.demand}" stroke-width="2" stroke-dasharray="0.1 4" stroke-linecap="round" stroke-linejoin="round"/>`);
    parts.push(`<path d="${path(history.map(h => [index(h.date), h.units_sold]))}" fill="none" stroke="${CHART_COLORS.sold}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`);
  }
  const last = history[history.length - 1];
  const forecastPoints = (last ? [[todayIndex, last.true_demand ?? last.units_sold]] : []).concat(forecast.map(f => [index(f.date), f.units]));
  parts.push(`<path d="${path(forecastPoints)}" fill="none" stroke="${CHART_COLORS.demand}" stroke-width="2" stroke-dasharray="6 4" stroke-linecap="round" stroke-linejoin="round"/>`);
  const end = forecast[forecast.length - 1];
  parts.push(`<circle cx="${x(index(end.date))}" cy="${y(end.units)}" r="4" fill="${CHART_COLORS.demand}" stroke="#fff" stroke-width="2"/>`);
  const ticks = [];
  for (let i = 0; i <= lastIndex; i += history.length ? 14 : 1) ticks.push(i);
  ticks.forEach(i => {
    const label = shortDate(addDays(base, i));
    parts.push(`<text x="${x(i)}" y="${H - 8}" text-anchor="middle" font-size="9" fill="${CHART_COLORS.muted}">${label}</text>`);
  });
  parts.push(`<line x1="${x(todayIndex)}" x2="${x(todayIndex)}" y1="${m.top}" y2="${m.top + plotH}" stroke="${CHART_COLORS.text}" stroke-width="1" stroke-opacity="0.35"/>`);
  const nearLeft = todayIndex < 3;
  parts.push(`<text x="${x(todayIndex) + (nearLeft ? 4 : -4)}" y="${m.top + plotH - 5}" text-anchor="${nearLeft ? 'start' : 'end'}" font-size="9" fill="${CHART_COLORS.muted}">Today</text>`);
  parts.push(`<line class="crosshair" x1="0" x2="0" y1="${m.top}" y2="${m.top + plotH}" stroke="${CHART_COLORS.text}" stroke-width="1" visibility="hidden"/>`);
  parts.push(`<rect class="hit" x="${m.left - dayW / 2}" y="${m.top}" width="${plotW + dayW}" height="${plotH}" fill="transparent"/>`);
  return {svg: `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily units for ${escapeHtml(product.details.product_name)}: last ${history.length} days and ${forecast.length}-day forecast">${parts.join('')}</svg>`, x, index, dayW, W, m, lastIndex, base};
}
function chartRows(product) {
  const rows = new Map();
  product.history.forEach(h => rows.set(h.date, {date: h.date, sold: h.units_sold, demand: h.true_demand, stockout: h.stockout}));
  product.forecast.forEach(f => rows.set(f.date, {...(rows.get(f.date) || {date: f.date}), forecast: f.units}));
  product.festivals.forEach(f => { if (rows.has(f.date)) rows.get(f.date).festival = f.name; });
  return [...rows.values()].sort((a, b) => a.date.localeCompare(b.date));
}
function attachChartHover(container, chart, product) {
  const byIndex = new Map(chartRows(product).map(row => [chart.index(row.date), row]));
  const svg = container.querySelector('svg');
  const crosshair = svg.querySelector('.crosshair');
  const tooltip = container.querySelector('.chart-tooltip');
  const hide = () => { crosshair.setAttribute('visibility', 'hidden'); tooltip.hidden = true; };
  svg.querySelector('.hit').addEventListener('pointermove', event => {
    const box = svg.getBoundingClientRect();
    const px = (event.clientX - box.left) * chart.W / box.width;
    const i = Math.max(0, Math.min(chart.lastIndex, Math.round((px - chart.m.left) / chart.dayW)));
    const row = byIndex.get(i);
    if (!row) return hide();
    crosshair.setAttribute('x1', chart.x(i)); crosshair.setAttribute('x2', chart.x(i)); crosshair.setAttribute('visibility', 'visible');
    tooltip.replaceChildren();
    const line = (value, label, color, dash) => {
      const item = document.createElement('div');
      if (color) { const key = document.createElement('span'); key.className = `key ${dash || ''}`; key.style.borderColor = color; item.append(key); }
      const strong = document.createElement('strong'); strong.textContent = value; item.append(strong, ` ${label}`);
      tooltip.append(item);
    };
    const title = document.createElement('div'); title.className = 'tip-date'; title.textContent = shortDate(row.date); tooltip.append(title);
    if (row.sold != null) line(row.sold.toLocaleString('en-IN'), 'units sold (simulated)', CHART_COLORS.sold);
    if (row.demand != null) line(row.demand.toLocaleString('en-IN'), 'true demand (simulated)', CHART_COLORS.demand, 'dotted');
    if (row.forecast != null) line(row.forecast.toLocaleString('en-IN'), 'forecast', CHART_COLORS.demand, 'dashed');
    if (row.stockout) line('Stockout', 'day (simulated)');
    if (row.festival) line(row.festival, '');
    tooltip.hidden = false;
    const left = (chart.x(i) / chart.W) * box.width;
    tooltip.style.left = `${Math.min(Math.max(left + 12, 0), box.width - 190)}px`;
  });
  svg.querySelector('.hit').addEventListener('pointerleave', hide);
}
function legendItem(label, color, kind) {
  const mark = kind === 'area' ? `<span class="swatch" style="background:${color}"></span>` : `<span class="key ${kind || ''}" style="border-color:${color}"></span>`;
  return `<span class="legend-item">${mark}${escapeHtml(label)}</span>`;
}
function renderProductPanel(product) {
  const d = product.details;
  const tracked = product.history.length > 0;
  const chart = productChart(product);
  const fmtNumber = (value, digits = 0) => value == null ? '–' : Number(value).toLocaleString('en-IN', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const facts = [
    ['Brand', escapeHtml(d.brand || '–')], ['Category', escapeHtml(d.category || '–')], ['MRP', d.price == null ? '–' : `₹${fmtNumber(d.price, 2)}`],
    ['Stock (simulated)', `${fmtNumber(product.stock.units)} units on ${escapeHtml(shortDate(product.stock.date))}`],
    ['Days of cover', `<span class="${d.risk_status === 'stockout' ? 'risk' : ''}">${fmtNumber(d.days_of_cover, 1)} days · ${escapeHtml(d.risk_status)}</span>`],
    ['Next delivery', `in ${fmtNumber(d.days_until_delivery)} day${d.days_until_delivery === 1 ? '' : 's'}`], ['Reorder qty', `${fmtNumber(d.reorder_qty)} units`],
    ['Daily forecast (used for risk)', `${fmtNumber(d.forecast_daily, 1)} units`],
    ['Forecast source', d.forecast_source === 'lightgbm_direct' ? 'direct (LightGBM)' : 'analog (no history)'],
  ];
  const legend = [
    tracked ? legendItem('Units sold (simulated)', CHART_COLORS.sold) : '',
    tracked ? legendItem('True demand (simulated)', CHART_COLORS.demand, 'dotted') : '',
    legendItem(tracked ? '7-day forecast' : '7-day forecast (estimated, from 5 analogs)', CHART_COLORS.demand, 'dashed'),
    tracked ? legendItem('Stockout day (simulated)', CHART_COLORS.stockout, 'area') : '',
    legendItem('Stock (simulated)', CHART_COLORS.text), product.festivals.length ? legendItem('Festival', CHART_COLORS.muted) : '',
  ].join('');
  const analogs = product.analogs.length ? `<h3>Built from ${product.analogs.length} analog products</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Weight</th><th>Price factor</th><th>Its forecast/day</th></tr></thead><tbody>${product.analogs.map(a => `<tr><td>${productLink(a.product_name, a.sku_id, product.city, 'regional')}</td><td>${(a.weight * 100).toFixed(1)}%</td><td>×${a.price_adjustment.toFixed(2)}</td><td>${fmtNumber(a.forecast_daily, 1)}</td></tr>`).join('')}</tbody></table></div><p class="subhead">Forecast = Σ weight × price factor × analog forecast (estimated). Weights are normalised inverse distances on price, pack size and brand popularity.</p>` : '';
  const rows = chartRows(product);
  const table = `<details class="chart-table"><summary>Data table</summary><div class="table-wrap"><table><thead><tr><th>Date</th><th>Units sold</th><th>True demand</th><th>Forecast</th><th>Stockout</th><th>Festival</th></tr></thead><tbody>${rows.map(row => `<tr><td>${escapeHtml(row.date)}</td><td>${fmtNumber(row.sold)}</td><td>${fmtNumber(row.demand, 1)}</td><td>${fmtNumber(row.forecast, 1)}</td><td>${row.stockout ? 'yes' : ''}</td><td>${escapeHtml(row.festival || '')}</td></tr>`).join('')}</tbody></table></div></details>`;
  document.getElementById('product-panel-title').textContent = d.product_name;
  document.getElementById('product-panel-body').innerHTML = `<div class="subhead">${escapeHtml(product.city)} · ${escapeHtml(product.sku_id)} · ${product.scope === 'model' ? 'model risk report' : 'regional report'} · today ${escapeHtml(shortDate(product.today))}</div>
    <div class="product-layout"><div><div class="chart-legend">${legend}</div><div class="chart-box">${chart.svg}<div class="chart-tooltip" hidden></div></div></div>
    <dl class="product-facts">${facts.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join('')}</dl></div>
    ${festivalNote(product)}<div class="explain" id="product-explain"><h3>Why this forecast?</h3><p class="subhead">Loading…</p></div>${analogs}${table}<p class="subhead">${product.notes.map(escapeHtml).join(' ')}</p>`;
  attachChartHover(document.querySelector('#product-panel-body .chart-box'), chart, product);
}
function festivalNote(product) {
  const plan = product.festival_planning;
  if (!plan) return '';
  const fmtNumber = (value, digits = 0) => Number(value).toLocaleString('en-IN', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  return `<div class="occasion" style="margin-top:14px"><strong>Festival planning: ${escapeHtml(plan.festival)} (${shortDate(plan.date)})</strong><br>7-day forecast ${fmtNumber(plan.seven_day_units, 1)} units × ${escapeHtml(plan.category)} uplift ×${plan.uplift.toFixed(2)} = <strong>${fmtNumber(plan.festival_week_units)} units</strong> in the festival week.<br><span class="subhead">Uplift: ${escapeHtml(product.city)} ${escapeHtml(plan.category)} demand in the week ending on ${escapeHtml(plan.festival)} vs the 4 normal weeks before it, averaged over ${plan.occurrences} past years (${plan.first_year}–${plan.last_year}, simulated sales history). Planning estimate, not a model forecast (the model forecasts 7 days ahead).</span></div>`;
}
const DRIVER_COLORS = {up: '#2a78d6', down: '#eb6834', bar: '#1f8f4d'};
function signedUnits(value) {
  const size = Math.abs(value);
  return `${value < 0 ? '−' : '+'}${size.toLocaleString('en-IN', {maximumFractionDigits: size < 10 ? 1 : 0})}`;
}
function driverBars(rows, options) {
  // horizontal bars from a zero line; rows: [{label, value, title}]
  const W = options.W || 540, rowH = 26, labelW = options.labelW || 230, valueW = options.valueW || 64, top = 6;
  const H = top + rows.length * rowH + 6;
  const diverging = rows.some(row => row.value < 0);
  const max = Math.max(1e-9, ...rows.map(row => Math.abs(row.value)));
  const plotW = W - labelW - valueW;
  const zero = diverging ? labelW + plotW / 2 : labelW;
  const scale = (diverging ? plotW / 2 : plotW) / max;
  const parts = [`<line x1="${zero}" x2="${zero}" y1="${top}" y2="${H - 6}" stroke="${CHART_COLORS.grid}" stroke-width="1"/>`];
  rows.forEach((row, i) => {
    const y = top + i * rowH, w = Math.max(1, Math.abs(row.value) * scale);
    const x = row.value < 0 ? zero - w : zero;
    const color = options.color || (row.value < 0 ? DRIVER_COLORS.down : DRIVER_COLORS.up);
    const valueX = row.value < 0 ? x - 4 : x + w + 4;
    parts.push(`<g><title>${escapeHtml(row.title || '')}</title><rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`);
    parts.push(`<text x="${labelW - 8}" y="${y + rowH / 2 + 3}" text-anchor="end" font-size="10" fill="${CHART_COLORS.text}">${escapeHtml(row.label)}</text>`);
    parts.push(`<rect x="${x}" y="${y + 6}" width="${w}" height="${rowH - 12}" rx="3" fill="${color}"/>`);
    parts.push(`<text x="${valueX}" y="${y + rowH / 2 + 3}" text-anchor="${row.value < 0 ? 'end' : 'start'}" font-size="10" fill="${CHART_COLORS.text}">${escapeHtml(row.text)}</text></g>`);
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(options.aria)}">${parts.join('')}</svg>`;
}
function renderExplanation(payload) {
  const box = document.getElementById('product-explain');
  if (!box) return;
  const e = payload.explanation;
  const fmt = value => Number(value).toLocaleString('en-IN', {maximumFractionDigits: 1});
  const rows = e.drivers.map(d => ({label: d.label, value: d.units, text: `${signedUnits(d.units)}/day`,
    title: `${d.label}: ${signedUnits(d.units)} units/day (approximate, from SHAP in log space)`}));
  const rest = e.rest.count ? ` + ${e.rest.count} smaller driver${e.rest.count === 1 ? '' : 's'} (${signedUnits(e.rest.units)})` : '';
  const analog = payload.analog ? `<p class="explain-summary"><strong>Borrowed from ${payload.analog.count} similar products.</strong> This product has no sales history; its forecast (${fmt(payload.forecast_daily)} units/day, estimated) is a weighted, price-adjusted average of ${payload.analog.count} tracked products. Below: why the model forecasts the top-weighted one, ${productLink(payload.analog.product_name, payload.analog.sku_id, payload.city, 'regional')} (weight ${(payload.analog.weight * 100).toFixed(0)}%, price factor ×${payload.analog.price_adjustment.toFixed(2)}).</p>` : '';
  const table = `<details class="chart-table"><summary>Drivers table</summary><div class="table-wrap"><table><thead><tr><th>Driver</th><th>Units/day</th></tr></thead><tbody><tr><td>Baseline</td><td>${fmt(e.baseline)}</td></tr>${e.all_drivers.map(d => `<tr><td>${escapeHtml(d.label)}</td><td>${signedUnits(d.units)}</td></tr>`).join('')}<tr><td><strong>Forecast</strong></td><td><strong>${fmt(e.forecast)}</strong></td></tr></tbody></table></div></details>`;
  box.innerHTML = `<h3>Why this forecast?</h3>${analog}<p class="explain-summary">Baseline <strong>${fmt(e.baseline)}</strong> units/day ${e.drivers.map(d => signedUnits(d.units)).join(' ')}${rest} = forecast <strong>${fmt(e.forecast)}</strong> units/day (average of ${escapeHtml(shortDate(e.window.start))}–${escapeHtml(shortDate(e.window.end))})</p>
    <div class="chart-legend">${legendItem('Raises the forecast', DRIVER_COLORS.up, 'area')}${legendItem('Lowers the forecast', DRIVER_COLORS.down, 'area')}</div>
    ${driverBars(rows, {aria: `Top ${rows.length} drivers of the forecast for ${e.sku_id} in ${e.city}`})}${table}
    <p class="subhead">${payload.notes.map(escapeHtml).join(' ')} Baseline = the model's starting point (its average log forecast, as units). Day-of-week effects largely cancel over a full 7-day window.</p>`;
}
async function loadExplanation(sku, city) {
  const response = await fetch(`/api/explain?${new URLSearchParams({sku_id: sku, city})}`);
  const payload = await response.json();
  if (!openProduct || openProduct.sku !== sku) return;  // another product was opened meanwhile
  const box = document.getElementById('product-explain');
  if (!box) return;
  if (!response.ok) { box.innerHTML = `<h3>Why this forecast?</h3><p class="subhead">${escapeHtml(payload.error || 'No explanation available')}</p>`; return; }
  renderExplanation(payload);
}
async function renderDriversCard() {
  const response = await fetch('/api/explain/importance');
  const payload = await response.json();
  if (!response.ok) { document.getElementById('drivers-caption').textContent = payload.error || 'Driver importance unavailable'; return; }
  Object.entries(payload.segments).forEach(([segment, block]) => {
    const rows = block.features.map(f => ({label: f.label, value: f.mean_abs_shap, text: `${f.mean_abs_shap.toFixed(3)} · ~${f.typical_effect_pct.toFixed(0)}%`,
      title: `${f.label} (${f.feature}): mean |SHAP| ${f.mean_abs_shap.toFixed(3)} on the log scale, a typical ~${f.typical_effect_pct.toFixed(0)}% effect on the forecast`}));
    const box = document.getElementById(`drivers-${segment}`);
    if (box) box.innerHTML = driverBars(rows, {W: 400, labelW: 140, valueW: 84, color: DRIVER_COLORS.bar, aria: `Top ${rows.length} demand drivers, ${segment} model`});
  });
  document.getElementById('drivers-scope').textContent = `· global importance, top 10 features per model`;
  document.getElementById('drivers-caption').textContent = `Mean |SHAP| per feature on the evaluation test window (${shortDate(payload.window[0])}–${shortDate(payload.window[1])} ${payload.window[1].slice(0, 4)}, simulated sales history), in the model's log scale; ~% = the typical effect on a forecast. "Product" is the product's own identity code: what the model learned about each item beyond price, brand and pack size.`;
}
async function openProductPanel(sku, city, scope, opener, festival = '') {
  const panel = document.getElementById('product-panel');
  productPanelOpener = opener || productPanelOpener;
  openProduct = {sku, city, scope, festival};
  saveUrlState();
  panel.classList.add('open');
  panel.setAttribute('aria-hidden', 'false');
  document.getElementById('product-panel-title').textContent = 'Loading…';
  document.getElementById('product-panel-body').innerHTML = '';
  document.getElementById('product-panel-close').focus();
  const response = await fetch(`/api/product?${new URLSearchParams({sku_id: sku, city, scope, festival: festival || ''})}`);
  const payload = await response.json();
  if (!response.ok) { document.getElementById('product-panel-title').textContent = payload.error || 'Unable to load product'; return; }
  renderProductPanel(payload);
  loadExplanation(payload.sku_id, payload.city).catch(error => console.error(error));
}
function closeProductPanel() {
  const panel = document.getElementById('product-panel');
  if (!panel.classList.contains('open')) return;
  panel.classList.remove('open');
  panel.setAttribute('aria-hidden', 'true');
  openProduct = null;
  saveUrlState();
  if (productPanelOpener) productPanelOpener.focus();
  productPanelOpener = null;
}
document.addEventListener('click', event => {
  const link = event.target.closest('.product-link');
  if (!link) return;
  event.preventDefault();
  openProductPanel(link.dataset.sku, link.dataset.city, link.dataset.scope, link, link.dataset.festival).catch(error => showToast(error.message, true));
});
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeProductPanel(); });
document.getElementById('product-panel-close').addEventListener('click', closeProductPanel);
const READY_FILTERS = ['status', 'category', 'brand', 'sort', 'order', 'page'];
const readinessState = {status: '', category: '', brand: '', sort: '', order: '', page: '0'};
let readinessRequest = 0;
const READY_COLUMNS = [['Product'], ['Brand'], ['Category'], ['Current stock', 'stock'], ['Festival-week demand', 'demand'], ['Available (stock + regular deliveries)'], ['Shortfall', 'shortfall'], ['Order by', 'order_by'], ['Extra order (above regular)', 'extra'], ['Status']];
const STATUS_CLASS = {'Order now': 'cover', 'At risk': 'cover high', 'Ready': 'safe'};
function readinessQuery(extra = {}) {
  const query = new URLSearchParams({city: selectedRegion, festival: planFestival});
  READY_FILTERS.filter(key => key !== 'page').forEach(key => { if (readinessState[key]) query.set(key, readinessState[key]); });
  Object.entries(extra).forEach(([key, value]) => query.set(key, value));
  return query;
}
function setReadiness(changes) { Object.assign(readinessState, changes); if (!('page' in changes)) readinessState.page = '0'; saveUrlState(); renderReadiness().catch(error => console.error(error)); }
async function renderReadiness() {
  const request = ++readinessRequest;
  const offset = Number(readinessState.page) * STOCKOUT_PAGE_SIZE;
  const response = await fetch(`/api/readiness?${readinessQuery({offset: String(offset), limit: String(STOCKOUT_PAGE_SIZE)})}`);
  const r = await response.json();
  if (request !== readinessRequest) return;
  const body = document.getElementById('ready-body');
  if (!response.ok) { body.innerHTML = `<tr><td class="empty" colspan="10">${escapeHtml(r.error || 'Unable to load readiness')}</td></tr>`; return; }
  const weekStart = addDays(isoDay(r.date), -6);
  document.getElementById('ready-title').textContent = `${r.festival} readiness (${shortDate(r.date)}, in ${r.days_until} days)`;
  document.getElementById('readiness-kpi').innerHTML = `<span>${escapeHtml(r.festival)} readiness · order now</span><strong style="color:var(--red)">${r.counts['Order now'].toLocaleString()}</strong><small class="metric-note">${r.counts['At risk'].toLocaleString()} at risk · ${r.counts.Ready.toLocaleString()} ready · planning estimate</small>`;
  document.getElementById('ready-tabs').innerHTML = regionalData.map(item => `<button type="button" class="filter ${item.region === r.city ? 'active' : ''}" data-ready-city="${escapeHtml(item.region)}">${escapeHtml(item.region)}</button>`).join('');
  document.getElementById('ready-status').innerHTML = `<option value="">All statuses</option>${['Order now', 'At risk', 'Ready'].map(status => `<option value="${status}">${status} (${r.counts[status].toLocaleString()})</option>`).join('')}`;
  document.getElementById('ready-status').value = readinessState.status;
  document.getElementById('ready-category').innerHTML = `<option value="">All categories</option>${r.categories.map(category => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join('')}`;
  document.getElementById('ready-category').value = readinessState.category;
  document.getElementById('ready-brand-list').innerHTML = r.brands.map(brand => `<option value="${escapeHtml(brand)}"></option>`).join('');
  if (document.activeElement !== document.getElementById('ready-brand')) document.getElementById('ready-brand').value = readinessState.brand;
  document.getElementById('ready-count').textContent = `${r.total.toLocaleString()} of ${r.products.toLocaleString()} products`;
  document.getElementById('ready-download').href = `/api/readiness?${readinessQuery({format: 'csv'})}`;
  document.getElementById('ready-caption').textContent = `Planning estimate · stock simulated · top-up above regular orders. Festival-week demand (${shortDate(weekStart)}–${shortDate(r.date)}) = 7-day forecast × the category's ${r.festival} uplift in ${r.city} (measured on past years). Available = stock still usable at the start of the festival week (stock on ${shortDate(r.stock_date)} − forecast demand before the week + regular deliveries before it; perishables only if still within shelf life on ${shortDate(r.date)}) + the regular deliveries during the week, each one cycle of normal demand. Shortfall = demand − available; gaps under max(5 units, 10% of demand) count as Ready. Order now: a shortfall, and the last delivery before the festival arrives within 7 days (orders take 7 days). Extra order = the quantity on top of regular orders: shortfall + 2 days of festival demand.`;
  document.getElementById('ready-head').innerHTML = `<tr>${READY_COLUMNS.map(([label, key]) => {
    if (!key) return `<th>${label}</th>`;
    const arrow = readinessState.sort === key ? ((readinessState.order || (key === 'order_by' ? 'asc' : 'desc')) === 'asc' ? ' ↑' : ' ↓') : '';
    return `<th class="sortable" data-ready-sort="${key}">${label}${arrow}</th>`;
  }).join('')}</tr>`;
  body.innerHTML = r.rows.map(row => {
    const due = row.order_by && row.status !== 'Ready' ? `${shortDate(row.order_by)}${row.order_by <= r.data_date ? ' (now)' : ''}` : '–';
    const gap = row.status === 'Ready' ? (row.shortfall > 0 ? `<span class="subhead" title="below max(5 units, 10% of demand)">small gap ${fmt(row.shortfall)}</span>` : '–') : fmt(row.shortfall);
    return `<tr><td>${productLink(row.product_name, row.sku_id, row.city, 'regional', r.festival)}</td><td>${escapeHtml(row.brand)}</td><td>${escapeHtml(row.category)}</td><td>${fmt(row.opening_stock)}</td><td>${fmt(row.festival_week_units)}</td><td title="usable at week start ${fmt(row.usable_at_week_start)} + regular deliveries in the week ${fmt(row.week_deliveries)}">${fmt(row.available)}</td><td class="${row.status !== 'Ready' ? 'risk' : ''}">${gap}</td><td>${due}</td><td>${row.extra_order ? fmt(row.extra_order) : '–'}</td><td><span class="${STATUS_CLASS[row.status]}">${escapeHtml(row.status)}</span></td></tr>`;
  }).join('') || `<tr><td class="empty" colspan="10">No products match these filters</td></tr>`;
  const pageCount = Math.max(1, Math.ceil(r.total / STOCKOUT_PAGE_SIZE));
  document.getElementById('ready-page').textContent = r.total ? `Products ${offset + 1}-${Math.min(offset + STOCKOUT_PAGE_SIZE, r.total)} of ${r.total.toLocaleString()}` : 'No products';
  document.getElementById('ready-prev').disabled = Number(readinessState.page) === 0;
  document.getElementById('ready-next').disabled = Number(readinessState.page) >= pageCount - 1;
}
function attachReadinessControls() {
  document.getElementById('ready-status').addEventListener('change', event => setReadiness({status: event.target.value}));
  document.getElementById('ready-category').addEventListener('change', event => setReadiness({category: event.target.value}));
  document.getElementById('ready-brand').addEventListener('change', event => setReadiness({brand: event.target.value.trim()}));
  document.getElementById('ready-clear').addEventListener('click', () => { document.getElementById('ready-brand').value = ''; setReadiness({status: '', category: '', brand: '', sort: '', order: ''}); });
  document.getElementById('ready-prev').addEventListener('click', () => setReadiness({page: String(Math.max(0, Number(readinessState.page) - 1))}));
  document.getElementById('ready-next').addEventListener('click', () => setReadiness({page: String(Number(readinessState.page) + 1)}));
  document.getElementById('ready-head').addEventListener('click', event => {
    const key = event.target.closest('[data-ready-sort]')?.dataset.readySort;
    if (!key) return;
    const defaultOrder = key === 'order_by' ? 'asc' : 'desc';
    const current = readinessState.sort === key ? (readinessState.order || defaultOrder) : '';
    setReadiness({sort: key, order: current ? (current === 'asc' ? 'desc' : 'asc') : defaultOrder});
  });
  document.getElementById('ready-tabs').addEventListener('click', event => {
    const city = event.target.closest('[data-ready-city]')?.dataset.readyCity;
    if (!city || city === selectedRegion) return;
    selectedRegion = city; saveUrlState(); renderRegional(); refreshStockoutViews(); loadDashboard(selectedRegion).catch(error => console.error(error));
  });
}
async function load() {
  const isAdmin = document.body.classList.contains('admin-page');
  if (!isAdmin) {
    categories = await (await fetch('/api/categories')).json();
    renderCategories();
    document.getElementById('store-search').addEventListener('submit', event => {
      event.preventDefault();
      document.getElementById('catalog').scrollIntoView({ behavior: 'smooth' });
    });
    document.getElementById('search').addEventListener('input', () => {
      if (document.getElementById('search').value.trim()) selectedCategory = '';
      customerPage = 0;
      clearTimeout(customerSearchTimer);
      customerSearchTimer = setTimeout(() => renderCatalog().catch(error => {
        document.getElementById('catalog').innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
      }), 250);
    });
    document.getElementById('customer-prev').addEventListener('click', () => {
      if (customerPage > 0) { customerPage -= 1; renderCatalog(); }
    });
    document.getElementById('customer-next').addEventListener('click', () => {
      if (customerPage < customerPageCount - 1) {
        customerPage += 1;
        renderCatalog();
      }
    });
    document.getElementById('basket-toggle').addEventListener('click', () => document.getElementById('basket-panel').classList.add('open'));
    document.getElementById('basket-close').addEventListener('click', () => document.getElementById('basket-panel').classList.remove('open'));
    document.getElementById('checkout').addEventListener('click', () => document.getElementById('payment').classList.add('open'));
    document.getElementById('place-order').addEventListener('click', () => {
      if (!basket.length) return;
      alert('Demo order placed successfully.');
      basket = [];
      renderBasket();
      document.getElementById('payment').classList.remove('open');
    });
    await renderCatalog();
    renderBasket();
    return;
  }

  document.getElementById('alert-preview').addEventListener('click', () => requestAlert(false));
  document.getElementById('alert-send').addEventListener('click', () => requestAlert(true));
  readUrlState();
  const simulated = 'Simulated data: sales are synthetic POS history; stock, days until stockout, next delivery and reorder qty come from simulated stock.';
  createStockoutView('m', 'model', 'model-view', `Tracked products, stock counted on {day}. ${simulated} Trend = last 28 days vs the 28 before.`);
  createStockoutView('r', 'regional', 'regional-view', `Full catalog, stock simulated for {day}. ${simulated} Catalog products without sales history show their forecast instead.`);
  refreshStockoutViews();
  if (openProduct) openProductPanel(openProduct.sku, openProduct.city, openProduct.scope, null, openProduct.festival).catch(error => showToast(error.message, true));
  document.getElementById('plan-festival').addEventListener('change', event => { planFestival = event.target.value; readinessState.page = '0'; saveUrlState(); renderFestivalPlanning().catch(error => console.error(error)); renderReadiness().catch(error => console.error(error)); });
  attachReadinessControls();
  renderDriversCard().catch(error => console.error(error));
  await loadDashboard(selectedRegion);
  regionalData = await (await fetch('/api/regional-intelligence')).json();
  renderRegional();
  document.getElementById('admin-search').addEventListener('input', () => {
    adminProductPage = 0;
    clearTimeout(adminSearchTimer);
    adminSearchTimer = setTimeout(() => renderAdminProducts().catch(error => console.error(error)), 250);
  });
  document.getElementById('product-prev').addEventListener('click', () => { adminProductPage -= 1; renderAdminProducts(); });
  document.getElementById('product-next').addEventListener('click', () => { adminProductPage += 1; renderAdminProducts(); });
  await renderAdminProducts();
}
let toastTimer;
function showToast(message, isError = false) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = `toast show${isError ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = 'toast'; }, 6000);
}
function alertSummary(result) {
  const counts = `${result.counts.stockout} stockout, ${result.counts.overstock} overstock`;
  if (result.status === 'sent') return `Sent to ${result.recipients.join(', ')} (${counts})`;
  if (result.status === 'dry_run') return `Preview ready (${counts}); not sent`;
  if (result.status === 'skipped') return `${result.vendor_name}: nothing to send (${result.detail})`;
  return `Failed for ${result.vendor_name}: ${result.detail}`;
}
async function requestAlert(send) {
  const city = selectedRegion;
  if (send && !confirm(`Email the ${city} vendor now? This sends a real email.`)) return;
  // open the preview tab before awaiting, so it isn't blocked as a popup
  const preview = send ? null : window.open('', '_blank');
  const buttons = [document.getElementById('alert-preview'), document.getElementById('alert-send')];
  buttons.forEach(button => { button.disabled = true; });
  try {
    const response = await fetch(`/api/send-alert?${new URLSearchParams({city, send: send ? '1' : '0'})}`, {
      method: 'POST', headers: {'X-Alert-Request': '1'},
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    const results = payload.results;
    const failed = results.some(result => result.status === 'failed');
    showToast(`${payload.city}: ${results.map(alertSummary).join(' · ')}`, failed);
    const previewUrl = results.find(result => result.preview_url)?.preview_url;
    if (preview) { if (previewUrl) preview.location = previewUrl; else preview.close(); }
  } catch (error) {
    if (preview) preview.close();
    showToast(error.message, true);
  } finally {
    buttons.forEach(button => { button.disabled = false; });
  }
}
async function loadDashboard(city) {
  const response = await fetch(`/api/dashboard?${new URLSearchParams({city: city || 'all'})}`);
  if (!response.ok) throw new Error('Unable to load forecast KPIs.');
  data = await response.json();
  renderSummary();
}
function renderSummary() {
  const statusEl = document.getElementById('status');
  const evaluation = data.evaluation_period ? `${shortDate(data.evaluation_period[0])}–${longDate(data.evaluation_period[1])}` : '–';
  if (statusEl) statusEl.textContent = data.data_date ? `Data as of ${longDate(data.data_date)} · Evaluation period (accuracy figures): ${evaluation}` : 'NO RISK DATA';
  const festival = data.next_festival;
  const festivalLabel = festival ? `${escapeHtml(festival.name)} · ${festival.days_until}d` : 'None scheduled';
  const riskCaption = `Model forecast · ${data.tracked_products ?? '–'} tracked products · stock on ${data.latest ? shortDate(data.latest) : '–'} (simulated)`;
  const segmentNames = {perishable: 'Perishable', nonperishable: 'Non-perishable'};
  const segments = (data.metrics && data.metrics.segments) || {};
  const wapeLabel = Object.entries(segments).map(([segment, metric]) =>
    `${escapeHtml(segmentNames[segment] || segment)} ${metric.final_test.overall['WAPE_%']}%`
  ).join('<br>') || 'n/a';
  document.getElementById('summary').innerHTML =`<div class="metric"><span>Stockout risk${data.latest ? ` on ${shortDate(data.latest)}` : ''}</span><strong class="red">${data.stockout.length} products</strong><small class="metric-note">${riskCaption}</small></div><div class="metric"><span>Overstock risk${data.latest ? ` on ${shortDate(data.latest)}` : ''}</span><strong class="amber">${data.overstock.length} products</strong><small class="metric-note">${riskCaption}</small></div><div class="metric"><span>Next festival</span><strong>${festivalLabel}</strong></div><div class="metric"><span>Forecast WAPE</span><strong>${wapeLabel}</strong><small class="metric-note">Evaluation period ${evaluation} (held-out test window), not the serving date</small></div>`;
  renderModelRisk();
}
function renderModelRisk() {
  const cover = row => `${money(row.days_of_cover)} days`;
  const emptyRow = columns => `<tr><td colspan="${columns}" class="empty">None today</td></tr>`;
  const overstock = [...data.overstock].sort((a, b) => b.days_of_cover - a.days_of_cover).slice(0, 15);
  document.getElementById('model-risk-scope').textContent = data.latest ? `${data.city === 'all' ? 'All cities' : selectedRegion} · stock on ${longDate(data.latest)}, forecast for the next 7 days` : '';
  document.getElementById('model-overstock').innerHTML = overstock.map(row => `<tr><td>${productLink(row.product_name, row.sku_id, row.city, 'model')}</td><td>${escapeHtml(row.brand)}</td><td class="risk">${cover(row)}</td><td>${escapeHtml(row.shelf_life_days)} days</td></tr>`).join('') || emptyRow(4);
}
load().catch(error => {
  console.error('Failed to load dashboard data:', error);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = 'DATA UNAVAILABLE';
  const catalogEl = document.getElementById('catalog');
  if (catalogEl) catalogEl.innerHTML = `<div class="empty">${escapeHtml(error.message || 'Store data is unavailable.')}</div>`;
});
</script>
</body></html>"""

ADMIN_PAGE = PAGE.replace('<body>', '<body class="admin-page">')


class DashboardHandler(SimpleHTTPRequestHandler):
    def send_json_not_found(self, error, name):
        body = json.dumps({"error": error, "name": name}).encode("utf-8")
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_stockout(self, path, params):
        """/api/stockout: filtered, sorted rows ({total, rows}, or every row as CSV with format=csv); /api/stockout/brands: per brand."""
        scope = params.get("scope", "model")
        try:
            table = stockout_view.table_for(scope)
            if path == "/api/stockout/brands":
                brands = stockout_view.brand_summary(stockout_view.filter_rows(table, params, skip={"brand"}))
                in_city = stockout_view.filter_rows(table, {"city": params.get("city", "")})
                categories = in_city["category"].value_counts().sort_index()
                self.send_json(200, {
                    "total": len(brands), "rows": stockout_view.to_records(brands),
                    "categories": [{"category": name, "products": int(count)} for name, count in categories.items() if name],
                })
                return
            rows = stockout_view.sort_rows(stockout_view.filter_rows(table, params),
                                           params.get("sort", ""), params.get("order", ""))
            in_city = len(stockout_view.filter_rows(table, {"city": params.get("city", "")}))
            shown = stockout_view.page(rows, params.get("offset", ""), params.get("limit", ""))
        except stockout_view.UnknownCity as error:
            self.send_json_not_found("Unknown city", str(error))
            return
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        except FileNotFoundError as error:
            self.send_json(503, {"error": f"{Path(error.filename).name} not generated; run run_pipeline.py"})
            return
        if params.get("format") == "csv":
            body = stockout_view.to_csv(rows).encode("utf-8")
            city = re.sub(r"[^\w-]+", "_", params.get("city", "") or "all")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="stockout_{scope}_{city}.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json(200, {"total": len(rows), "rows": stockout_view.to_records(shown), "products_in_city": in_city,
                             "planning_day": table.attrs.get("planning_day")})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/send-alert":
            self.send_json_not_found("Not found", path)
            return
        # Only the dashboard page may call this: a custom header can't be sent
        # cross-site without a CORS preflight (which this server never allows),
        # and the Host check stops DNS-rebinding pages.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if self.headers.get("X-Alert-Request") != "1" or host not in {"127.0.0.1", "localhost"}:
            self.send_json(403, {"error": "Forbidden"})
            return
        import alerts

        query = parse_qs(urlparse(self.path).query)
        city = query.get("city", [""])[0].strip()
        send = query.get("send", ["0"])[0] == "1"
        if not city or city.casefold() == "all":
            self.send_json(400, {"error": "Choose one city"})
            return
        try:
            results = alerts.send_alert(city, send=send)
        except alerts.AlertError as error:
            self.send_json(400, {"error": str(error)})
            return
        for result in results:
            if "html_path" in result:
                result["preview_url"] = f"/alerts/{quote(Path(result.pop('html_path')).name)}"
                result.pop("csv_path", None)
        self.send_json(200, {"city": results[0]["city"] if results else city, "send": send, "results": results})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/customer", "/admin"}:
            page = ADMIN_PAGE if path == "/admin" else PAGE
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/catalog":
            query = parse_qs(urlparse(self.path).query)
            if "offset" in query or "limit" in query or "search" in query or "category" in query:
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(100, max(1, int(query.get("limit", [24])[0])))
                search = query.get("search", [""])[0]
                category = query.get("category", [""])[0]
                body = json.dumps(catalog_page(offset, limit, search, category)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(read_catalog()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/dashboard":
            payload = dashboard_payload(parse_qs(urlparse(self.path).query).get("city", [""])[0])
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/product":
            params = {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}
            try:
                payload = product_detail.product_detail(params.get("city", ""), params.get("sku_id", ""),
                                                        params.get("scope", "regional"), params.get("festival", ""))
            except product_detail.ProductNotFound:
                self.send_json_not_found("Product not found", params.get("sku_id", ""))
                return
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            except FileNotFoundError as error:
                self.send_json(503, {"error": f"{Path(error.filename).name} not generated; run run_pipeline.py"})
                return
            self.send_json(200, payload)
            return
        if path in {"/api/explain", "/api/explain/importance"}:
            params = {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}
            try:
                payload = (explain_forecast.importance_payload() if path == "/api/explain/importance"
                           else explain_forecast.explain(params.get("city", ""), params.get("sku_id", "")))
            except explain_forecast.ExplanationNotFound:
                self.send_json_not_found("No explanation for this product", params.get("sku_id", ""))
                return
            except FileNotFoundError as error:
                self.send_json(503, {"error": f"{Path(error.filename).name} not generated; run run_pipeline.py"})
                return
            self.send_json(200, payload)
            return
        if path == "/api/readiness":
            params = {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}
            try:
                rows, payload = festival_readiness.readiness_payload(params)
            except stockout_view.UnknownCity as error:
                self.send_json_not_found("Unknown city", str(error))
                return
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            except FileNotFoundError as error:
                self.send_json(503, {"error": f"{Path(error.filename).name} not generated; run run_pipeline.py"})
                return
            if params.get("format") == "csv":
                body = festival_readiness.to_csv(rows).encode("utf-8")
                name = re.sub(r"[^\w-]+", "_", f"{payload['festival']}_readiness_{payload['city']}").lower()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{name}.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json(200, payload)
            return
        if path == "/api/festival-planning":
            params = {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}
            try:
                payload = festival_uplift.festival_planning(params.get("city", ""), params.get("festival", ""),
                                                            params.get("rank", "units"))
            except stockout_view.UnknownCity as error:
                self.send_json_not_found("Unknown city", str(error))
                return
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            except FileNotFoundError as error:
                self.send_json(503, {"error": f"{Path(error.filename).name} not generated; run run_pipeline.py"})
                return
            self.send_json(200, payload)
            return
        if path in {"/api/stockout", "/api/stockout/brands"}:
            self.send_stockout(path, {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()})
            return
        if path == "/api/categories":
            body = json.dumps(read_csv_from_path(DATA_DIR / "bigbasket_categories.csv")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/regional-intelligence":
            body = json.dumps(regional_intelligence()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/datasets":
            body = json.dumps(dataset_info()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/dataset":
            query = parse_qs(urlparse(self.path).query)
            name = query.get("name", [""])[0]
            offset = max(0, int(query.get("offset", [0])[0]))
            limit = min(100, max(1, int(query.get("limit", [100])[0])))
            rows = dataset_preview(name, offset, limit)
            if rows is None:
                self.send_json_not_found("Dataset not found", name)
                return
            body = json.dumps({"rows": rows}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/alerts/"):
            import alerts

            name = path.removeprefix("/alerts/")
            file_path = alerts.ALERTS_DIR / name
            if not re.fullmatch(r"[\w.-]+\.html", name) or not file_path.is_file():
                self.send_json_not_found("Alert preview not found", name)
                return
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/download/"):
            name = path.removeprefix("/download/")
            file_path = ARCHIVE_PATH if name == "archive.zip" else dataset_paths().get(name)
            if file_path is None or not file_path.exists():
                self.send_json_not_found("File not found", name)
                return
            body = file_path.read_bytes()
            self.send_response(200)
            content_type = "application/zip" if name == "archive.zip" else "text/csv; charset=utf-8"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, port), DashboardHandler)

    def warm_stockout_tables():
        # build the stockout tables in the background so the first page load is quick
        for scope in stockout_view.SCOPES:
            try:
                stockout_view.table_for(scope)
            except FileNotFoundError:
                pass
    threading.Thread(target=warm_stockout_tables, daemon=True).start()
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()