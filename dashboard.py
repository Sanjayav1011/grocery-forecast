import csv
import io
import json
import re
import sys
import zipfile
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import PROJECT_DIR, ARCHIVE_PATH, ARCHIVE_CSV, DATA_DIR, OUTPUTS_DIR, MODELS_DIR
CATALOG_CACHE = None
STOREFRONT_CATALOG_CACHE = None
REGIONAL_INTELLIGENCE_CACHE = None
NOT_FORECAST_FILE = "bigbasket_not_forecast.csv"  # written by src/forecast_regional.py


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


def next_festival(occasions):
    # Rows with days_until_next_festival == 0 are the festival days themselves.
    today = date.today().isoformat()
    upcoming = [row for row in occasions if row["date"] >= today and row["days_until_next_festival"] == "0"]
    return min(upcoming, key=lambda row: row["date"]) if upcoming else None


def product_key(row):
    """Normalised product name + pack size: the catalog lists some products under several SKUs."""
    name = " ".join(re.sub(r"[^0-9a-z]+", " ", row.get("product_name", "").casefold()).split())
    return name, row.get("pack_size_value", ""), row.get("pack_size_unit", "")


def unique_products(rows):
    """One row per product_key: the directly forecast SKU if there is one, else the first (lowest) SKU id."""
    best = {}
    for row in rows:
        rank = (row["forecast_source"] != "lightgbm_direct", row["sku_id"])
        key = product_key(row)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, row)
    return [row for _, row in best.values()]


def not_forecast_count():
    path = DATA_DIR / NOT_FORECAST_FILE
    return len(read_csv_from_path(path)) if path.exists() else None


def regional_intelligence():
    global REGIONAL_INTELLIGENCE_CACHE
    if REGIONAL_INTELLIGENCE_CACHE is not None:
        return REGIONAL_INTELLIGENCE_CACHE
    rows = read_csv_from_path(DATA_DIR / "bigbasket_regional_inventory.csv")
    occasions = read_csv_from_path(DATA_DIR / "festival_calendar_5yr.csv")
    occasion = next_festival(occasions) or occasions[-1]
    days_until = (date.fromisoformat(occasion["date"]) - date.today()).days
    occasion_name = occasion["next_festival"].lower()
    if date.today().month == 5:
        focus_label = "school start"
        focus_categories = ["book", "stationery", "beverage", "snack", "fruit", "vegetable", "dairy", "milk"]
    elif any(word in occasion_name for word in ["diwali", "ganesh", "mahalaya", "dussehra", "navratri", "holi"]):
        focus_label = occasion["next_festival"]
        focus_categories = ["sweet", "fruit", "beverage", "dairy", "milk", "snack"]
    elif any(word in occasion_name for word in ["christmas", "new year", "eid"]):
        focus_label = occasion["next_festival"]
        focus_categories = ["beverage", "snack", "sweet", "fruit", "dairy", "milk"]
    else:
        focus_label = occasion["next_festival"]
        focus_categories = ["fruit", "vegetable", "dairy", "milk", "beverage", "sweet"]
    result = []
    excluded = not_forecast_count()
    for region in sorted({row["region"] for row in rows}):
        region_rows = [row for row in rows if row["region"] == region]
        stockout = sorted((row for row in region_rows if row["risk_status"] == "stockout"), key=lambda row: float(row["days_of_cover"]))[:10]
        overstock = sorted((row for row in region_rows if row["risk_status"] == "overstock"), key=lambda row: float(row["days_of_cover"]), reverse=True)[:10]
        for row in region_rows:
            product_text = " ".join([row.get("category", ""), row.get("subcategory", ""), row.get("product_name", "")]).lower()
            row["occasion_demand_score"] = float(row["forecast_units"]) * (2.5 if any(category in product_text for category in focus_categories) else 0.5)
        demand = sorted(unique_products(region_rows), key=lambda row: row["occasion_demand_score"], reverse=True)[:10]
        for row in stockout:
            row["recommended_restock_units"] = int(row["reorder_qty"])  # shared rule, inventory.classify_risk
        result.append({
            "region": region,
            "sku_count": len(region_rows),
            "stockout_count": sum(row["risk_status"] == "stockout" for row in region_rows),
            "overstock_count": sum(row["risk_status"] == "overstock" for row in region_rows),
            "occasion": occasion["next_festival"],
            "occasion_date": occasion["date"],
            "days_until_occasion": days_until,
            "demand_focus": focus_label,
            "focus_categories": focus_categories,
            "stockout": stockout,
            "overstock": overstock,
            "demand": demand,
            "not_forecast_count": excluded,
            "not_forecast_download": f"/download/{quote(NOT_FORECAST_FILE)}",
        })
    REGIONAL_INTELLIGENCE_CACHE = result
    return result


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
    .regional-intelligence, .model-risk { margin:0 0 30px; } .region-tabs { display:flex; flex-wrap:wrap; gap:7px; margin:14px 0; } .region-tab { border:1px solid #b9d5be; border-radius:18px; padding:8px 14px; background:#fff; color:#28633d; cursor:pointer; font-size:11px; } .region-tab.active { background:var(--green); color:#fff; } .region-summary { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:12px; } .region-metric { border:1px solid var(--line); border-radius:7px; background:#fff; padding:12px; } .region-metric span { display:block; color:var(--muted); font-size:10px; margin-bottom:5px; } .region-metric strong { font-size:16px; color:var(--text); } .region-metric.alert strong { color:var(--red); } .region-metric.warn strong { color:var(--amber); } .occasion { border:1px solid #cfe5d2; border-radius:7px; background:#eff9f0; padding:12px; margin-bottom:12px; color:#28633d; font-size:12px; } .regional-columns { display:grid; grid-template-columns:1fr 1fr; gap:12px; } .regional-panel { border:1px solid var(--line); border-radius:7px; background:#fff; overflow:hidden; } .regional-panel h3 { margin:0; padding:11px 12px; font-size:12px; color:var(--text); background:#f0f6f0; } .regional-panel table { min-width:0; } .regional-panel td, .regional-panel th { padding:8px 10px; } .regional-panel td { font-size:10px; } .regional-panel .risk { color:var(--red); } .regional-panel .safe { color:var(--green); }
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
    <section class="regional-intelligence admin-only"><div class="catalog-head"><h2>Regional inventory intelligence</h2><span class="subhead" id="regional-caption">Restock decisions by region</span></div><div class="region-tabs" id="region-tabs"></div><div class="region-summary" id="region-summary"></div><div class="occasion" id="occasion"></div><div class="subhead" id="not-forecast" style="margin:-4px 0 12px"></div><div class="regional-columns"><div class="regional-panel"><h3>Stockout risk and recommended restock</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Cover</th><th>Restock</th><th>Forecast source</th></tr></thead><tbody id="regional-stockout"></tbody></table></div></div><div class="regional-panel"><h3>Overstock risk</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Cover</th><th>Action</th><th>Forecast source</th></tr></thead><tbody id="regional-overstock"></tbody></table></div></div></div><div class="regional-panel" style="margin-top:12px"><h3>Highest demand for the upcoming occasion</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Forecast</th><th>Category</th><th>Forecast source</th></tr></thead><tbody id="regional-demand"></tbody></table></div></div></section>
  <div class="summary admin-only" id="summary"></div>
    <section class="model-risk admin-only"><div class="catalog-head"><h2>Model risk report <span id="model-risk-scope" class="subhead"></span></h2><div class="alert-actions"><button type="button" id="alert-preview">Preview alert</button><button type="button" class="send" id="alert-send">Email vendor</button></div></div><div class="regional-columns"><div class="regional-panel"><h3>Stockout risk: reorder before stock runs out</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Brand</th><th>Cover</th><th>Next delivery</th><th>Reorder</th></tr></thead><tbody id="model-stockout"></tbody></table></div></div><div class="regional-panel"><h3>Overstock risk: more than can sell before expiry</h3><div class="table-wrap"><table><thead><tr><th>Product</th><th>Brand</th><th>Cover</th><th>Shelf life</th></tr></thead><tbody id="model-overstock"></tbody></table></div></div></div></section>
    <div class="section-eyebrow customer-only">Picked for your kitchen</div><div class="catalog-head customer-only"><h2>Shop groceries <span id="catalog-count" class="subhead"></span></h2><div class="subhead">Fresh picks for every day</div></div>
  <div class="catalog customer-only" id="catalog"></div>
    <div class="storefront-pagination customer-only"><button id="customer-prev" type="button">Previous</button><span id="customer-page-label"></span><button id="customer-next" type="button">Next</button></div>
    <aside class="basket-panel customer-only" id="basket-panel"><div class="basket-head"><h2>Your basket</h2><button id="basket-close" aria-label="Close basket">&times;</button></div><div id="basket-items"></div><div class="basket-total"><span>Total</span><span id="basket-total">₹0.00</span></div><button class="checkout" id="checkout">Continue to payment</button><div class="payment" id="payment"><h3>Payment</h3><label for="customer-name">Name</label><input id="customer-name" placeholder="Your name"><label for="customer-phone">Phone</label><input id="customer-phone" placeholder="10-digit phone number"><label for="payment-method">Payment method</label><input id="payment-method" value="UPI / Card / Cash on delivery"><button class="checkout" id="place-order">Place demo order</button></div></aside>
    <section class="admin-products admin-only"><div class="catalog-head"><h2>Product catalog details <span id="admin-product-count" class="subhead"></span></h2><input id="admin-search" class="admin-product-search" type="search" placeholder="Search all product details"></div><div class="table-wrap product-admin-table"><table><thead><tr><th>Product</th><th>Category</th><th>Subcategory</th><th>Type</th><th>Brand</th><th>Size</th><th>MRP</th><th>EAN / SKU</th><th>Description</th><th>Image URL</th><th>Product URL</th></tr></thead><tbody id="admin-product-body"></tbody></table></div><div class="dataset-page"><button id="product-prev">Previous</button><span id="product-page-label"></span><button id="product-next">Next</button></div></section>
</main>
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
function renderRegional() { const region = regionalData.find(item => item.region === selectedRegion) || regionalData[0]; if (!region) return; selectedRegion = region.region; document.getElementById('region-tabs').innerHTML = regionalData.map(item => `<button class="region-tab ${item.region === selectedRegion ? 'active' : ''}" data-region="${item.region}">${item.region}</button>`).join(''); document.querySelectorAll('.region-tab').forEach(button => button.addEventListener('click', () => { selectedRegion = button.dataset.region; renderRegional(); loadDashboard(selectedRegion).catch(error => console.error(error)); })); document.getElementById('region-summary').innerHTML = `<div class="region-metric"><span>Region SKUs</span><strong>${region.sku_count.toLocaleString()}</strong></div><div class="region-metric alert"><span>Stockout risk</span><strong>${region.stockout_count.toLocaleString()}</strong></div><div class="region-metric warn"><span>Overstock risk</span><strong>${region.overstock_count.toLocaleString()}</strong></div><div class="region-metric"><span>Planning window</span><strong>${region.days_until_occasion} days</strong></div>`; document.getElementById('regional-caption').textContent = `Forecast: LightGBM (direct for ${data?.tracked_products ?? '–'} tracked, analog for the rest) · Stock: simulated`; document.getElementById('occasion').textContent = `Upcoming occasion: ${region.occasion} on ${region.occasion_date}. Demand focus: ${region.demand_focus} (${region.focus_categories.join(', ')}).`; document.getElementById('not-forecast').innerHTML = region.not_forecast_count == null ? '' : `${region.not_forecast_count.toLocaleString()} products not forecast (free samples / no price / packaging) · <a href="${region.not_forecast_download}" style="color:var(--green)">download list</a>`; document.getElementById('regional-stockout').innerHTML = region.stockout.map(row => `<tr><td>${escapeHtml(row.product_name)}</td><td class="risk">${money(row.days_of_cover)} days</td><td>${row.recommended_restock_units} units</td><td>${escapeHtml(row.forecast_source)}</td></tr>`).join(''); document.getElementById('regional-overstock').innerHTML = region.overstock.map(row => `<tr><td>${escapeHtml(row.product_name)}</td><td class="risk">${money(row.days_of_cover)} days</td><td class="safe">Reduce orders</td><td>${escapeHtml(row.forecast_source)}</td></tr>`).join(''); document.getElementById('regional-demand').innerHTML = region.demand.map(row => `<tr><td>${escapeHtml(row.product_name)}</td><td>${row.forecast_units} units</td><td>${escapeHtml(row.category)}</td><td>${escapeHtml(row.forecast_source)}</td></tr>`).join(''); }
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
  if (statusEl) statusEl.textContent = data.latest ? `UPDATED ${data.latest}` : 'NO RISK DATA';
  const festival = data.next_festival;
  const festivalLabel = festival ? `${escapeHtml(festival.name)} · ${festival.days_until}d` : 'None scheduled';
  const riskCaption = `Model forecast · ${data.tracked_products ?? '–'} tracked products`;
  const segmentNames = {perishable: 'Perishable', nonperishable: 'Non-perishable'};
  const segments = (data.metrics && data.metrics.segments) || {};
  const wapeLabel = Object.entries(segments).map(([segment, metric]) =>
    `${escapeHtml(segmentNames[segment] || segment)} ${metric.final_test.overall['WAPE_%']}%`
  ).join('<br>') || 'n/a';
  document.getElementById('summary').innerHTML =`<div class="metric"><span>Stockout risk today</span><strong class="red">${data.stockout.length} products</strong><small class="metric-note">${riskCaption}</small></div><div class="metric"><span>Overstock risk today</span><strong class="amber">${data.overstock.length} products</strong><small class="metric-note">${riskCaption}</small></div><div class="metric"><span>Next festival</span><strong>${festivalLabel}</strong></div><div class="metric"><span>Forecast WAPE</span><strong>${wapeLabel}</strong></div>`;
  renderModelRisk();
}
function renderModelRisk() {
  const cover = row => `${money(row.days_of_cover)} days`;
  const emptyRow = columns => `<tr><td colspan="${columns}" class="empty">None today</td></tr>`;
  const stockout = [...data.stockout].sort((a, b) => a.days_of_cover - b.days_of_cover).slice(0, 15);
  const overstock = [...data.overstock].sort((a, b) => b.days_of_cover - a.days_of_cover).slice(0, 15);
  document.getElementById('model-risk-scope').textContent = data.latest ? `${data.city === 'all' ? 'All cities' : selectedRegion} · ${data.latest}` : '';
  document.getElementById('model-stockout').innerHTML = stockout.map(row => `<tr><td>${escapeHtml(row.product_name || row.sku_id)}</td><td>${escapeHtml(row.brand)}</td><td class="risk">${cover(row)}</td><td>${escapeHtml(row.days_until_delivery)} days</td><td>${escapeHtml(row.reorder_qty)} units</td></tr>`).join('') || emptyRow(5);
  document.getElementById('model-overstock').innerHTML = overstock.map(row => `<tr><td>${escapeHtml(row.product_name || row.sku_id)}</td><td>${escapeHtml(row.brand)}</td><td class="risk">${cover(row)}</td><td>${escapeHtml(row.shelf_life_days)} days</td></tr>`).join('') || emptyRow(4);
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
            city = parse_qs(urlparse(self.path).query).get("city", [""])[0].strip().casefold()
            stockout = read_csv("stockout_risk_report.csv")
            overstock = read_csv("overstock_risk_report.csv")
            # Reports cover every day of the test window; "today" is the latest date.
            latest = max((row["date"] for row in stockout + overstock), default=None)
            stockout = [row for row in stockout if row["date"] == latest]
            overstock = [row for row in overstock if row["date"] == latest]
            if city and city != "all":
                stockout = [row for row in stockout if row["city"].casefold() == city]
                overstock = [row for row in overstock if row["city"].casefold() == city]
            festival = next_festival(read_csv_from_path(DATA_DIR / "festival_calendar_5yr.csv"))
            if festival:
                festival = {
                    "name": festival["next_festival"],
                    "date": festival["date"],
                    "days_until": (date.fromisoformat(festival["date"]) - date.today()).days,
                }
            payload = {
                "metrics": read_json("metrics.json"), "stockout": stockout, "overstock": overstock,
                "latest": latest, "next_festival": festival, "city": city or "all", "tracked_products": tracked_product_count(),
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()