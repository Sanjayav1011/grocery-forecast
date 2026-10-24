"""
sample_products.py
--------------------
Picks the REAL BigBasket products the synthetic POS history is generated
for: a fixed-seed sample stratified across the 7 model categories.

Inputs : data/archive_products.csv
Outputs: data/category_map.csv      (BigBasket category -> model category rules)
         data/sampled_products.csv  (the product master used by the generator
                                     and build_features.py)

sku_id is the same BB###### id build_bigbasket_regional_risk.py assigns
(row order in archive_products.csv), so products line up across datasets.

The archive has no fresh fruit or vegetables, so the closest real products
stand in: frozen vegetables for "vegetables" and dry fruits for "fruits".

Two kinds of mapping rules: CATEGORY_MAP decides which products can be
SAMPLED (tracked with POS history); FORECAST_ONLY_MAP adds more catalog
sub-categories that fit a model category, so forecast_regional.py can cover
them with analog forecasts without changing the tracked set.
"""

import re

import numpy as np
import pandas as pd

from config import ARCHIVE_CSV, DATA_DIR, RANDOM_SEED

CATEGORY_MAP_PATH = DATA_DIR / "category_map.csv"
SAMPLED_PRODUCTS_PATH = DATA_DIR / "sampled_products.csv"
PRODUCTS_PER_CATEGORY = 43

# (BigBasket category, sub-category, sub-sub-category or "*" for any) -> model category
CATEGORY_MAP = [
    ("bakery, cakes & dairy", "dairy", "*", "dairy"),
    ("gourmet & world food", "dairy & cheese", "*", "dairy"),
    ("snacks & branded foods", "frozen veggies & snacks", "frozen vegetables", "vegetables"),
    ("foodgrains, oil & masala", "dry fruits", "*", "fruits"),
    ("gourmet & world food", "snacks, dry fruits, nuts", "dry fruits & berries", "fruits"),
    ("snacks & branded foods", "indian mithai", "fresh sweets", "sweets"),
    ("beverages", "tea", "*", "beverages"),
    ("beverages", "coffee", "*", "beverages"),
    ("beverages", "energy & soft drinks", "*", "beverages"),
    ("beverages", "fruit juices & drinks", "*", "beverages"),
    ("beverages", "water", "*", "beverages"),
    ("snacks & branded foods", "snacks & namkeen", "*", "snacks"),
    ("snacks & branded foods", "biscuits & cookies", "*", "snacks"),
    ("cleaning & household", "detergents & dishwash", "*", "household"),
    ("cleaning & household", "all purpose cleaners", "*", "household"),
    ("cleaning & household", "disposables, garbage bag", "*", "household"),
    ("cleaning & household", "fresheners & repellents", "*", "household"),
    ("beauty & hygiene", "*", "*", "personal_care"),
]

# Forecast-only: no products are sampled from these. "snacks" doubles as
# packaged foods and "vegetables" as frozen foods; packaged (long-life)
# Indian sweets go to snacks, fresh bakery to the short-shelf-life sweets.
FORECAST_ONLY_MAP = [
    ("snacks & branded foods", "breakfast cereals", "*", "snacks"),
    ("snacks & branded foods", "chocolates & candies", "*", "snacks"),
    ("snacks & branded foods", "noodle, pasta, vermicelli", "*", "snacks"),
    ("snacks & branded foods", "pickles & chutney", "*", "snacks"),
    ("snacks & branded foods", "ready to cook & eat", "*", "snacks"),
    ("snacks & branded foods", "spreads, sauces, ketchup", "*", "snacks"),
    ("snacks & branded foods", "indian mithai", "tinned, packed sweets", "snacks"),
    ("snacks & branded foods", "indian mithai", "chikki & gajjak", "snacks"),
    ("snacks & branded foods", "frozen veggies & snacks", "frozen veg snacks", "vegetables"),
    ("snacks & branded foods", "frozen veggies & snacks", "frozen non-veg snacks", "vegetables"),
    ("snacks & branded foods", "frozen veggies & snacks", "frozen indian breads", "vegetables"),
    ("bakery, cakes & dairy", "bakery snacks", "*", "snacks"),
    ("bakery, cakes & dairy", "cookies, rusk & khari", "*", "snacks"),
    ("bakery, cakes & dairy", "cakes & pastries", "*", "sweets"),
    ("bakery, cakes & dairy", "gourmet breads", "*", "sweets"),
    ("bakery, cakes & dairy", "non dairy", "*", "dairy"),
    ("beverages", "nutrition mix, supplement", "*", "beverages"),
    ("gourmet & world food", "drinks & beverages", "*", "beverages"),
    ("gourmet & world food", "cereals & breakfast", "*", "snacks"),
    ("gourmet & world food", "chocolates & biscuits", "*", "snacks"),
    ("gourmet & world food", "pasta, soup & noodles", "*", "snacks"),
    ("gourmet & world food", "sauces, spreads & dips", "*", "snacks"),
    ("gourmet & world food", "tinned & processed food", "*", "snacks"),
    ("gourmet & world food", "snacks, dry fruits, nuts", "healthy, baked snacks", "snacks"),
    ("gourmet & world food", "snacks, dry fruits, nuts", "nachos & chips", "snacks"),
    ("gourmet & world food", "snacks, dry fruits, nuts", "trail & cocktail mixes", "snacks"),
    ("gourmet & world food", "snacks, dry fruits, nuts", "roasted seeds & nuts", "fruits"),
    ("cleaning & household", "bins & bathroom ware", "*", "household"),
    ("cleaning & household", "car & shoe care", "*", "household"),
    ("cleaning & household", "mops, brushes & scrubs", "*", "household"),
    ("cleaning & household", "party & festive needs", "*", "household"),
    ("baby care", "baby bath & hygiene", "*", "personal_care"),
    ("baby care", "mothers & maternity", "*", "personal_care"),
]

# Categories are sampled in the order they were introduced, so adding a
# category never changes the products already picked for earlier ones.
SAMPLING_ORDER = ["beverages", "dairy", "fruits", "household", "snacks", "sweets", "vegetables", "personal_care"]

# Beauty/personal-care words that never appear in real food names; products
# in FOOD categories whose name contains one are mis-filed and excluded.
# (A bare "cream" would also drop cream biscuits and creamy coffee.)
NON_FOOD_KEYWORDS = [
    "toner", "serum", "shampoo", "conditioner", "soap", "lotion", "moisturiser", "moisturizer",
    "face wash", "cleanser", "sunscreen", "face cream", "skin cream", "body cream", "hair oil",
    "lip balm", "deodorant", "perfume",
]
NON_FOOD_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in NON_FOOD_KEYWORDS) + r")\b", re.IGNORECASE)
NON_FOOD_CATEGORIES = {"household", "personal_care"}  # categories where those words are expected

# "2 x 200 g", "500 ml", "1.5 L", "6 pcs" -> (total amount, unit) in g / ml / pcs
SIZE_PATTERN = re.compile(r"(?:(\d+)\s*[xX]\s*)?(\d+(?:\.\d+)?)\s*(kg|g|gm|l|ltr|ml|pcs|pc)\b", re.IGNORECASE)
UNIT_SCALE = {"kg": ("g", 1000), "g": ("g", 1), "gm": ("g", 1), "l": ("ml", 1000), "ltr": ("ml", 1000),
              "ml": ("ml", 1), "pcs": ("pcs", 1), "pc": ("pcs", 1)}


def parse_pack_size(*texts):
    """Total pack amount and unit from the first text that has one, else (NaN, "")."""
    for text in texts:
        if not isinstance(text, str):
            continue
        match = SIZE_PATTERN.search(text)
        if match:
            count, amount, unit = match.groups()
            base_unit, scale = UNIT_SCALE[unit.lower()]
            return float(amount) * scale * int(count or 1), base_unit
    return np.nan, ""


def parse_mrp(value):
    try:
        return float(re.sub(r"[^0-9.]", "", str(value)))
    except ValueError:
        return np.nan


def load_catalog():
    catalog = pd.read_csv(ARCHIVE_CSV, low_memory=False)
    catalog.insert(0, "sku_id", [f"BB{index:06d}" for index in range(1, len(catalog) + 1)])
    for col in ["Category", "Sub-Category", "Sub-sub-Category"]:
        catalog[col] = catalog[col].astype(str).str.strip().str.lower()
    return catalog


def map_categories(catalog, rules):
    model_category = pd.Series(np.nan, index=catalog.index, dtype=object)
    for category, sub, subsub, target in rules:
        mask = catalog["Category"] == category
        if sub != "*":
            mask &= catalog["Sub-Category"] == sub
        if subsub != "*":
            mask &= catalog["Sub-sub-Category"] == subsub
        model_category[mask] = target
    return model_category


def prepare_catalog():
    """
    Whole catalog with its model category (any rule), sampling_category
    (sampling rules only), parsed MRP, brand popularity, pack size, and an
    `excluded_non_food` flag for mis-filed items.
    """
    catalog = load_catalog()
    catalog["model_category"] = map_categories(catalog, CATEGORY_MAP + FORECAST_ONLY_MAP)
    catalog["sampling_category"] = map_categories(catalog, CATEGORY_MAP)
    catalog["mrp"] = catalog["MRP"].map(parse_mrp)
    catalog["brand"] = catalog["Brand"].fillna("").astype(str).str.strip()
    # brand popularity = how many products the brand has across the WHOLE catalog
    catalog["brand_catalog_count"] = catalog.groupby("brand")["sku_id"].transform("count")
    pack = [parse_pack_size(size, name) for size, name in zip(catalog["SKU Size"], catalog["SKU Name"])]
    catalog["pack_size_value"] = [value for value, _ in pack]
    catalog["pack_size_unit"] = [unit for _, unit in pack]
    in_food_category = catalog["model_category"].notna() & ~catalog["model_category"].isin(NON_FOOD_CATEGORIES)
    catalog["excluded_non_food"] = in_food_category & catalog["SKU Name"].fillna("").str.contains(NON_FOOD_PATTERN)
    return catalog


def sample():
    catalog = prepare_catalog()
    candidates = catalog[catalog["sampling_category"].notna() & catalog["SKU Name"].notna()]
    candidates = candidates.drop_duplicates(["SKU Name", "brand", "SKU Size"])
    valid = (candidates["mrp"] > 0) & (candidates["brand"] != "")

    rng = np.random.default_rng(RANDOM_SEED)
    picks = []
    for category in SAMPLING_ORDER:
        rows = candidates[candidates["sampling_category"] == category]
        # prefer products with a valid MRP and brand; top up from the rest only if short
        # Shuffle BEFORE dropping excluded items, so an exclusion only swaps
        # that product for the next one in the same seeded order.
        preferred, rest = rows[valid.loc[rows.index]], rows[~valid.loc[rows.index]]
        preferred = preferred.loc[rng.permutation(preferred.index)]
        take = preferred[~preferred["excluded_non_food"]].head(PRODUCTS_PER_CATEGORY)
        if len(take) < PRODUCTS_PER_CATEGORY:
            rest = rest.loc[rng.permutation(rest.index)]
            take = pd.concat([take, rest[~rest["excluded_non_food"]].head(PRODUCTS_PER_CATEGORY - len(take))])
        picks.append(take)

    products = pd.concat(picks).rename(columns={
        "SKU Name": "product_name", "SKU Size": "sku_size", "sampling_category": "category",
        "Category": "bb_category", "Sub-Category": "bb_subcategory", "Sub-sub-Category": "bb_subsubcategory",
    })[["sku_id", "product_name", "brand", "mrp", "sku_size", "pack_size_value", "pack_size_unit",
        "brand_catalog_count", "category", "bb_category", "bb_subcategory", "bb_subsubcategory"]]
    products = products.sort_values(["category", "sku_id"]).reset_index(drop=True)

    pd.DataFrame([(*rule, 1) for rule in CATEGORY_MAP] + [(*rule, 0) for rule in FORECAST_ONLY_MAP],
                 columns=["bb_category", "bb_subcategory", "bb_subsubcategory", "model_category", "use_for_sampling"]
                 ).to_csv(CATEGORY_MAP_PATH, index=False)
    products.to_csv(SAMPLED_PRODUCTS_PATH, index=False)
    return products


if __name__ == "__main__":
    products = sample()
    print(f"Saved {len(products)} products to {SAMPLED_PRODUCTS_PATH}")
    print(products.groupby("category").agg(
        products=("sku_id", "count"), valid_mrp=("mrp", lambda s: int((s > 0).sum())),
        brands=("brand", "nunique"), pack_parsed=("pack_size_value", lambda s: int(s.notna().sum())),
        median_mrp=("mrp", "median")).to_string())
