from pathlib import Path
import os

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Core path definitions
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Data asset paths
ARCHIVE_PATH = Path(os.getenv("ARCHIVE_PATH", DATA_DIR / "archive.zip"))
ARCHIVE_CSV = DATA_DIR / "archive_products.csv"

# Pipeline constants
LEAD_TIME_DAYS = 7  # Forecast horizon used for leak-free lags
RANDOM_SEED = 42

# Model inputs (built by build_features.py). Every lag/rolling feature is
# shifted by at least LEAD_TIME_DAYS so the model never sees future sales.
FEATURE_COLS = [
    "unit_price", "on_promotion", "days_until_next_festival", "days_since_last_festival",
    "next_festival_weight", "is_festival_week", "day_of_week", "is_weekend", "month", "week_of_year",
    "is_month_start", "is_month_end", "lag_7", "lag_14", "lag_21", "lag_364",
    "roll_mean_7_lag7", "roll_mean_14_lag7", "roll_mean_28_lag7", "roll_std_14_lag7",
    "log_mrp", "pack_size_value",
    "city_code", "category_code", "sku_id_code", "brand_code",
]
CATEGORICAL_FEATURES = ["city_code", "category_code", "sku_id_code", "brand_code"]

# Days a category stays sellable; overstock means more cover than can be sold
# before it spoils (see train_model.build_risk_report).
# Synthetic replenishment: each SKU is delivered every
# min(DELIVERY_CYCLE_DAYS, shelf life) days, ordered up to demand for that
# cycle plus SAFETY_STOCK_DAYS.
DELIVERY_CYCLE_DAYS = 7
SAFETY_STOCK_DAYS = 2

# Synthetic demand ~ (MRP / category median) ** -PRICE_ELASTICITY; the
# regional analog forecasts use the same value to adjust for price.
PRICE_ELASTICITY = 0.7

# "vegetables" and "fruits" are frozen vegetables and dry fruits in the
# sampled catalog (BigBasket has no fresh produce), so they keep for months.
SHELF_LIFE_DAYS = {
    "dairy": 3,
    "sweets": 4,
    "vegetables": 180,
    "fruits": 120,
    "snacks": 180,
    "beverages": 270,
    "household": 365,
    "personal_care": 730,
}

# Ensure core directories exist
for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    d.mkdir(exist_ok=True)
