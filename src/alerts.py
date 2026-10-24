"""
alerts.py
-----------
Emails each city's vendor the day's stockout / overstock rows from the model
risk report (outputs/stockout_risk_report.csv and overstock_risk_report.csv,
written by train_model.py). Nothing here changes the forecast or the risk
rule; it only formats rows that inventory.classify_risk already labelled.

Vendors are in config/vendors.json: one entry per city with vendor_name,
vendor_email (a string or a list) and optional category_overrides, which
send the rows of those model categories to a different vendor in a separate
email. SMTP settings come from .env (see .env.example).

Dry run is the default: the HTML and CSV go to outputs/alerts/ and nothing
is sent. --send emails over SMTP with STARTTLS and logs every attempt to
outputs/alerts/sent_log.csv.

Usage:
    python -m src.alerts --city Bengaluru|all [--date YYYY-MM-DD] [--send] [--always]
"""

import argparse
import csv
import html
import json
import os
import re
import smtplib
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, PROJECT_DIR  # noqa: E402

REPORT_DIR = OUTPUTS_DIR
ALERTS_DIR = OUTPUTS_DIR / "alerts"
VENDORS_PATH = PROJECT_DIR / "config" / "vendors.json"
ENV_PATH = PROJECT_DIR / ".env"
SENT_LOG_COLS = ["timestamp", "city", "vendor_name", "recipients", "stockout_count", "overstock_count",
                 "status", "detail"]

STOCKOUT_CAP = 25
OVERSTOCK_CAP = 15
ENV_KEYS = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_APP_PASSWORD", "ALERT_FROM_NAME"]
ENV_DEFAULTS = {"SMTP_HOST": "smtp.gmail.com", "SMTP_PORT": "587", "ALERT_FROM_NAME": "Grocery Forecast Alerts"}
# Columns of the risk report attached as CSV, plus risk_status
ATTACHMENT_COLS = ["date", "city", "risk_status", "sku_id", "product_name", "brand", "category", "opening_stock",
                   "horizon_demand", "avg_daily_pred", "days_of_cover", "days_until_delivery", "shelf_life_days",
                   "reorder_qty"]


class AlertError(Exception):
    """A problem the user can fix (bad city or date, missing settings, SMTP failure)."""


def load_env(path=None):
    """
    SMTP settings from .env, with real environment variables taking priority
    (as python-dotenv does). Returns a dict of ENV_KEYS; never prints values.
    """
    path = Path(path or ENV_PATH)
    values = {}
    if path.exists():
        try:
            from dotenv import dotenv_values
            values = {key: value for key, value in dotenv_values(path).items() if value is not None}
        except ImportError:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.removeprefix("export ").split("=", 1)
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key.strip()] = value
    env = {key: os.environ.get(key, values.get(key, ENV_DEFAULTS.get(key, ""))) for key in ENV_KEYS}
    missing = [key for key in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_APP_PASSWORD") if not env[key]]
    if missing:
        raise AlertError(f"missing SMTP settings in {path.name}: {', '.join(missing)} (see .env.example)")
    return env


def load_vendors():
    return json.loads(VENDORS_PATH.read_text(encoding="utf-8"))


def as_list(emails):
    return [emails] if isinstance(emails, str) else list(emails)


def vendor_routes(city):
    """
    [(vendor_name, emails, categories, exclude)] for a city: one route per
    override vendor (only its categories), then the city vendor for the rest.
    """
    vendors = {name.casefold(): (name, entry) for name, entry in load_vendors().items()}
    if city.casefold() not in vendors:
        raise AlertError(f"no vendor for city {city!r} in {VENDORS_PATH.name}; known: {', '.join(load_vendors())}")
    _, entry = vendors[city.casefold()]
    grouped = {}
    for category, override in entry.get("category_overrides", {}).items():
        key = (override["vendor_name"], tuple(as_list(override["vendor_email"])))
        grouped.setdefault(key, set()).add(category)
    routes = [(name, list(emails), categories, set()) for (name, emails), categories in grouped.items()]
    overridden = set().union(*grouped.values()) if grouped else set()
    routes.append((entry["vendor_name"], as_list(entry["vendor_email"]), None, overridden))
    return routes


def load_report():
    """Stockout and overstock rows of the model risk report, with risk_status."""
    frames = []
    for status in ("stockout", "overstock"):
        path = REPORT_DIR / f"{status}_risk_report.csv"
        if not path.exists():
            raise AlertError(f"{path.name} not found; run src/train_model.py (or run_pipeline.py) first")
        frames.append(pd.read_csv(path).assign(risk_status=status))
    return pd.concat(frames, ignore_index=True)


def canonical_city(city):
    return next((name for name in load_vendors() if name.casefold() == city.casefold()), city)


def build_alert(city, date=None, categories=None, exclude=()):
    """
    The alert for one city and day (default: the report's latest day, as on
    the dashboard): subject, html, text, csv, counts. categories / exclude
    limit it to some model categories (for category_overrides).
    """
    report = load_report()
    first, last = report["date"].min(), report["date"].max()
    date = date or last
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise AlertError(f"date {date!r} is not YYYY-MM-DD")
    if not first <= date <= last:
        raise AlertError(f"date {date} is outside the risk report ({first} to {last})")
    city = canonical_city(city)
    rows = report[(report["date"] == date) & (report["city"].str.casefold() == city.casefold())]
    if categories is not None:
        rows = rows[rows["category"].isin(categories)]
    rows = rows[~rows["category"].isin(exclude)]
    stockout = rows[rows["risk_status"] == "stockout"].sort_values("days_of_cover")
    overstock = rows[rows["risk_status"] == "overstock"].sort_values("days_of_cover", ascending=False)
    counts = {"stockout": len(stockout), "overstock": len(overstock)}
    subject = f"[Stockout {counts['stockout']} | Overstock {counts['overstock']}] {city} inventory alert – {date}"
    attachment = pd.concat([stockout, overstock])[ATTACHMENT_COLS]
    return {
        "city": city, "date": date, "counts": counts, "subject": subject,
        "html": render_html(city, date, stockout, overstock),
        "text": render_text(city, date, stockout, overstock),
        "csv": attachment.to_csv(index=False),
        "csv_name": f"{date}_{slug(city)}_risk.csv",
    }


def product(row):
    name = row["product_name"] if isinstance(row["product_name"], str) and row["product_name"] else row["sku_id"]
    return f"{name} ({row['sku_id']})"


def stockout_cells(row):
    return [product(row), row["category"], f"{row['opening_stock']:.0f}", f"{row['horizon_demand']:.1f}",
            f"{row['days_of_cover']:.1f}", f"{row['days_until_delivery']:.0f}", f"{row['reorder_qty']:.0f}"]


def overstock_cells(row):
    return [product(row), f"{row['opening_stock']:.0f}", f"{row['days_of_cover']:.1f}", f"{row['shelf_life_days']:.0f}"]


STOCKOUT_HEADERS = ["Product", "Category", "Stock", "7-day forecast", "Days of cover", "Days to next delivery",
                    "Reorder qty"]
OVERSTOCK_HEADERS = ["Product", "Stock", "Days of cover", "Shelf life (days)"]
FOOTER = ("This alert is auto-generated by the grocery demand-forecasting pipeline from the model risk report. "
          "Please do not reply to this email.")

CELL = "padding:6px 8px;border:1px solid #dce8dc;font-size:13px;"
HEAD = CELL + "background:#f0f6f0;color:#173b28;text-align:left;"


def html_table(title, color, headers, rows, cell_fn, cap):
    if rows.empty:
        return f'<h3 style="color:{color};font-size:15px;margin:20px 0 6px;">{title}</h3><p style="font-size:13px;">None.</p>'
    head = "".join(f'<th style="{HEAD}">{html.escape(h)}</th>' for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td style="{CELL}">{html.escape(str(value))}</td>' for value in cell_fn(row)) + "</tr>"
        for _, row in rows.head(cap).iterrows())
    more = (f'<p style="font-size:13px;color:#66806f;">+{len(rows) - cap} more in the attached CSV.</p>'
            if len(rows) > cap else "")
    return (f'<h3 style="color:{color};font-size:15px;margin:20px 0 6px;">{title} ({len(rows)})</h3>'
            f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%;">'
            f"<tr>{head}</tr>{body}</table>{more}")


def render_html(city, date, stockout, overstock):
    city_e, date_e = html.escape(city), html.escape(date)
    reorder = int(stockout["reorder_qty"].sum())
    return (
        '<!doctype html><html><body style="margin:0;padding:16px;background:#ffffff;">'
        '<div style="max-width:760px;font-family:Arial,Helvetica,sans-serif;color:#173b28;">'
        f'<h2 style="font-size:19px;margin:0 0 4px;">{city_e} inventory alert – {date_e}</h2>'
        '<p style="font-size:14px;margin:8px 0;">'
        f'<b style="color:#d94b4b;">{len(stockout)} stockout risk</b> · '
        f'<b style="color:#d38b00;">{len(overstock)} overstock risk</b> · '
        f'{reorder} units to reorder</p>'
        '<p style="font-size:13px;color:#66806f;margin:0;">Stockout: stock will not last until the next delivery. '
        'Overstock: more stock than can sell before it expires. The full list is attached as CSV.</p>'
        + html_table("STOCKOUT", "#d94b4b", STOCKOUT_HEADERS, stockout, stockout_cells, STOCKOUT_CAP)
        + html_table("OVERSTOCK", "#d38b00", OVERSTOCK_HEADERS, overstock, overstock_cells, OVERSTOCK_CAP)
        + f'<p style="font-size:12px;color:#888888;margin-top:24px;border-top:1px solid #dce8dc;padding-top:10px;">'
        f"{html.escape(FOOTER)}</p></div></body></html>")


def text_table(title, headers, rows, cell_fn, cap):
    lines = [f"{title} ({len(rows)})"]
    if rows.empty:
        return lines + ["  None.", ""]
    lines.append("  " + " | ".join(headers))
    lines += ["  " + " | ".join(cell_fn(row)) for _, row in rows.head(cap).iterrows()]
    if len(rows) > cap:
        lines.append(f"  +{len(rows) - cap} more in the attached CSV.")
    return lines + [""]


def render_text(city, date, stockout, overstock):
    lines = [f"{city} inventory alert – {date}", "",
             f"Stockout risk: {len(stockout)}   Overstock risk: {len(overstock)}   "
             f"Units to reorder: {int(stockout['reorder_qty'].sum())}", ""]
    lines += text_table("STOCKOUT", STOCKOUT_HEADERS, stockout, stockout_cells, STOCKOUT_CAP)
    lines += text_table("OVERSTOCK", OVERSTOCK_HEADERS, overstock, overstock_cells, OVERSTOCK_CAP)
    return "\n".join(lines + ["--", FOOTER])


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def make_message(alert, vendor_name, recipients, env):
    message = EmailMessage()
    message["Subject"] = alert["subject"]
    message["From"] = formataddr((env["ALERT_FROM_NAME"], env["SMTP_USER"]))
    message["To"] = ", ".join(formataddr((vendor_name, email)) for email in recipients)
    message.set_content(alert["text"])
    message.add_alternative(alert["html"], subtype="html")
    message.add_attachment(alert["csv"].encode("utf-8"), maintype="text", subtype="csv", filename=alert["csv_name"])
    return message


def tls_context():
    """
    Verifying TLS context. Uses certifi's CA bundle when installed: python.org
    builds on macOS ship without system CA certificates, which otherwise
    fails with CERTIFICATE_VERIFY_FAILED.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def smtp_send(message, env):
    """Sends over STARTTLS; raises AlertError with a readable reason (never the password)."""
    try:
        with smtplib.SMTP(env["SMTP_HOST"], int(env["SMTP_PORT"]), timeout=30) as server:
            server.starttls(context=tls_context())
            server.login(env["SMTP_USER"], env["SMTP_APP_PASSWORD"])
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as error:
        raise AlertError(f"SMTP login failed for {env['SMTP_USER']} ({error.smtp_code}); check SMTP_USER and "
                         "SMTP_APP_PASSWORD (for Gmail, an App Password, not the account password)") from None
    except smtplib.SMTPRecipientsRefused as error:
        raise AlertError(f"recipients refused: {', '.join(error.recipients)}") from None
    except smtplib.SMTPException as error:
        raise AlertError(f"SMTP error: {type(error).__name__}: {error}") from None
    except (OSError, ValueError) as error:
        raise AlertError(f"could not connect to {env['SMTP_HOST']}:{env['SMTP_PORT']}: {error}") from None


def log_send(city, vendor_name, recipients, counts, status, detail=""):
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERTS_DIR / "sent_log.csv"
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(SENT_LOG_COLS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), city, vendor_name, ";".join(recipients),
                         counts["stockout"], counts["overstock"], status, detail])


def send_alert(city, send=False, date=None, always=False):
    """
    Builds the alert for each of the city's vendors. Dry run (send=False)
    writes outputs/alerts/<date>_<city>.html/.csv; send=True emails it.
    Vendors with no risk rows are skipped unless always. Returns one result
    dict per vendor with status "dry_run", "sent", "failed" or "skipped".
    """
    results = []
    env = load_env() if send else None
    for vendor_name, recipients, categories, exclude in vendor_routes(city):
        alert = build_alert(city, date, categories, exclude)
        result = {"city": alert["city"], "date": alert["date"], "vendor_name": vendor_name, "recipients": recipients,
                  "counts": alert["counts"], "subject": alert["subject"]}
        if not any(alert["counts"].values()) and not always:
            results.append({**result, "status": "skipped", "detail": "no stockout or overstock rows"})
            continue
        if not send:
            ALERTS_DIR.mkdir(parents=True, exist_ok=True)
            stem = f"{alert['date']}_{slug(alert['city'])}" + (f"_{slug(vendor_name)}" if categories else "")
            html_path, csv_path = ALERTS_DIR / f"{stem}.html", ALERTS_DIR / f"{stem}.csv"
            html_path.write_text(alert["html"], encoding="utf-8")
            csv_path.write_text(alert["csv"], encoding="utf-8")
            results.append({**result, "status": "dry_run", "html_path": str(html_path), "csv_path": str(csv_path)})
            continue
        try:
            smtp_send(make_message(alert, vendor_name, recipients, env), env)
        except AlertError as error:
            log_send(alert["city"], vendor_name, recipients, alert["counts"], "failed", str(error))
            results.append({**result, "status": "failed", "detail": str(error)})
            continue
        log_send(alert["city"], vendor_name, recipients, alert["counts"], "sent")
        results.append({**result, "status": "sent"})
    return results


def describe(result):
    counts = f"stockout {result['counts']['stockout']}, overstock {result['counts']['overstock']}"
    to = ", ".join(result["recipients"])
    if result["status"] == "dry_run":
        return f"[dry run] {result['city']} ({counts}) -> {result['html_path']} (not sent; would go to {to})"
    if result["status"] == "sent":
        return f"[sent] {result['city']} ({counts}) to {to}"
    if result["status"] == "skipped":
        return f"[skipped] {result['city']} -> {result['vendor_name']}: {result['detail']} (use --always to send anyway)"
    return f"[FAILED] {result['city']} to {to}: {result['detail']}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Email each city's vendor its stockout / overstock alert.")
    parser.add_argument("--city", required=True, help="city name from config/vendors.json, or 'all'")
    parser.add_argument("--date", help="report date YYYY-MM-DD (default: latest)")
    parser.add_argument("--send", action="store_true", help="really send email (default: dry run to outputs/alerts/)")
    parser.add_argument("--always", action="store_true", help="also alert vendors with no risk rows")
    args = parser.parse_args(argv)

    try:
        cities = list(load_vendors()) if args.city.casefold() == "all" else [args.city]
        results = [result for city in cities for result in send_alert(city, args.send, args.date, args.always)]
    except AlertError as error:
        print(f"Error: {error}")
        return 1
    for result in results:
        print(describe(result))
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
