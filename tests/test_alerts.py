"""
Email-alert checks on a small fake risk report. Every test points alerts.py
at tmp_path (report, vendors, .env, outputs/alerts) and mocks smtplib.SMTP,
so nothing touches real outputs, the real .env, or the network.
"""

import csv
import json
import smtplib
import ssl
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import pandas as pd
import pytest

import alerts

REPORT_COLS = ["date", "city", "sku_id", "product_name", "brand", "category", "predicted_units", "opening_stock",
               "days_of_cover", "horizon_demand", "avg_daily_pred", "days_until_delivery", "shelf_life_days",
               "reorder_qty"]
SECRET = "app-password-must-not-leak"


def report_rows(n, city, date, category, cover):
    return [{
        "date": date, "city": city, "sku_id": f"SKU{city[:2]}{category[:2]}{i:03d}", "product_name": f"Product {i}",
        "brand": "Brand", "category": category, "predicted_units": 10.0, "opening_stock": 20 + i,
        "days_of_cover": cover + i * 0.01, "horizon_demand": 70.0, "avg_daily_pred": 10.0,
        "days_until_delivery": 5, "shelf_life_days": 3 if category == "dairy" else 180, "reorder_qty": 70,
    } for i in range(n)]


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """30 Bengaluru stockouts + 20 overstocks on the latest day; Mumbai has risks only on an older day."""
    latest, older = "2026-08-25", "2026-08-24"
    stockout = report_rows(28, "Bengaluru", latest, "snacks", 1.0) + report_rows(2, "Bengaluru", latest, "dairy", 0.5)
    stockout += report_rows(1, "Bengaluru", older, "snacks", 2.0) + report_rows(1, "Mumbai", older, "snacks", 2.0)
    overstock = report_rows(20, "Bengaluru", latest, "dairy", 30.0)
    pd.DataFrame(stockout, columns=REPORT_COLS).to_csv(tmp_path / alerts.REPORT_FILES["stockout"], index=False)
    pd.DataFrame(overstock, columns=REPORT_COLS).to_csv(tmp_path / alerts.REPORT_FILES["overstock"], index=False)
    vendors = {
        "Bengaluru": {"vendor_name": "Blr vendor", "vendor_email": ["blr@example.com", "blr2@example.com"],
                      "category_overrides": {}},
        "Mumbai": {"vendor_name": "Mum vendor", "vendor_email": "mum@example.com", "category_overrides": {}},
    }
    (tmp_path / "vendors.json").write_text(json.dumps(vendors))
    (tmp_path / ".env").write_text(
        "# test settings\nSMTP_HOST=smtp.test\nSMTP_PORT=2525\nSMTP_USER=alerts@example.com\n"
        f"SMTP_APP_PASSWORD='{SECRET}'\nALERT_FROM_NAME=Test Alerts\n")
    for key in alerts.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(alerts, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(alerts, "ALERTS_DIR", tmp_path / "alerts")
    monkeypatch.setattr(alerts, "VENDORS_PATH", tmp_path / "vendors.json")
    monkeypatch.setattr(alerts, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(alerts, "load_readiness", lambda city: None)  # no festival section unless a test adds one
    return tmp_path


def test_build_alert_counts_subject_and_caps(fake_project):
    alert = alerts.build_alert("bengaluru")
    assert alert["date"] == "2026-08-25"
    assert alert["counts"] == {"stockout": 30, "overstock": 20}
    assert alert["subject"] == "[Stockout 30 | Overstock 20] Bengaluru inventory alert – 2026-08-25"
    # caps: 25 stockout rows + 15 overstock rows shown, the rest summarised
    assert alert["html"].count("<tr>") == (1 + 25) + (1 + 15)
    assert "+5 more" in alert["html"] and "+5 more" in alert["text"]
    # stockouts sorted by days of cover, lowest first: the two dairy rows lead
    stockout_html = alert["html"].split("STOCKOUT")[1].split("OVERSTOCK")[0]
    assert stockout_html.index("SKUBeda000") < stockout_html.index("SKUBeda001") < stockout_html.index("SKUBesn000")
    # the attachment has every row, not just the capped ones
    attachment = list(csv.DictReader(alert["csv"].splitlines()))
    assert len(attachment) == 50
    assert {row["risk_status"] for row in attachment} == {"stockout", "overstock"}
    assert "auto-generated" in alert["html"] and "auto-generated" in alert["text"]


def test_build_alert_for_an_older_date_and_bad_dates(fake_project):
    assert alerts.build_alert("Bengaluru", "2026-08-24")["counts"] == {"stockout": 1, "overstock": 0}
    with pytest.raises(alerts.AlertError, match="outside the risk report"):
        alerts.build_alert("Bengaluru", "2027-01-01")
    with pytest.raises(alerts.AlertError, match="YYYY-MM-DD"):
        alerts.build_alert("Bengaluru", "25/08/2026")


def test_dry_run_writes_files_and_skips_cities_without_risks(fake_project):
    with mock.patch.object(alerts.smtplib, "SMTP") as smtp:
        [result] = alerts.send_alert("Bengaluru")
        [skipped] = alerts.send_alert("Mumbai")
        [forced] = alerts.send_alert("Mumbai", always=True)
    smtp.assert_not_called()
    assert result["status"] == "dry_run"
    assert (fake_project / "alerts" / "2026-08-25_bengaluru.html").exists()
    assert (fake_project / "alerts" / "2026-08-25_bengaluru.csv").exists()
    assert skipped["status"] == "skipped"
    assert forced["status"] == "dry_run" and forced["counts"] == {"stockout": 0, "overstock": 0}
    assert not (fake_project / "alerts" / "sent_log.csv").exists()  # only real sends are logged


def test_send_alert_uses_starttls_and_logs(fake_project, capsys):
    with mock.patch.object(alerts.smtplib, "SMTP") as smtp:
        [result] = alerts.send_alert("Bengaluru", send=True)
    assert result["status"] == "sent"
    smtp.assert_called_once_with("smtp.test", 2525, timeout=30)
    server = smtp.return_value.__enter__.return_value
    server.starttls.assert_called_once()
    context = server.starttls.call_args.kwargs["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    server.login.assert_called_once_with("alerts@example.com", SECRET)
    message = server.send_message.call_args.args[0]
    assert message["Subject"].startswith("[Stockout 30 | Overstock 20] Bengaluru")
    assert "blr@example.com" in message["To"] and "blr2@example.com" in message["To"]
    assert "Test Alerts" in message["From"]
    assert [part.get_filename() for part in message.iter_attachments()] == ["2026-08-25_bengaluru_risk.csv"]
    assert message.get_body(("html",)) is not None and message.get_body(("plain",)) is not None

    [row] = list(csv.DictReader((fake_project / "alerts" / "sent_log.csv").open()))
    assert row["city"] == "Bengaluru" and row["status"] == "sent"
    assert row["recipients"] == "blr@example.com;blr2@example.com"
    assert (row["stockout_count"], row["overstock_count"]) == ("30", "20")
    assert SECRET not in capsys.readouterr().out + (fake_project / "alerts" / "sent_log.csv").read_text()


def test_smtp_failure_is_reported_and_logged_without_the_password(fake_project, capsys):
    with mock.patch.object(alerts.smtplib, "SMTP") as smtp:
        smtp.return_value.__enter__.return_value.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad")
        exit_code = alerts.main(["--city", "Bengaluru", "--send"])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAILED" in output and "App Password" in output
    log = (fake_project / "alerts" / "sent_log.csv").read_text()
    assert ",failed," in log
    assert SECRET not in output + log


def test_category_overrides_split_the_alert(fake_project):
    vendors = json.loads((fake_project / "vendors.json").read_text())
    vendors["Bengaluru"]["category_overrides"] = {"dairy": {"vendor_name": "Dairy co", "vendor_email": "dairy@example.com"}}
    (fake_project / "vendors.json").write_text(json.dumps(vendors))
    dairy, rest = alerts.send_alert("Bengaluru")
    assert (dairy["vendor_name"], dairy["recipients"], dairy["counts"]) == ("Dairy co", ["dairy@example.com"],
                                                                           {"stockout": 2, "overstock": 20})
    assert (rest["vendor_name"], rest["counts"]) == ("Blr vendor", {"stockout": 28, "overstock": 0})
    assert dairy["html_path"].endswith("2026-08-25_bengaluru_dairy-co.html")


def test_tls_context_verifies_with_or_without_certifi(monkeypatch):
    import certifi

    with mock.patch.object(alerts.ssl, "create_default_context", wraps=ssl.create_default_context) as create:
        context = alerts.tls_context()
    create.assert_called_once_with(cafile=certifi.where())
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname

    monkeypatch.setitem(sys.modules, "certifi", None)  # import certifi -> ImportError
    with mock.patch.object(alerts.ssl, "create_default_context", wraps=ssl.create_default_context) as create:
        context = alerts.tls_context()
    create.assert_called_once_with()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_load_env_requires_smtp_settings(fake_project):
    (fake_project / ".env").write_text("SMTP_HOST=smtp.test\n")
    with pytest.raises(alerts.AlertError, match="SMTP_USER, SMTP_APP_PASSWORD") as error:
        alerts.load_env()
    assert SECRET not in str(error.value)


def test_dashboard_send_alert_endpoint(fake_project):
    import dashboard

    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(query, headers):
        request = urllib.request.Request(f"{base}/api/send-alert?{query}", method="POST", headers=headers)
        return urllib.request.urlopen(request, timeout=10)

    try:
        with pytest.raises(urllib.error.HTTPError) as error:  # no X-Alert-Request header
            post("city=Bengaluru&send=0", {})
        assert error.value.code == 403
        with mock.patch.object(alerts.smtplib, "SMTP") as smtp:
            payload = json.loads(post("city=Bengaluru&send=0", {"X-Alert-Request": "1"}).read())
        smtp.assert_not_called()
        [result] = payload["results"]
        assert result["status"] == "dry_run" and result["preview_url"] == "/alerts/2026-08-25_bengaluru.html"
        assert "html_path" not in result
        preview = urllib.request.urlopen(base + result["preview_url"], timeout=10).read().decode()
        assert "Bengaluru inventory alert" in preview
        with pytest.raises(urllib.error.HTTPError) as error:
            post("city=Pune&send=0", {"X-Alert-Request": "1"})
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
