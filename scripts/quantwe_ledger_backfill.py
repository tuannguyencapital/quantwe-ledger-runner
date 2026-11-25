"""
QUANTWE IMMUTABLE LEDGER — 10-YEAR BACKFILL & DAILY EOD ENGINE (NO TOKEN REQUIRED)
- Nguồn giá OHLCV 2016–2026: VPS TradingView Feed (chính) + VNDirect dchart (dự phòng)
- Nguồn Cổ tức & BCTC: API công khai TCBS (không cần token)
- Tương thích cả khóa mới `sb_secret_...` lẫn khóa Legacy JWT
- Chế độ Append-Only tuyệt đối (ignore-duplicates), xuất Cold Backup Parquet + SHA-256
"""
import hashlib
import json
import os
import random
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

try:
    from curl_cffi import requests as http_req
    USE_CFFI = True
except ImportError:
    import requests as http_req
    USE_CFFI = False

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"].strip()
FORCE_FULL = os.environ.get("FORCE_FULL_BACKFILL", "0") == "1"

# Danh mục chuẩn: 3 Chỉ số + Rổ VN100
INDICES = ["VNINDEX", "VN30", "HNXINDEX"]
VN100_TICKERS = [
    # VN30
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    # VN70 Midcap tiêu biểu
    "ANV", "BCG", "BMP", "BSI", "BWE", "CII", "CMG", "CTR", "CTS", "DBC",
    "DCM", "DGC", "DGW", "DIG", "DPM", "DXG", "DXS", "EIB", "EVF", "FRT",
    "FTS", "GEX", "GMD", "HAG", "HCM", "HDC", "HDG", "HHV", "HSG", "HT1",
    "IMP", "KBC", "KDC", "KDH", "KOS", "LPB", "MSB", "NAB", "NKG", "NLG",
    "NT2", "OCB", "ORS", "PAN", "PC1", "PDR", "PHR", "PNJ", "PTB", "PVD",
    "PVT", "REE", "SBT", "SCS", "SIP", "SJS", "SZC", "TCH", "TLG", "VCI",
    "VGC", "VHC", "VIX", "VND", "VPI", "VSC", "VTP", "DSE", "HVN", "BFC",
]
UNIVERSE = INDICES + VN100_TICKERS


def sb_headers(write=False):
    """Tự động xử lý chuẩn khóa mới `sb_secret_...` và Legacy JWT `eyJ...`."""
    h = {"apikey": SUPABASE_SERVICE_KEY}
    if SUPABASE_SERVICE_KEY.startswith("eyJ"):
        h["Authorization"] = f"Bearer {SUPABASE_SERVICE_KEY}"
    if write:
        h["Content-Type"] = "application/json"
        h["Prefer"] = "resolution=ignore-duplicates,return=minimal"
    return h


def http_get_json(url, timeout=20):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    kwargs = {"headers": headers, "timeout": timeout}
    if USE_CFFI:
        kwargs["impersonate"] = "chrome120"
    r = http_req.get(url, **kwargs)
    if r.status_code != 200:
        return None
    return r.json()


def sb_post_batches(table, rows, batch_size=500):
    if not rows:
        return 0
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    hdr = sb_headers(write=True)
    sent = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        r = http_req.post(url, headers=hdr, json=chunk, timeout=30)
        if r.status_code in (200, 201, 204):
            sent += len(chunk)
        else:
            print(f"  [WARN] {table} batch {i}: HTTP {r.status_code} -> {r.text[:180]}")
    return sent


def get_latest_ohlcv_date():
    url = f"{SUPABASE_URL}/rest/v1/ledger_ohlcv_daily?select=trade_date&order=trade_date.desc&limit=1"
    r = http_req.get(url, headers=sb_headers(), timeout=15)
    if r.status_code == 200 and r.json():
        return r.json()[0]["trade_date"]
    return None


# ----------------------------------------------------------------------------
# 1. KÉO LỊCH SỬ OHLCV (VPS CHÍNH -> VNDIRECT DỰ PHÒNG)
# ----------------------------------------------------------------------------
def fetch_ohlcv_symbol(symbol, ts_from, ts_to):
    endpoints = [
        ("VPS", f"https://histdatafeed.vps.com.vn/tradingview/history?symbol={symbol}&resolution=D&from={ts_from}&to={ts_to}"),
        ("VNDIRECT", f"https://dchart-api.vndirect.com.vn/dchart/history?resolution=D&symbol={symbol}&from={ts_from}&to={ts_to}"),
    ]
    for src, url in endpoints:
        try:
            data = http_get_json(url, timeout=20)
            if not data or data.get("s") != "ok" or not data.get("t"):
                continue
            times = data["t"]
            opens, highs, lows, closes = data["o"], data["h"], data["l"], data["c"]
            vols = data.get("v") or [0] * len(times)
            rows = []
            for idx, ts in enumerate(times):
                t_date = datetime.fromtimestamp(int(ts), tz=timezone(timedelta(hours=7))).strftime("%Y-%m-%d")
                c_val = float(closes[idx])
                v_val = int(vols[idx] or 0)
                rows.append({
                    "symbol": symbol,
                    "trade_date": t_date,
                    "open_raw": round(float(opens[idx]), 2),
                    "high_raw": round(float(highs[idx]), 2),
                    "low_raw": round(float(lows[idx]), 2),
                    "close_raw": round(c_val, 2),
                    "adj_close": round(c_val, 2),
                    "volume": v_val,
                    "value_vnd": round(c_val * 1000.0 * v_val, 0) if symbol not in INDICES else 0,
                    "source": src,
                })
            return rows
        except Exception:
            continue
    return []


# ----------------------------------------------------------------------------
# 2. KÉO CỔ TỨC / QUYỀN & BCTC TỪ API CÔNG KHAI TCBS (KHÔNG CẦN TOKEN)
# ----------------------------------------------------------------------------
def fetch_tcbs_dividends(symbol):
    url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/company/{symbol}/dividend-payment-histories?page=0&size=50"
    try:
        data = http_get_json(url, timeout=15)
        items = (data or {}).get("listDividendPaymentHis") or []
        rows = []
        for it in items:
            raw_d = str(it.get("exerciseDate") or "").strip()
            if not raw_d:
                continue
            try:
                ex_date = datetime.strptime(raw_d, "%d/%m/%y").strftime("%Y-%m-%d")
            except ValueError:
                try:
                    ex_date = datetime.strptime(raw_d, "%d/%m/%Y").strftime("%Y-%m-%d")
                except ValueError:
                    continue
            if ex_date < "2016-01-01":
                continue
            method = str(it.get("issueMethod") or "").lower()
            pct = float(it.get("cashDividendPercentage") or 0)
            if "cash" in method or "tiền" in method:
                rows.append({
                    "symbol": symbol,
                    "ex_date": ex_date,
                    "action_type": "CASH_DIV",
                    "cash_amount": round(pct * 10.0, 2),  # 10% mệnh giá = 1.0 nghìn đồng
                    "ratio_num": 0,
                    "ratio_den": 1,
                    "description": f"Cổ tức tiền mặt {pct*100:.1f}%",
                })
            else:
                rows.append({
                    "symbol": symbol,
                    "ex_date": ex_date,
                    "action_type": "STOCK_DIV",
                    "cash_amount": 0,
                    "ratio_num": round(pct, 4),
                    "ratio_den": 1,
                    "description": f"Cổ tức cổ phiếu {pct*100:.1f}%",
                })
        return rows
    except Exception:
        return []


def fetch_tcbs_financials(symbol):
    """Lấy chỉ số tài chính năm (2016–2026) để tính Piotroski F-Score / Z-Score."""
    url = f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{symbol}/financialratio?yearly=1&isAll=true"
    try:
        items = http_get_json(url, timeout=15)
        if not isinstance(items, list):
            return []
        rows = []
        for it in items:
            yr = it.get("year")
            if not yr or int(yr) < 2016:
                continue
            fin_vals = {
                "roa": it.get("roa"),
                "roe": it.get("roe"),
                "grossMargin": it.get("grossProfitMargin"),
                "debtToEquity": it.get("debtOnEquity"),
                "currentRatio": it.get("currentPayment"),
                "assetTurnover": it.get("assetOnEquity"),
                "eps": it.get("earningPerShare"),
                "pe": it.get("priceToEarning"),
                "pb": it.get("priceToBook"),
                "revenueGrowth": it.get("revenueGrowth"),
            }
            payload_hash = hashlib.sha256(
                json.dumps({"s": symbol, "y": int(yr), "v": fin_vals}, sort_keys=True).encode()
            ).hexdigest()
            rows.append({
                "symbol": symbol,
                "year": int(yr),
                "quarter": 0,
                "company_type": "BANK" if symbol in ("ACB","BID","CTG","EIB","HDB","LPB","MBB","MSB","NAB","OCB","SHB","SSB","STB","TCB","TPB","VCB","VIB","VPB") else "CORP",
                "financial_values": fin_vals,
                "payload_hash": payload_hash,
            })
        return rows
    except Exception:
        return []


# ----------------------------------------------------------------------------
# 3. CHẠY TỔNG HỢP & XUẤT COLD BACKUP PARQUET + SHA-256
# ----------------------------------------------------------------------------
def run_backfill():
    latest = None if FORCE_FULL else get_latest_ohlcv_date()
    if latest:
        start_dt = datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=10)
        mode = f"INCREMENTAL (từ {start_dt.strftime('%Y-%m-%d')})"
    else:
        start_dt = datetime(2016, 1, 1)
        mode = "FULL BACKFILL 10 NĂM (2016-01-01 -> Hôm nay)"

    ts_from = int(start_dt.timestamp())
    ts_to = int((datetime.now() + timedelta(days=1)).timestamp())
    print(f"=== QUANTWE IMMUTABLE LEDGER | CHẾ ĐỘ: {mode} | {len(UNIVERSE)} MÃ ===")

    all_ohlcv, all_divs, all_fins = [], [], []
    is_full = latest is None or FORCE_FULL

    for idx, sym in enumerate(UNIVERSE, 1):
        ohlcv = fetch_ohlcv_symbol(sym, ts_from, ts_to)
        if ohlcv:
            sb_post_batches("ledger_ohlcv_daily", ohlcv)
            all_ohlcv.extend(ohlcv)

        if sym not in INDICES and (is_full or date.today().weekday() == 4):
            divs = fetch_tcbs_dividends(sym)
            if divs:
                sb_post_batches("ledger_corporate_actions", divs)
                all_divs.extend(divs)
            fins = fetch_tcbs_financials(sym)
            if fins:
                sb_post_batches("ledger_financials_pit", fins)
                all_fins.extend(fins)

        if idx % 15 == 0 or idx == len(UNIVERSE):
            print(f"  [{idx:3d}/{len(UNIVERSE)}] Đã nạp: OHLCV={len(all_ohlcv):,} | Quyền={len(all_divs):,} | BCTC={len(all_fins):,}")
        time.sleep(random.uniform(0.15, 0.35))

    # Xuất Cold Backup Parquet + đóng dấu SHA-256
    os.makedirs("immutable_backup", exist_ok=True)
    today_str = date.today().isoformat()
    if all_ohlcv:
        fname = f"ledger_ohlcv_{today_str}.parquet"
        fpath = os.path.join("immutable_backup", fname)
        pd.DataFrame(all_ohlcv).to_parquet(fpath, index=False)
        with open(fpath, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        sb_post_batches("ledger_backup_manifest", [{
            "backup_date": today_str,
            "file_name": fname,
            "table_name": "ledger_ohlcv_daily",
            "row_count": len(all_ohlcv),
            "sha256_hash": sha,
            "storage_uri": f"github-artifact://{fname}",
        }])
        print(f"\n[COLD BACKUP] {fname} | {len(all_ohlcv):,} dòng | SHA-256: {sha[:16]}...")

    print("[HOÀN TẤT] Đã chốt Sổ cái Bất biến thành công!")


if __name__ == "__main__":
    run_backfill()
