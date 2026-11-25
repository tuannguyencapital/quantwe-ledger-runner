"""
QUANTWE IMMUTABLE LEDGER — GIAI ĐOẠN 2: ANALYTICS ENGINE (CHUẨN KHÓA sb_secret_...)
Tự tính:
  1. JdK RRG (RS-Ratio, RS-Momentum, 4 góc phần tư so với VNINDEX)
  2. Piotroski F-Score (0-9), Altman Z-Score (phi tài chính), Beneish M-Score
  3. Chốt dấu SHA-256 & ghi Append-Only vào bảng ledger_signals_audit
"""
import hashlib
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"].strip()


def sb_headers(write=False):
    """Tự động nhận diện khóa mới sb_secret_... (bỏ Authorization) và khóa cũ eyJ..."""
    h = {"apikey": SUPABASE_SERVICE_KEY}
    if SUPABASE_SERVICE_KEY.startswith("eyJ"):
        h["Authorization"] = f"Bearer {SUPABASE_SERVICE_KEY}"
    if write:
        h["Content-Type"] = "application/json"
        h["Prefer"] = "resolution=ignore-duplicates,return=minimal"
    return h


def sb_fetch_all(table, select, filters="", page_size=1000):
    rows, offset = [], 0
    hdr = sb_headers(write=False)
    while True:
        url = f"{SUPABASE_URL}/rest/v1/{table}?select={select}{filters}&limit={page_size}&offset={offset}"
        r = requests.get(url, headers=hdr, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


# ----------------------------------------------------------------------------
# 1. ĐỒ THỊ RRG CHUẨN JdK (RS-RATIO & RS-MOMENTUM)
# ----------------------------------------------------------------------------
def compute_jdk_rrg(df_prices, benchmark="VNINDEX", window=10):
    if benchmark not in df_prices.columns:
        return {}
    bench = df_prices[benchmark].ffill()
    results = {}

    for sym in df_prices.columns:
        if sym in ("VNINDEX", "VN30", "HNXINDEX", "VN30F1M"):
            continue
        s = df_prices[sym].ffill()
        rs = (s / bench) * 100.0
        rs_ma = rs.rolling(window).mean()
        rs_std = rs.rolling(window).std(ddof=0).replace(0, np.nan)
        rs_ratio = 100.0 + ((rs - rs_ma) / rs_std)
        rs_ratio_smooth = rs_ratio.rolling(window).mean()

        mom_raw = rs_ratio_smooth.pct_change(1) * 100.0
        mom_ma = mom_raw.rolling(window).mean()
        mom_std = mom_raw.rolling(window).std(ddof=0).replace(0, np.nan)
        rs_mom = 100.0 + ((mom_raw - mom_ma) / mom_std)
        rs_mom_smooth = rs_mom.rolling(window).mean()

        r_val = rs_ratio_smooth.dropna()
        m_val = rs_mom_smooth.dropna()
        if r_val.empty or m_val.empty:
            continue
        rr, rm = float(r_val.iloc[-1]), float(m_val.iloc[-1])

        if rr >= 100 and rm >= 100:
            quad = "LEADING"
        elif rr >= 100 and rm < 100:
            quad = "WEAKENING"
        elif rr < 100 and rm < 100:
            quad = "LAGGING"
        else:
            quad = "IMPROVING"

        results[sym] = (round(rr, 2), round(rm, 2), quad)
    return results


# ----------------------------------------------------------------------------
# 2. FORENSIC HEALTH: PIOTROSKI F-SCORE, ALTMAN Z-SCORE, BENEISH M-SCORE
# ----------------------------------------------------------------------------
def _num(d, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def compute_forensic_scores(fin_curr, fin_prev, company_type=""):
    is_financial = str(company_type or "").upper() in (
        "BANK", "INSURANCE", "SECURITIES", "NGÂN HÀNG", "CHỨNG KHOÁN"
    )

    f_score = 0
    roa_c, roa_p = _num(fin_curr, "roa", "ROA"), _num(fin_prev, "roa", "ROA")
    cfo_c = _num(fin_curr, "cashFlowFromOperating", "operatingCashFlow", "cfo")
    ni_c = _num(fin_curr, "netIncome", "profitAfterTax", "netProfit")
    lev_c, lev_p = _num(fin_curr, "debtToEquity", "totalDebtToAssets"), _num(fin_prev, "debtToEquity", "totalDebtToAssets")
    cr_c, cr_p = _num(fin_curr, "currentRatio"), _num(fin_prev, "currentRatio")
    gm_c, gm_p = _num(fin_curr, "grossProfitMargin", "grossMargin"), _num(fin_prev, "grossProfitMargin", "grossMargin")
    at_c, at_p = _num(fin_curr, "assetTurnover"), _num(fin_prev, "assetTurnover")
    sh_c, sh_p = _num(fin_curr, "sharesOutstanding"), _num(fin_prev, "sharesOutstanding")

    if roa_c is not None and roa_c > 0: f_score += 1
    if cfo_c is not None and cfo_c > 0: f_score += 1
    if roa_c is not None and roa_p is not None and roa_c > roa_p: f_score += 1
    if cfo_c is not None and ni_c is not None and cfo_c > ni_c: f_score += 1
    if lev_c is not None and lev_p is not None and lev_c <= lev_p: f_score += 1
    if cr_c is not None and cr_p is not None and cr_c > cr_p: f_score += 1
    if sh_c is not None and sh_p is not None and sh_c <= sh_p: f_score += 1
    if gm_c is not None and gm_p is not None and gm_c > gm_p: f_score += 1
    if at_c is not None and at_p is not None and at_c > at_p: f_score += 1

    z_score = None
    if not is_financial:
        ta = _num(fin_curr, "totalAssets")
        wc = _num(fin_curr, "workingCapital")
        re = _num(fin_curr, "retainedEarnings")
        ebit = _num(fin_curr, "ebit", "operatingProfit")
        mcap = _num(fin_curr, "marketCap")
        tl = _num(fin_curr, "totalLiabilities")
        rev = _num(fin_curr, "revenue", "netSales")
        if ta and ta > 0 and tl and tl > 0:
            x1 = (wc or 0) / ta
            x2 = (re or 0) / ta
            x3 = (ebit or 0) / ta
            x4 = (mcap or 0) / tl
            x5 = (rev or 0) / ta
            z_score = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5, 2)

    m_score = None
    if not is_financial:
        rev_c, rev_p = _num(fin_curr, "revenue", "netSales"), _num(fin_prev, "revenue", "netSales")
        rec_c, rec_p = _num(fin_curr, "accountsReceivable", "shortTermReceivables"), _num(fin_prev, "accountsReceivable", "shortTermReceivables")
        ta_c = _num(fin_curr, "totalAssets")
        if rev_c and rev_p and rev_p > 0 and ta_c and ta_c > 0:
            sgi = rev_c / rev_p
            dsri = ((rec_c or 0) / rev_c) / (((rec_p or 0) / rev_p) or 1e-6) if rec_c and rec_p else 1.0
            gmi = (gm_p / gm_c) if (gm_c and gm_p and gm_c > 0) else 1.0
            tata = ((ni_c or 0) - (cfo_c or 0)) / ta_c
            m_score = round(-4.84 + 0.92 * dsri + 0.528 * gmi + 0.892 * sgi + 4.679 * tata, 2)

    return f_score, z_score, m_score


# ----------------------------------------------------------------------------
# 3. CHẠY TỔNG HỢP & CHỐT KIỂM TOÁN BẤT BIẾN (SHA-256)
# ----------------------------------------------------------------------------
def run_daily_analytics():
    since = (date.today() - timedelta(days=180)).isoformat()
    print(f"[1/3] Tải OHLCV 180 ngày gần nhất (từ {since}) từ Supabase...")
    ohlcv = sb_fetch_all("ledger_ohlcv_daily", "symbol,trade_date,adj_close", f"&trade_date=gte.{since}")
    if not ohlcv:
        raise RuntimeError("Bảng ledger_ohlcv_daily đang trống. Hãy chạy Backfill Bước 1 trước!")

    df = pd.DataFrame(ohlcv)
    latest_date = df["trade_date"].max()
    pivot = df.pivot_table(index="trade_date", columns="symbol", values="adj_close").sort_index()
    rrg_map = compute_jdk_rrg(pivot)
    print(f"   -> Đã tính RRG cho {len(rrg_map)} mã tại ngày chốt sổ {latest_date}.")

    print("[2/3] Tải BCTC năm gần nhất & chấm điểm Forensic (F/Z/M-Score)...")
    fin_rows = sb_fetch_all("ledger_financials_pit", "symbol,year,company_type,financial_values", "&quarter=eq.0&order=year.desc")
    fin_by_sym = {}
    for r in fin_rows:
        fin_by_sym.setdefault(r["symbol"], []).append(r)

    audit_records = []
    for sym, (rs_r, rs_m, quad) in rrg_map.items():
        f_list = fin_by_sym.get(sym, [])
        f_score, z_score, m_score = None, None, None
        if len(f_list) >= 2:
            f_score, z_score, m_score = compute_forensic_scores(
                f_list[0]["financial_values"],
                f_list[1]["financial_values"],
                f_list[0].get("company_type", ""),
            )

        quad_pts = {"LEADING": 45, "IMPROVING": 35, "WEAKENING": 20, "LAGGING": 10}[quad]
        f_pts = int((f_score or 4) * 5.5)
        qw_score = max(0, min(100, quad_pts + f_pts))

        core = {
            "symbol": sym,
            "signal_date": latest_date,
            "qw_score": qw_score,
            "riskguard_score": None,
            "rrg_quadrant": quad,
            "rs_ratio": rs_r,
            "rs_momentum": rs_m,
            "f_score": f_score,
            "z_score": z_score,
            "m_score": m_score,
        }
        core["payload_hash"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        audit_records.append(core)

    print(f"[3/3] Ghi kiểm toán bất biến {len(audit_records)} mã vào ledger_signals_audit...")
    url = f"{SUPABASE_URL}/rest/v1/ledger_signals_audit"
    hdr_write = sb_headers(write=True)
    for i in range(0, len(audit_records), 200):
        r = requests.post(url, headers=hdr_write, json=audit_records[i : i + 200], timeout=30)
        if r.status_code not in (200, 201, 204):
            print(f"  [WARN] batch {i}: {r.status_code} {r.text[:200]}")

    top5 = sorted(audit_records, key=lambda x: x["qw_score"], reverse=True)[:5]
    print("\n[HOÀN TẤT] Top 5 mã QW Score cao nhất hôm nay:")
    for t in top5:
        print(f"  - {t['symbol']:4s} | QW: {t['qw_score']:3d} | RRG: {t['rrg_quadrant']:9s} (RS={t['rs_ratio']}, Mom={t['rs_momentum']}) | F={t['f_score']} | Z={t['z_score']}")


if __name__ == "__main__":
    run_daily_analytics()
