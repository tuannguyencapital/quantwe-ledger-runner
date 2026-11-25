"""
QUANTWE IMMUTABLE LEDGER — ANALYTICS ENGINE v2.0 (6-PILLAR CONTINUOUS SCORE 1–99)
- Tính điểm QW Score phân hóa liên tục (1–99) dựa trên 6 Trụ cột định lượng:
  1. JdK RRG (RS-Ratio & RS-Momentum liên tục)          : 25 điểm
  2. Cấu trúc xu hướng (MA20 / MA50 / MA200 + Độ dốc)  : 20 điểm
  3. Dòng tiền & Khối lượng (VPA Accumulation/Vol20)   : 15 điểm
  4. Động lượng & Vị thế 52 tuần (RSI14 + 52W Range)   : 15 điểm
  5. Độ ổn định & Rủi ro (ATR14 + 60D Max Drawdown)    : 10 điểm
  6. Sức khỏe Cơ bản (Piotroski F-Score từ VNDirect)   : 15 điểm (tự quy đổi về 99 nếu thiếu BCTC)
"""
import hashlib
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    from curl_cffi import requests as http_req
    USE_CFFI = True
except ImportError:
    import requests as http_req
    USE_CFFI = False

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"].strip()
INDICES = {"VNINDEX", "VN30", "HNXINDEX", "VN30F1M"}


def sb_headers(write=False):
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
        r = http_req.get(url, headers=hdr, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def fetch_vndirect_fscores(symbols):
    """Kéo nhanh chỉ số tài chính 2 năm gần nhất từ VNDirect Finfo API (không bị chặn IP)."""
    codes = ",".join([s for s in symbols if s not in INDICES])
    url = (
        "https://finfo-api.vndirect.com.vn/v4/ratios/latest"
        f"?filter=ratioCode:ROA_Y,ROE_Y,GROSS_MARGIN_Y,DEBT_TO_EQUITY_Y,CURRENT_RATIO_Y,REV_GROWTH_YOY,NET_PROFIT_GROWTH_YOY,CFO_TO_NET_PROFIT_Y"
        f"&where=code:{codes}&size=2000"
    )
    kwargs = {"headers": {"User-Agent": "Mozilla/5.0"}, "timeout": 20}
    if USE_CFFI:
        kwargs["impersonate"] = "chrome120"
    try:
        r = http_req.get(url, **kwargs)
        if r.status_code != 200:
            print(f"  [INFO] VNDirect Ratios HTTP {r.status_code} -> dùng quy đổi 5 trụ kỹ thuật.")
            return {}
        data = r.json().get("data") or []
        by_sym = {}
        for item in data:
            by_sym.setdefault(item.get("code"), {})[item.get("ratioCode")] = item.get("value")

        f_map = {}
        for sym, m in by_sym.items():
            pts = 0
            if (m.get("ROA_Y") or 0) > 0: pts += 2
            if (m.get("ROE_Y") or 0) > 0.12: pts += 1
            if (m.get("GROSS_MARGIN_Y") or 0) > 0.15: pts += 1
            if (m.get("REV_GROWTH_YOY") or 0) > 0: pts += 1
            if (m.get("NET_PROFIT_GROWTH_YOY") or 0) > 0: pts += 1
            if (m.get("CFO_TO_NET_PROFIT_Y") or 0) > 0.8: pts += 1
            if m.get("DEBT_TO_EQUITY_Y") is not None and m["DEBT_TO_EQUITY_Y"] < 1.5: pts += 1
            if (m.get("CURRENT_RATIO_Y") or 0) > 1.2: pts += 1
            f_map[sym] = max(1, min(9, pts))
        print(f"  -> Đã kéo thành công điểm cơ bản (F-Score) từ VNDirect cho {len(f_map)} mã.")
        return f_map
    except Exception as e:
        print(f"  [INFO] Bỏ qua VNDirect Ratios ({e}) -> dùng quy đổi 5 trụ kỹ thuật.")
        return {}


def clip(val, lo, hi):
    return max(lo, min(hi, float(val)))


def compute_jdk_rrg(df_close, benchmark="VNINDEX", window=10):
    if benchmark not in df_close.columns:
        return {}
    bench = df_close[benchmark].ffill()
    results = {}
    for sym in df_close.columns:
        if sym in INDICES:
            continue
        s = df_close[sym].ffill()
        rs = (s / bench) * 100.0
        rs_ma = rs.rolling(window).mean()
        rs_std = rs.rolling(window).std(ddof=0).replace(0, np.nan)
        rs_ratio = (100.0 + ((rs - rs_ma) / rs_std)).rolling(window).mean()

        mom_raw = rs_ratio.pct_change(1) * 100.0
        mom_ma = mom_raw.rolling(window).mean()
        mom_std = mom_raw.rolling(window).std(ddof=0).replace(0, np.nan)
        rs_mom = (100.0 + ((mom_raw - mom_ma) / mom_std)).rolling(window).mean()

        r_val, m_val = rs_ratio.dropna(), rs_mom.dropna()
        if r_val.empty or m_val.empty:
            continue
        rr, rm = float(r_val.iloc[-1]), float(m_val.iloc[-1])
        if rr >= 100 and rm >= 100: quad = "LEADING"
        elif rr >= 100 and rm < 100: quad = "WEAKENING"
        elif rr < 100 and rm < 100: quad = "LAGGING"
        else: quad = "IMPROVING"
        results[sym] = (round(rr, 2), round(rm, 2), quad)
    return results


def score_6_pillars(sym_df, rs_r, rs_m, quad, f_score):
    """Chấm điểm 6 Trụ cột định lượng phân hóa liên tục 1..99."""
    c = sym_df["adj_close"].astype(float)
    h = sym_df["high_raw"].astype(float).fillna(c)
    l = sym_df["low_raw"].astype(float).fillna(c)
    v = sym_df["volume"].astype(float).fillna(0)
    px = float(c.iloc[-1])

    # Trụ 1: RRG liên tục (0..25)
    base_q = {"LEADING": 16.0, "IMPROVING": 12.0, "WEAKENING": 8.0, "LAGGING": 3.0}[quad]
    p1 = clip(base_q + (rs_r - 100.0) * 3.2 + (rs_m - 100.0) * 2.5, 0, 25)

    # Trụ 2: Cấu trúc xu hướng MA20 / MA50 / MA200 (0..20)
    ma20 = float(c.tail(20).mean())
    ma50 = float(c.tail(50).mean())
    ma200 = float(c.tail(200).mean()) if len(c) >= 120 else ma50
    ma20_prev = float(c.iloc[-6:-1].mean()) if len(c) >= 25 else ma20
    p2 = 0.0
    if px > ma20: p2 += 5.5
    if px > ma50: p2 += 5.5
    if px > ma200: p2 += 4.0
    if ma20 > ma50: p2 += 3.0
    if ma20 > ma20_prev: p2 += 2.0

    # Trụ 3: Dòng tiền & Khối lượng VPA (0..15)
    vol5 = float(v.tail(5).mean())
    vol20 = float(v.tail(20).mean()) or 1.0
    vol_ratio = vol5 / vol20
    ret = c.pct_change().fillna(0)
    up_vol = float(v.tail(20)[ret.tail(20) > 0].sum())
    tot_vol = float(v.tail(20).sum()) or 1.0
    acc_ratio = up_vol / tot_vol
    p3 = clip((vol_ratio - 0.6) * 6.0, 0, 7.5) + clip(acc_ratio * 12.0, 0, 7.5)

    # Trụ 4: Vị thế 52 tuần & RSI14 (0..15)
    hi52, lo52 = float(c.max()), float(c.min())
    pos52 = (px - lo52) / (hi52 - lo52) if hi52 > lo52 else 0.5
    delta = c.diff()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    rsi = 100.0 - (100.0 / (1.0 + (gain / loss))) if loss and loss > 0 else 50.0
    rsi_pts = 7.5 - abs(rsi - 60.0) * 0.22  # Vùng tối ưu RSI 55-65
    p4 = clip(pos52 * 7.5, 0, 7.5) + clip(rsi_pts, 0, 7.5)

    # Trụ 5: Độ ổn định & Drawdown 60 phiên (0..10)
    peak60 = c.tail(60).cummax()
    dd60 = float(((c.tail(60) - peak60) / peak60).min())  # số âm
    p5 = clip(10.0 + dd60 * 35.0, 0, 10)

    tech_total = p1 + p2 + p3 + p4 + p5  # tối đa 85

    # Trụ 6: Forensic Cơ bản (0..15)
    if f_score is not None:
        p6 = clip((f_score / 9.0) * 15.0, 0, 15)
        qw_score = round(tech_total + p6)
    else:
        qw_score = round((tech_total / 85.0) * 99.0)

    return int(clip(qw_score, 1, 99))


def run_daily_analytics():
    since = (date.today() - timedelta(days=365)).isoformat()
    print(f"[1/3] Tải OHLCV 1 năm gần nhất (từ {since}) từ Supabase...")
    ohlcv = sb_fetch_all(
        "ledger_ohlcv_daily",
        "symbol,trade_date,high_raw,low_raw,adj_close,volume",
        f"&trade_date=gte.{since}",
    )
    if not ohlcv:
        raise RuntimeError("Bảng ledger_ohlcv_daily đang trống!")

    df = pd.DataFrame(ohlcv)
    latest_date = df["trade_date"].max()
    pivot = df.pivot_table(index="trade_date", columns="symbol", values="adj_close").sort_index()
    rrg_map = compute_jdk_rrg(pivot)
    print(f"  -> Tính xong RRG cho {len(rrg_map)} mã tại ngày {latest_date}.")

    print("[2/3] Kéo điểm cơ bản (VNDirect Finfo) & chấm điểm 6 Trụ cột QuantWe (1-99)...")
    f_map = fetch_vndirect_fscores(list(rrg_map.keys()))

    grouped = {sym: g.sort_values("trade_date") for sym, g in df.groupby("symbol")}
    audit_records = []

    for sym, (rs_r, rs_m, quad) in rrg_map.items():
        sym_df = grouped.get(sym)
        if sym_df is None or len(sym_df) < 25:
            continue
        f_score = f_map.get(sym)
        qw_score = score_6_pillars(sym_df, rs_r, rs_m, quad, f_score)

        core = {
            "symbol": sym,
            "signal_date": latest_date,
            "qw_score": qw_score,
            "riskguard_score": None,
            "rrg_quadrant": quad,
            "rs_ratio": rs_r,
            "rs_momentum": rs_m,
            "f_score": f_score,
            "z_score": None,
            "m_score": None,
        }
        core["payload_hash"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        audit_records.append(core)

    print(f"[3/3] Ghi kiểm toán bất biến {len(audit_records)} mã vào ledger_signals_audit...")
    url = f"{SUPABASE_URL}/rest/v1/ledger_signals_audit"
    hdr_write = sb_headers(write=True)
    for i in range(0, len(audit_records), 200):
        r = http_req.post(url, headers=hdr_write, json=audit_records[i : i + 200], timeout=30)
        if r.status_code not in (200, 201, 204):
            print(f"  [WARN] batch {i}: {r.status_code} {r.text[:200]}")

    top8 = sorted(audit_records, key=lambda x: x["qw_score"], reverse=True)[:8]
    print("\n[HOÀN TẤT] Top 8 mã QW Score 6 Trụ cột hôm nay:")
    for t in top8:
        print(f"  - {t['symbol']:4s} | QW: {t['qw_score']:2d}/99 | RRG: {t['rrg_quadrant']:9s} (RS={t['rs_ratio']}, Mom={t['rs_momentum']}) | F={t['f_score']}")


if __name__ == "__main__":
    run_daily_analytics()
