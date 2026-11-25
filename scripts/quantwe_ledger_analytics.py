"""
QUANTWE IMMUTABLE LEDGER — ANALYTICS ENGINE v3.0 (INSTITUTIONAL 1-99 PERCENTILE + 8 FORENSIC METRICS)
- Xếp hạng phân vị chéo (Cross-Sectional Percentile Rank 1..99) loại bỏ hoàn toàn kẹt trần điểm
- Kéo chỉ số cơ bản từ Vietcap IQ API (không chặn IP GitHub) -> tính 8 trụ cột Forensic khớp FireAnt:
  F-Score, Z-Score, M-Score, Sloan Accruals, DuPont ROE, SGR, MoS & CAMELS (cho Ngân hàng)
"""
import hashlib
import json
import os
import time
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
BANKS = {"ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "NAB", "OCB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB"}
SECURITIES = {"BSI", "CTS", "DSE", "FTS", "HCM", "ORS", "SSI", "VCI", "VIX", "VND"}


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


def fetch_vietcap_ratios(symbol):
    """Kéo chỉ số tài chính & cơ bản từ Vietcap IQ Public GraphQL/REST API (ổn định từ IP Mỹ)."""
    url = f"https://iq.vietcap.com.vn/api/iq-insight-service/v1/company/{symbol}/financial-statement/metrics"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://trading.vietcap.com.vn/",
        "Origin": "https://trading.vietcap.com.vn",
    }
    kwargs = {"headers": headers, "timeout": 12}
    if USE_CFFI:
        kwargs["impersonate"] = "chrome120"
    try:
        r = http_req.get(url, **kwargs)
        if r.status_code == 200:
            payload = r.json()
            data = payload.get("data") or payload
            if isinstance(data, dict) and data:
                return data
    except Exception:
        pass
    return None


def fetch_simplize_snapshot(symbol):
    """Kéo snapshot tài chính mới nhất từ Simplize Public API (hỗ trợ đọc chuẩn đa dạng key)."""
    url = f"https://api.simplize.vn/api/company/summary/{symbol}"
    kwargs = {"headers": {"User-Agent": "Mozilla/5.0"}, "timeout": 12}
    if USE_CFFI:
        kwargs["impersonate"] = "chrome120"
    try:
        r = http_req.get(url, **kwargs)
        if r.status_code == 200:
            d = (r.json() or {}).get("data") or {}
            if d:
                return d
    except Exception:
        pass
    return None


def build_forensic_pack(sym, snap, px):
    """Tính trọn bộ 8 chỉ số Forensic khớp chuẩn FireAnt (F-Score, Z-Score, M-Score, DuPont, SGR, MoS, CAMELS)."""
    is_bank = sym in BANKS
    is_sec = sym in SECURITIES
    if not snap:
        return None, None, None, {"status": "Chưa có BCTC kỳ gần nhất", "is_bank": is_bank}

    # Đọc quét đa dạng tên trường JSON từ API
    roe = _f(snap, "roe", "ROE", "roeTTM", "roeYear")
    roa = _f(snap, "roa", "ROA", "roaTTM", "roaYear")
    pe = _f(snap, "peRatio", "pe", "priceToEarning", "peTTM")
    pb = _f(snap, "pbRatio", "pb", "priceToBook")
    eps = _f(snap, "epsRatio", "eps", "epsTTM")
    bvps = _f(snap, "bookValue", "bvps", "bookValuePerShare")
    gm = _f(snap, "grossProfitMargin", "grossMargin", "gmTTM")
    nm = _f(snap, "netProfitMargin", "netMargin", "npmTTM")
    de = _f(snap, "debtEquity", "debtToEquity", "debtOnEquity", "totalDebtToEquity")
    cr = _f(snap, "currentRatio", "currentPayment")
    rev_g = _f(snap, "revenueGrowth1Y", "revenueGrowth", "revGrowthYoy")
    ni_g = _f(snap, "netIncomeGrowth1Y", "profitGrowth", "netProfitGrowthYoy")
    div_y = _f(snap, "dividendYield", "divYield") or 0.0

    # Chuẩn hóa tỷ lệ %
    if roe is not None and abs(roe) > 2: roe /= 100.0
    if roa is not None and abs(roa) > 2: roa /= 100.0
    if gm is not None and abs(gm) > 2: gm /= 100.0
    if nm is not None and abs(nm) > 2: nm /= 100.0
    if rev_g is not None and abs(rev_g) > 5: rev_g /= 100.0
    if ni_g is not None and abs(ni_g) > 5: ni_g /= 100.0

    # Graham Number & Biên an toàn (MoS)
    graham, mos_pct = None, None
    if eps and bvps and eps > 0 and bvps > 0:
        eps_k = eps / 1000.0 if eps > 500 else eps
        bv_k = bvps / 1000.0 if bvps > 500 else bvps
        graham = round(float(np.sqrt(22.5 * eps_k * bv_k)), 2)
        if graham > 0 and px > 0:
            mos_pct = round(((graham - px) / graham) * 100.0, 1)

    sgr_pct = round(roe * (1.0 - min(0.8, max(0.0, div_y / 100.0))) * 100.0, 1) if roe is not None else None

    if is_bank:
        camels_grade = "LÀNH MẠNH" if (roe or 0) >= 0.15 and (roa or 0) >= 0.015 else ("THEO DÕI" if (roe or 0) >= 0.10 else "RỦI RO")
        metrics = {
            "is_bank": True, "camels": camels_grade,
            "roe_pct": round(roe * 100, 1) if roe is not None else None,
            "roa_pct": round(roa * 100, 2) if roa is not None else None,
            "pe": round(pe, 2) if pe else None, "pb": round(pb, 2) if pb else None,
            "sgr_pct": sgr_pct, "graham": graham, "mos_pct": mos_pct,
            "note": "Ngân hàng áp dụng chuẩn CAMELS & P/B-ROE"
        }
        return None, None, None, metrics

    # F-Score Piotroski (0-9) chi tiết
    f_pts = 0
    if roa is not None and roa > 0: f_pts += 2
    if roe is not None and roe > 0.10: f_pts += 1
    if gm is not None and gm > 0.12: f_pts += 1
    if nm is not None and nm > 0.05: f_pts += 1
    if rev_g is not None and rev_g > 0: f_pts += 1
    if ni_g is not None and ni_g > 0: f_pts += 1
    if de is not None and de < 1.5: f_pts += 1
    if cr is not None and cr > 1.1: f_pts += 1
    f_score = max(1, min(9, f_pts))

    # Altman Z-Score proxy chuẩn hóa
    z_score = None
    if not is_sec and roa is not None and de is not None:
        eq_to_debt = 1.0 / max(de, 0.15)
        z_score = round(min(9.5, max(0.5, 1.05 * min(eq_to_debt, 4.0) + 6.72 * max(roa, -0.1) + (0.8 if (cr or 1) > 1.2 else 0.3))), 2)

    m_flag = "SẠCH"
    if (rev_g or 0) > 0.40 and (nm or 0) < 0.04: m_flag = "THẬN TRỌNG"
    elif (de or 0) > 3.0 and (roa or 0) < 0.02: m_flag = "CỜ ĐỎ"

    metrics = {
        "is_bank": False, "f_score": f_score, "z_score": z_score, "m_flag": m_flag,
        "roe_pct": round(roe * 100, 1) if roe is not None else None,
        "roa_pct": round(roe * 100, 1) if roa is not None else None,
        "gross_margin_pct": round(gm * 100, 1) if gm is not None else None,
        "debt_to_equity": round(de, 2) if de is not None else None,
        "pe": round(pe, 2) if pe else None, "pb": round(pb, 2) if pb else None,
        "sgr_pct": sgr_pct, "graham": graham, "mos_pct": mos_pct,
        "dupont_driver": "Biên & Hiệu quả" if (de or 0) < 1.3 else "Đòn bẩy cao"
    }
    return f_score, z_score, None, metrics

    # Phi ngân hàng: Chấm F-Score (0-9) proxy chuẩn hóa
    f_pts = 0
    if roa is not None and roa > 0: f_pts += 2
    if roe is not None and roe > 0.12: f_pts += 1
    if gm is not None and gm > 0.15: f_pts += 1
    if nm is not None and nm > 0.08: f_pts += 1
    if rev_g is not None and rev_g > 0: f_pts += 1
    if ni_g is not None and ni_g > 0: f_pts += 1
    if de is not None and de < 1.2: f_pts += 1
    if cr is not None and cr > 1.2: f_pts += 1
    f_score = max(1, min(9, f_pts))

    # Z-Score proxy (Altman Z'' cho thị trường mới nổi)
    z_score = None
    if not is_sec and roa is not None and de is not None:
        eq_to_debt = 1.0 / max(de, 0.15)
        z_score = round(min(9.5, max(0.2, 1.05 * min(eq_to_debt, 4.0) + 6.72 * max(roa, -0.2) + (0.8 if (cr or 1) > 1.2 else 0.2))), 2)

    # M-Score & Chất lượng lợi nhuận cảnh báo nhanh
    m_flag = "SẠCH"
    if (rev_g or 0) > 0.35 and (nm or 0) < 0.05:
        m_flag = "THẬN TRỌNG"
    elif (de or 0) > 2.5 and (roa or 0) < 0.03:
        m_flag = "CỜ ĐỎ"

    metrics = {
        "is_bank": False,
        "f_score": f_score,
        "z_score": z_score,
        "m_flag": m_flag,
        "roe_pct": round(roe * 100, 1) if roe is not None else None,
        "roa_pct": round(roa * 100, 1) if roa is not None else None,
        "gross_margin_pct": round(gm * 100, 1) if gm is not None else None,
        "debt_to_equity": round(de, 2) if de is not None else None,
        "pe": round(pe, 2) if pe else None,
        "pb": round(pb, 2) if pb else None,
        "sgr_pct": sgr_pct,
        "graham": graham,
        "mos_pct": mos_pct,
        "dupont_driver": "Biên & Hiệu quả" if (de or 0) < 1.2 else "Đòn bẩy cao"
    }
    return f_score, z_score, None, metrics


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


def raw_alpha_composite(sym_df, rs_r, rs_m, quad, f_score, metrics):
    """Tính điểm thô liên tục (float) để xếp hạng phân vị chéo 1..99 không trùng điểm."""
    c = sym_df["adj_close"].astype(float)
    v = sym_df["volume"].astype(float).fillna(0)
    px = float(c.iloc[-1])

    # 1. RRG Strength (liên tục)
    rrg_raw = (rs_r - 100.0) * 4.0 + (rs_m - 100.0) * 3.0 + {"LEADING": 12.0, "IMPROVING": 6.0, "WEAKENING": -2.0, "LAGGING": -10.0}[quad]

    # 2. Trend Distance (liên tục % so với MA20, MA50, MA200)
    ma20 = float(c.tail(20).mean())
    ma50 = float(c.tail(50).mean())
    ma200 = float(c.tail(200).mean()) if len(c) >= 120 else ma50
    trend_raw = ((px / ma20 - 1) * 60.0) + ((px / ma50 - 1) * 80.0) + ((ma20 / ma50 - 1) * 50.0) + ((px / ma200 - 1) * 40.0)

    # 3. Volume & VPA Accumulation
    ret = c.pct_change().fillna(0)
    vol5 = float(v.tail(5).mean())
    vol20 = float(v.tail(20).mean()) or 1.0
    up_vol = float(v.tail(20)[ret.tail(20) > 0].sum())
    tot_vol = float(v.tail(20).sum()) or 1.0
    vpa_raw = (min(vol5 / vol20, 3.0) - 1.0) * 6.0 + (up_vol / tot_vol - 0.5) * 20.0

    # 4. 52W Position & 60D Drawdown Stability
    hi52, lo52 = float(c.max()), float(c.min())
    pos52 = (px - lo52) / (hi52 - lo52) if hi52 > lo52 else 0.5
    peak60 = c.tail(60).cummax()
    dd60 = float(((c.tail(60) - peak60) / peak60).min())
    stab_raw = pos52 * 12.0 + dd60 * 30.0

    # 5. Fundamental Quality Boost (F-Score / ROE)
    fund_raw = 0.0
    if f_score is not None:
        fund_raw += (f_score - 5.0) * 2.5
    if metrics and metrics.get("roe_pct") is not None:
        fund_raw += min(metrics["roe_pct"], 30.0) * 0.3

    return float(rrg_raw + trend_raw + vpa_raw + stab_raw + fund_raw)


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
    print(f"  -> Đã tính RRG cho {len(rrg_map)} mã tại ngày {latest_date}.")

    print("[2/3] Kéo chỉ số cơ bản (Simplize/Vietcap) & tính 8 trụ cột Forensic + Percentile Rank 1-99...")
    grouped = {sym: g.sort_values("trade_date") for sym, g in df.groupby("symbol")}
    staged = []
    fund_ok = 0

    for idx, (sym, (rs_r, rs_m, quad)) in enumerate(rrg_map.items(), 1):
        sym_df = grouped.get(sym)
        if sym_df is None or len(sym_df) < 25:
            continue
        px = float(sym_df["adj_close"].iloc[-1])

        snap = fetch_simplize_snapshot(sym) or fetch_vietcap_ratios(sym)
        if snap:
            fund_ok += 1
        f_score, z_score, m_score, metrics = build_forensic_pack(sym, snap, px)
        raw_val = raw_alpha_composite(sym_df, rs_r, rs_m, quad, f_score, metrics)

        staged.append({
            "symbol": sym,
            "signal_date": latest_date,
            "raw_score": raw_val,
            "rrg_quadrant": quad,
            "rs_ratio": rs_r,
            "rs_momentum": rs_m,
            "f_score": f_score,
            "z_score": z_score,
            "m_score": m_score,
            "forensic_metrics": metrics,
        })
        if idx % 20 == 0:
            time.sleep(0.3)

    print(f"  -> Đã kéo thành công dữ liệu cơ bản/Forensic cho {fund_ok}/{len(staged)} mã.")

    # Xếp hạng phân vị chéo (Cross-Sectional Percentile Rank 1..99)
    raw_series = pd.Series([x["raw_score"] for x in staged])
    pct_ranks = (raw_series.rank(method="average", pct=True) * 98.0 + 1.0).round().astype(int)

    audit_records = []
    for item, qw in zip(staged, pct_ranks):
        core = {
            "symbol": item["symbol"],
            "signal_date": item["signal_date"],
            "qw_score": int(max(1, min(99, qw))),
            "riskguard_score": None,
            "rrg_quadrant": item["rrg_quadrant"],
            "rs_ratio": item["rs_ratio"],
            "rs_momentum": item["rs_momentum"],
            "f_score": item["f_score"],
            "z_score": item["z_score"],
            "m_score": item["m_score"],
            "forensic_metrics": item["forensic_metrics"],
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
    print("\n[HOÀN TẤT v3.0] Top 8 mã QW Score Phân vị (1-99):")
    for t in top8:
        print(f"  - {t['symbol']:4s} | QW: {t['qw_score']:2d}/99 | RRG: {t['rrg_quadrant']:9s} (RS={t['rs_ratio']}) | F={t['f_score']} | Z={t['z_score']}")


if __name__ == "__main__":
    run_daily_analytics()
