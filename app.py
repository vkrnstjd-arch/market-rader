
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Market Radar · Portfolio OS", page_icon="🧭", layout="wide")

MACRO = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "S&P500": "^GSPC",
    "BTC": "BTC-USD",
    "GOLD": "GC=F",   # Gold futures price proxy
}

M7 = {
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "NVDA": "NVDA",
    "GOOGL": "GOOGL",
    "META": "META",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
}

ALL = {**MACRO, **M7}
AUX_ASSETS = ["KOSDAQ", "BTC", "GOLD", *M7.keys()]


# =========================================================
# DATA
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_close(ticker: str) -> pd.Series:
    df = yf.download(
        ticker,
        start="1980-01-01",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index)
    try:
        close.index = close.index.tz_localize(None)
    except Exception:
        pass
    return close


def periods_per_year(asset_name: str) -> int:
    return 365 if asset_name == "BTC" else 252


# =========================================================
# INDICATORS
# =========================================================
def indicator_series(close: pd.Series, asset_name: str):
    # True calendar 52-week high: fixes BTC's 7-day trading issue.
    high_52w = close.rolling("365D", min_periods=60).max()
    dd52 = close / high_52w - 1

    ma50 = close.rolling(50, min_periods=30).mean()
    sep50 = close / ma50 - 1

    ret12 = close.pct_change(periods_per_year(asset_name))
    return dd52, sep50, ret12


def distress_percentile(series: pd.Series, current: float, years: int) -> float:
    cutoff = series.index.max() - pd.DateOffset(years=years)
    hist = series.loc[series.index >= cutoff].dropna()
    if len(hist) == 0 or pd.isna(current):
        return np.nan

    # More negative than history = higher distress score
    return float((hist >= current).mean() * 100)


def euphoria_percentile(series: pd.Series, current: float, years: int) -> float:
    cutoff = series.index.max() - pd.DateOffset(years=years)
    hist = series.loc[series.index >= cutoff].dropna()
    if len(hist) == 0 or pd.isna(current):
        return np.nan

    return float((hist <= current).mean() * 100)


def c_score_from_mdd(mdd_pct, asset="KOSPI"):
    """
    Absolute-MDD C-score. KOSPI and S&P500 use separate absolute anchors.
    This is a drawdown-severity score, not a valuation estimate.
    """
    if pd.isna(mdd_pct):
        return np.nan
    x = max(0.0, -float(mdd_pct))

    if asset == "S&P500":
        anchors_x = np.array([0, 3, 5, 7.5, 10, 15, 20, 25, 30], dtype=float)
        anchors_c = np.array([20, 40, 50, 70, 80, 90, 95, 98, 100], dtype=float)
        if x >= 30:
            return 100.0
        return float(np.interp(x, anchors_x, anchors_c))

    anchors_x = np.array([0, 3, 5, 7.5, 10, 15, 20, 25, 30, 35, 40], dtype=float)
    anchors_c = np.array([20, 40, 50, 60, 70, 80, 90, 95, 97.5, 99, 100], dtype=float)
    if x >= 40:
        return 100.0
    return float(np.interp(x, anchors_x, anchors_c))

def rating_label(score):
    if pd.isna(score):
        return "—"
    if score >= 100:
        return "🔴 역사적 위기"
    if score >= 97.5:
        return "🔴 매우 큰 폭락"
    if score >= 95:
        return "🟠 대폭락"
    if score >= 90:
        return "🟠 깊은 조정"
    if score >= 80:
        return "🟡 큰 조정"
    if score >= 70:
        return "🟢 본격 조정"
    return "⚪ 평범"


def calc_metrics(close: pd.Series, asset: str, years: int):
    dd, sep, ret12 = indicator_series(close, asset)

    cur_dd = dd.iloc[-1]
    cur_sep = sep.iloc[-1]
    cur_ret = ret12.iloc[-1]

    # C-score is used only for KOSPI / S&P500 cash-engine reference.
    # BTC / GOLD / KOSDAQ / M7 use a separate all-history ATH-drawdown cheapness test.
    c = c_score_from_mdd(cur_dd * 100, asset) if asset in MARKET_RULES else np.nan

    # E-score: unchanged. 12m return percentile 60% + 50d separation percentile 40%.
    p_ret_up = euphoria_percentile(ret12, cur_ret, years)
    p_sep_up = euphoria_percentile(sep, cur_sep, years)
    e = 0.60 * p_ret_up + 0.40 * p_sep_up

    return {
        "현재가": close.iloc[-1],
        "52주 MDD": cur_dd * 100,
        "50일 이격": cur_sep * 100,
        "12개월 수익률": cur_ret * 100 if pd.notna(cur_ret) else np.nan,
        "C-score": c,
        "판정": rating_label(c),
        "E-score": e,
        "기준일": str(close.index[-1].date()),
    }




# =========================================================
# AUXILIARY ASSET CHEAPNESS: ALL-HISTORY ATH DRAWDOWN
# =========================================================
def conservative_aux_label(bottom_gap_pct: float) -> str:
    """
    Conservative classification based on how far today's price sits ABOVE the
    worst historical drawdown bottom, measured in actual price terms.

    Example: worst MDD -50% => worst bottom price = 50 (peak=100).
             current MDD -45% => current price = 55 => bottom gap = +10%, not 5%p.
    """
    if pd.isna(bottom_gap_pct):
        return "—"
    g = round(max(0.0, float(bottom_gap_pct)), 8)
    if g <= 5.0:
        return "🔥 극단적 매수 구간"
    if g <= 10.0:
        return "🔴 매우 싸다"
    if g <= 20.0:
        return "🟠 싸다"
    if g <= 35.0:
        return "🟡 관심 구간"
    return "⚪ 보통"


def all_history_drawdown_metrics(close: pd.Series):
    """Compare current ATH drawdown with the worst ATH drawdown in all available data."""
    s = close.dropna().astype(float).sort_index()
    if s.empty:
        return {}
    running_ath = s.cummax()
    dd = s / running_ath - 1.0
    cur_dd = float(dd.iloc[-1])
    worst_dd = float(dd.min())
    worst_date = dd.idxmin()

    # Price-relative gap to the historical worst bottom, NOT MDD percentage-point gap.
    # Both are normalized to their own preceding ATH = 1.
    denom = 1.0 + worst_dd
    bottom_gap = ((1.0 + cur_dd) / denom - 1.0) * 100 if denom > 0 else np.nan

    return {
        "현재가": float(s.iloc[-1]),
        "ATH 대비 현재 MDD": cur_dd * 100,
        "역사적 최대 MDD": worst_dd * 100,
        "역사적 바닥 대비 괴리": bottom_gap,
        "역사적 최대 MDD 날짜": str(pd.Timestamp(worst_date).date()),
        "판정": conservative_aux_label(bottom_gap),
        "기준일": str(pd.Timestamp(s.index[-1]).date()),
    }


# =========================================================
# MDD FREQUENCY
# =========================================================
def mdd_day_frequency(close: pd.Series, years=None):
    high_52w = close.rolling("365D", min_periods=60).max()
    dd = (close / high_52w - 1).dropna()

    if years is not None:
        cutoff = dd.index.max() - pd.DateOffset(years=years)
        dd = dd.loc[dd.index >= cutoff]

    result = {}
    for level in [10, 20, 30, 40, 50]:
        result[f"-{level}%"] = float((dd <= -level / 100).mean() * 100)
    return result


def mdd_entry_frequency(close: pd.Series, years=None):
    """
    Counts new entries below each MDD threshold.
    Example: -20% -> the day the asset first crosses from above -20% to <= -20%.
    """
    high_52w = close.rolling("365D", min_periods=60).max()
    dd = (close / high_52w - 1).dropna()

    if years is not None:
        cutoff = dd.index.max() - pd.DateOffset(years=years)
        dd = dd.loc[dd.index >= cutoff]

    if len(dd) < 2:
        return {f"-{x}%": "—" for x in [10,20,30,40,50]}

    elapsed_years = max((dd.index[-1] - dd.index[0]).days / 365.25, 0.5)
    result = {}

    for level in [10, 20, 30, 40, 50]:
        threshold = -level / 100
        crossings = ((dd <= threshold) & (dd.shift(1) > threshold)).sum()

        if crossings == 0:
            result[f"-{level}%"] = "관측 없음"
        else:
            years_per = elapsed_years / crossings
            if years_per < 1:
                result[f"-{level}%"] = f"연 {1/years_per:.1f}회"
            else:
                result[f"-{level}%"] = f"약 {years_per:.1f}년에 1회"

    return result


# =========================================================
# MARKET-SPECIFIC CASH / REGIME RULES
# =========================================================

# IMPORTANT
# - The KOSPI numbers below are the rules discussed/calibrated in this conversation.
# - S&P500 uses a separate, less-cash-heavy calibration because its volatility/trend
#   characteristics differ. These are transparent starting parameters, not universal truths.
MARKET_RULES = {
    "KOSPI": {
        "label": "KOSPI",
        "bull_slope60": 1.0,
        "bear_slope60": -1.0,
        "bull_below200_max": 5,
        "bear_below200_min": 10,
        "cash_floor": {"BULL": 10.0, "BOX": 15.0, "BEAR": 20.0},
        "e_x": [0, 70, 75, 80, 85, 90, 94, 97, 100],
        "e_cash": [10, 10, 15, 20, 30, 40, 50, 60, 60],
        # Conservative reserve deployment: 1:2:3:4 (cumulative 10/30/60/100%)
        "dd_levels": [-15.0, -20.0, -30.0, -40.0],
        "dd_cumulative": [0.10, 0.30, 0.60, 1.00],
        "rebound_min_pct": 5.0,
        "recovery_remaining_invest_frac": 0.50,
        "recovery_above50_10d_min": 7,
        "reset_days": 40,
        "stabilize_days": 60,
        "cash_step_pct": 2.5,
        "cash_release_days": 21,   # cool-off: release excess cash slowly
        "cash_rebuild_days": 21,   # after crash/box reset: rebuild ammo slowly
    },
    "S&P500": {
        "label": "S&P500",
        # Smoother long-term trend: use a slightly smaller MA200 slope threshold.
        "bull_slope60": 0.75,
        "bear_slope60": -0.75,
        "bull_below200_max": 5,
        "bear_below200_min": 10,
        # Lower structural cash drag than KOSPI; box/bear still keep dry powder.
        "cash_floor": {"BULL": 5.0, "BOX": 10.0, "BEAR": 15.0},
        # E-score is percentile-normalized, but S&P500 requires more extreme E to hold 40~60% cash.
        "e_x": [0, 75, 80, 85, 90, 94, 97, 99, 100],
        "e_cash": [5, 5, 10, 20, 30, 40, 50, 60, 60],
        # Lower-vol market: meaningful drawdowns occur at shallower absolute levels.
        "dd_levels": [-10.0, -15.0, -20.0, -30.0],
        "dd_cumulative": [0.10, 0.30, 0.60, 1.00],
        "rebound_min_pct": 3.5,
        "recovery_remaining_invest_frac": 0.50,
        "recovery_above50_10d_min": 7,
        "reset_days": 30,
        "stabilize_days": 45,
        "cash_step_pct": 2.5,
        # S&P500 cash is released/rebuilt more slowly to avoid frequent tactical churn.
        "cash_release_days": 42,
        "cash_rebuild_days": 42,
    },
}

GENERIC_RULES = MARKET_RULES["KOSPI"]


def rules_for(asset: str):
    return MARKET_RULES.get(asset, GENERIC_RULES)


def quantize_cash(value, step=2.5):
    """Round target cash to 2.5%p increments and clamp to 0~60%."""
    value = float(np.clip(value, 0, 60))
    return round(value / step) * step


def e_cash_target(e_score, asset):
    """Market-specific cash target from the overheat E-score."""
    p = rules_for(asset)
    if pd.isna(e_score):
        return float(p["cash_floor"]["BULL"])
    e = float(np.clip(e_score, 0, 100))
    return quantize_cash(np.interp(e, np.array(p["e_x"], dtype=float), np.array(p["e_cash"], dtype=float)))


def regime_cash_floor(regime, asset):
    p = rules_for(asset)
    return float(p["cash_floor"].get(regime, p["cash_floor"]["BOX"]))


def classify_regime(ma200_slope60_pct, below200_20d, asset):
    """
    Uses MA200 slope + persistence. A 1~3 day MA200 break alone does not create a bear regime.
    """
    p = rules_for(asset)
    if pd.isna(ma200_slope60_pct) or pd.isna(below200_20d):
        return "BOX"
    if ma200_slope60_pct >= p["bull_slope60"] and below200_20d <= p["bull_below200_max"]:
        return "BULL"
    if ma200_slope60_pct <= p["bear_slope60"] and below200_20d >= p["bear_below200_min"]:
        return "BEAR"
    return "BOX"


def regime_label(regime):
    return {
        "BULL": "🟢 상승 추세",
        "BOX": "🟡 박스/중립",
        "BEAR": "🔴 하락 추세",
    }.get(regime, "—")


def drawdown_deployment(cycle_worst_mdd_pct, asset):
    """Return cumulative fraction of the frozen reserve that should already be invested."""
    p = rules_for(asset)
    levels = p["dd_levels"]
    cumulative = p["dd_cumulative"]
    if pd.isna(cycle_worst_mdd_pct):
        return 0.0, "대기", levels[0]

    mdd = float(cycle_worst_mdd_pct)
    for j in range(len(levels) - 1, -1, -1):
        if mdd <= levels[j]:
            next_level = levels[j + 1] if j + 1 < len(levels) else None
            return cumulative[j], f"{levels[j]:g}%: 누적 {cumulative[j]*100:.0f}% 투입", next_level
    return 0.0, "대기", levels[0]


def build_e_score_series(close: pd.Series, asset: str, years=5):
    """Walk-forward signal frame using only information available on each date."""
    dd, sep, ret12 = indicator_series(close, asset)
    df = pd.DataFrame({"price": close, "dd52": dd, "sep": sep, "ret12": ret12}).sort_index()

    ppy = periods_per_year(asset)
    win = max(int(ppy * years), ppy)
    minp = max(int(ppy * min(years, 2)), int(ppy * 0.75))

    rank_sep = df["sep"].rolling(win, min_periods=minp).rank(pct=True)
    rank_ret = df["ret12"].rolling(win, min_periods=minp).rank(pct=True)
    df["e"] = 0.60 * (rank_ret * 100) + 0.40 * (rank_sep * 100)
    df["mdd52_pct"] = df["dd52"] * 100
    df["c"] = df["mdd52_pct"].map(lambda x: c_score_from_mdd(x, asset))

    # Trend/regime inputs. A brief MA200 break does NOT define a bear market.
    df["ma50"] = df["price"].rolling(50, min_periods=30).mean()
    df["ma200"] = df["price"].rolling(200, min_periods=120).mean()
    df["ma50_slope20_pct"] = (df["ma50"] / df["ma50"].shift(20) - 1) * 100
    df["ma200_slope60_pct"] = (df["ma200"] / df["ma200"].shift(60) - 1) * 100
    df["below200"] = (df["price"] < df["ma200"]).astype(float)
    df["below200_20d"] = df["below200"].rolling(20, min_periods=10).sum()
    df["above50"] = (df["price"] > df["ma50"]).astype(float)
    df["above50_10d"] = df["above50"].rolling(10, min_periods=5).sum()
    df["regime"] = [
        classify_regime(s, b, asset)
        for s, b in zip(df["ma200_slope60_pct"], df["below200_20d"])
    ]
    return df


def cash_path_from_signals(df: pd.DataFrame, asset: str):
    """
    Stateful path-dependent engine.

    NORMAL
    - Market-specific E-score target + regime cash floor.
    - Cash can rise immediately when risk/overheat target rises.
    - When target falls, cash is released only 2.5%p at a time (hysteresis), not dumped in one day.
    - After a crash-cycle reset, ammo is rebuilt slowly toward the new target/floor.

    DRAWDOWN
    - KOSPI starts at -15%; S&P500 starts at -10%.
    - Freeze cash on hand as reserve.
    - Use the WORST local-cycle MDD reached (ratchet), so a rebound never restores sold cash.
    - Market-specific 1:2:3:4 deployment thresholds.
    - Recovery signal can invest half of the remaining cash if price/MA50 trend recovers and regime is not BEAR.
    - Old peak can reset after recovery OR prolonged BOX stabilization, even without reclaiming the old high.
    """
    p = rules_for(asset)
    first_dd = p["dd_levels"][0]
    rows = []
    prev_cash = None
    cycle_peak = None
    in_drawdown = False
    reserve_cash = None
    cycle_worst_mdd = 0.0
    trough_price = None
    trough_i = None
    max_since_trough = None
    recovery_applied = False
    rebuilding = False
    last_adjust_i = None

    for i, (dt, r) in enumerate(df.iterrows()):
        price = r.get("price", np.nan)
        e = r.get("e", np.nan)
        regime = r.get("regime", "BOX")
        c = r.get("c", np.nan)
        mdd52_pct = r.get("mdd52_pct", np.nan)
        ma50 = r.get("ma50", np.nan)
        ma50_slope20 = r.get("ma50_slope20_pct", np.nan)
        above50_10d = r.get("above50_10d", np.nan)
        ma200_slope60 = r.get("ma200_slope60_pct", np.nan)
        below200_20d = r.get("below200_20d", np.nan)

        if pd.isna(price):
            continue

        if cycle_peak is None:
            cycle_peak = float(price)

        # Local cycle peak is allowed to advance only outside an active drawdown cycle.
        if not in_drawdown:
            cycle_peak = max(float(cycle_peak), float(price))
        cycle_mdd = (float(price) / float(cycle_peak) - 1.0) * 100

        if not in_drawdown:
            target = max(e_cash_target(e, asset), regime_cash_floor(regime, asset))

            if prev_cash is None:
                cash = target
                last_adjust_i = i
            elif target > float(prev_cash) + 1e-9:
                if rebuilding:
                    # After a crash, do not jump from 0~5% straight back to a 10~20% floor.
                    if last_adjust_i is None or (i - last_adjust_i) >= p["cash_rebuild_days"]:
                        cash = min(float(prev_cash) + p["cash_step_pct"], float(target))
                        last_adjust_i = i
                    else:
                        cash = float(prev_cash)
                else:
                    # Overheat/risk accumulation may build to the target as the signal rises.
                    cash = float(target)
                    last_adjust_i = i
            elif target < float(prev_cash) - 1e-9:
                # E cooling without a true drawdown: release cash slowly, never all at once.
                if last_adjust_i is None or (i - last_adjust_i) >= p["cash_release_days"]:
                    cash = max(float(prev_cash) - p["cash_step_pct"], float(target))
                    last_adjust_i = i
                else:
                    cash = float(prev_cash)
            else:
                cash = float(prev_cash)

            if rebuilding and cash >= target - 1e-9:
                rebuilding = False

            mode = "정상/E·레짐"
            stage_text = f"{regime_label(regime)} · 목표현금 {target:g}%"
            next_level = first_dd
            used_frac = 0.0

            # Enter path-dependent drawdown mode.
            if cycle_mdd <= first_dd:
                in_drawdown = True
                reserve_cash = float(cash)
                cycle_worst_mdd = float(cycle_mdd)
                trough_price = float(price)
                trough_i = i
                max_since_trough = float(price)
                recovery_applied = False
                rebuilding = False

                used_frac, stage_text, next_level = drawdown_deployment(cycle_worst_mdd, asset)
                cash = min(float(cash), float(reserve_cash) * (1 - used_frac))
                mode = "하락장/MDD 투입"
        else:
            # Ratchet: remember the worst local-cycle MDD even after a rebound.
            if cycle_mdd < cycle_worst_mdd:
                cycle_worst_mdd = float(cycle_mdd)
                trough_price = float(price)
                trough_i = i
                max_since_trough = float(price)
            else:
                max_since_trough = max(float(max_since_trough), float(price)) if max_since_trough is not None else float(price)

            used_frac, stage_text, next_level = drawdown_deployment(cycle_worst_mdd, asset)
            stage_cash = float(reserve_cash) * (1 - used_frac)
            cash = min(float(prev_cash) if prev_cash is not None else stage_cash, stage_cash)
            mode = "하락장/MDD 투입"

            rebound_from_trough_pct = (
                (float(price) / float(trough_price) - 1.0) * 100
                if trough_price and trough_price > 0 else 0.0
            )

            recovery_signal = (
                not recovery_applied
                and regime != "BEAR"
                and pd.notna(ma50) and float(price) > float(ma50)
                and pd.notna(ma50_slope20) and float(ma50_slope20) > 0
                and pd.notna(above50_10d) and float(above50_10d) >= p["recovery_above50_10d_min"]
                and rebound_from_trough_pct >= p["rebound_min_pct"]
            )

            if recovery_signal:
                invest_frac = p["recovery_remaining_invest_frac"]
                cash = quantize_cash(float(cash) * (1 - invest_frac))
                recovery_applied = True
                mode = "회복 확인/추가 투입"
                stage_text += f" · 회복신호 → 남은 현금 {invest_frac*100:.0f}% 추가 투입"

            days_since_trough = (i - trough_i) if trough_i is not None else 0
            strong_reset = (
                recovery_applied
                and days_since_trough >= p["reset_days"]
                and regime != "BEAR"
                and pd.notna(ma50) and float(price) > float(ma50)
                and pd.notna(ma50_slope20) and float(ma50_slope20) > 0
            )

            # Explicit box/stabilization reset: old high must not dominate forever.
            box_reset = (
                days_since_trough >= p["stabilize_days"]
                and regime == "BOX"
                and pd.notna(above50_10d) and float(above50_10d) >= 5
                and rebound_from_trough_pct >= p["rebound_min_pct"]
            )

            if strong_reset or box_reset:
                in_drawdown = False
                cycle_peak = max(float(max_since_trough), float(price))
                reserve_cash = None
                cycle_worst_mdd = 0.0
                trough_price = None
                trough_i = None
                max_since_trough = None
                recovery_applied = False
                rebuilding = True
                last_adjust_i = i - p["cash_rebuild_days"]  # allow one small rebuild step now
                mode = "사이클 리셋/현금 재축적"
                why = "회복 추세" if strong_reset else "박스 안정화"
                stage_text = f"{why}로 새 로컬 고점 리셋 · {regime_label(regime)}"
                next_level = first_dd
                used_frac = 0.0

        rows.append({
            "date": dt,
            "cash": float(cash),
            "reserve_cash": reserve_cash,
            "used_frac": used_frac,
            "stage": stage_text,
            "next_level": next_level,
            "c": c,
            "e": e,
            "mdd_pct": mdd52_pct,
            "mdd52_pct": mdd52_pct,
            "cycle_mdd_pct": cycle_mdd,
            "cycle_worst_mdd_pct": cycle_worst_mdd if in_drawdown else np.nan,
            "mode": mode,
            "regime": regime,
            "ma200_slope60_pct": ma200_slope60,
            "below200_20d": below200_20d,
            "ma50_slope20_pct": ma50_slope20,
            "above50_10d": above50_10d,
            "recovery_applied": recovery_applied,
        })
        prev_cash = float(cash)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def standalone_cash_for_asset(close: pd.Series, asset: str, years: int):
    sig = build_e_score_series(close, asset, years)
    path = cash_path_from_signals(sig, asset)
    if path.empty:
        return np.nan, "데이터 없음", {}

    r = path.iloc[-1]
    cash = float(r["cash"])
    reserve = float(r["reserve_cash"]) if pd.notna(r["reserve_cash"]) else np.nan
    used_pct = float(r["used_frac"] * 100) if pd.notna(r["used_frac"]) else 0.0
    next_level = r["next_level"]

    if "하락장" in r["mode"] or "회복" in r["mode"]:
        nxt = f" · 다음 MDD 단계 {next_level:g}%" if pd.notna(next_level) else " · MDD 단계 전액 투입"
        reason = (
            f"{regime_label(r['regime'])} · 사이클 최저 MDD {r['cycle_worst_mdd_pct']:.1f}%"
            f" · 시작현금 {reserve:g}% 중 MDD 누적 {used_pct:.0f}% 단계"
            f" → 현금 {cash:g}%{nxt}"
        )
        if r["mode"] == "회복 확인/추가 투입":
            reason += " · 50일선 회복·상승 신호 반영"
    elif r["mode"] == "사이클 리셋/현금 재축적":
        reason = f"하락 사이클 리셋 · {regime_label(r['regime'])} → 현금 {cash:g}%부터 천천히 탄약 재축적"
    else:
        target = max(e_cash_target(r["e"], asset), regime_cash_floor(r["regime"], asset))
        reason = f"{regime_label(r['regime'])} · E-score {r['e']:.1f} · 목표현금 {target:g}% → 현재 현금 {cash:g}%"

    details = {
        "reserve_cash": reserve,
        "used_pct": used_pct,
        "next_level": next_level,
        "mode": r["mode"],
        "regime": r["regime"],
        "cycle_mdd": float(r["cycle_mdd_pct"]),
        "cycle_worst_mdd": float(r["cycle_worst_mdd_pct"]) if pd.notna(r["cycle_worst_mdd_pct"]) else np.nan,
        "ma200_slope60": float(r["ma200_slope60_pct"]) if pd.notna(r["ma200_slope60_pct"]) else np.nan,
        "below200_20d": float(r["below200_20d"]) if pd.notna(r["below200_20d"]) else np.nan,
        "ma50_slope20": float(r["ma50_slope20_pct"]) if pd.notna(r["ma50_slope20_pct"]) else np.nan,
    }
    return cash, reason, details

def m7_opportunity(m7_df: pd.DataFrame):
    if m7_df.empty:
        return "특별한 M7 기회 없음"

    row = m7_df.sort_values("C-score", ascending=False).iloc[0]
    name = m7_df.sort_values("C-score", ascending=False).index[0]
    c = row["C-score"]

    if c >= 95:
        return f"🔴 {name}: 대폭락 구간 (C {c:.1f})"
    if c >= 90:
        return f"🟠 {name}: 깊은 조정 (C {c:.1f})"
    if c >= 80:
        return f"🟡 {name}: 큰 조정 (C {c:.1f})"
    if c >= 70:
        return f"🟢 {name}: 본격 조정 (C {c:.1f})"
    return "⚪ 현재 M7에는 큰 낙폭 신호 없음"

# =========================================================
# STANDALONE WALK-FORWARD BACKTEST
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def run_standalone_cash_backtest(close: pd.Series, asset: str, percentile_years=5, trading_cost_bps=10):
    """
    Daily walk-forward backtest for one market.

    - C-score = absolute 52-week MDD only.
    - E-score = 60% 12m-return percentile + 40% 50d-separation percentile.
    - KOSPI and S&P500 use separate regime/cash/drawdown parameters.
    - Once the market-specific drawdown trigger is breached, cash on hand is frozen as reserve.
    - The reserve is deployed with market-specific 1:2:3:4 thresholds.
    - Signal at t close is applied to t+1 return.
    """
    sig = build_e_score_series(close, asset, percentile_years).copy()

    # Stocks: business-day calendar. Forward-fill local holidays.
    cal = pd.date_range(sig.index.min(), sig.index.max(), freq="B")
    sig = sig.reindex(cal).ffill()
    sig["ret1"] = sig["price"].pct_change()

    path = cash_path_from_signals(sig, asset)
    if path.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    prev_cash = None
    common = path.index.intersection(sig.index)

    for i in range(len(common) - 1):
        dt = common[i]
        nxt = common[i + 1]
        cash = path.loc[dt, "cash"]
        next_ret = sig.loc[nxt, "ret1"]
        if pd.isna(cash) or pd.isna(next_ret):
            continue

        turnover = 0 if prev_cash is None else abs(cash - prev_cash) / 100
        cost = turnover * (trading_cost_bps / 10000)
        invested = 1 - cash / 100
        strategy_ret = invested * next_ret - cost

        rows.append({
            "date": dt,
            "cash": cash,
            "reserve_cash": path.loc[dt, "reserve_cash"],
            "used_frac": path.loc[dt, "used_frac"],
            "c": path.loc[dt, "c"],
            "e": path.loc[dt, "e"],
            "mdd_pct": path.loc[dt, "mdd_pct"],
            "market_ret_next": next_ret,
            "strategy_ret": strategy_ret,
            "fixed0_ret": next_ret,
            "fixed10_ret": 0.90 * next_ret,
            "fixed20_ret": 0.80 * next_ret,
            "fixed30_ret": 0.70 * next_ret,
            "fixed50_ret": 0.50 * next_ret,
            "turnover": turnover,
            "cost": cost,
        })
        prev_cash = cash

    bt = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if bt.empty:
        return bt, pd.DataFrame()

    def stats(ret):
        ret = ret.dropna()
        if len(ret) < 252:
            return {"CAGR": np.nan, "MDD": np.nan, "Sharpe": np.nan, "Calmar": np.nan}
        wealth = (1 + ret).cumprod()
        years = len(ret) / 252
        cagr = wealth.iloc[-1] ** (1 / years) - 1
        dd_curve = wealth / wealth.cummax() - 1
        mdd = dd_curve.min()
        vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / vol if vol > 0 else np.nan
        calmar = cagr / abs(mdd) if mdd < 0 else np.nan
        return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Calmar": calmar}

    stats_df = pd.DataFrame({
        f"{asset} 새 C/E 현금룰": stats(bt["strategy_ret"]),
        "현금 0% 고정": stats(bt["fixed0_ret"]),
        "현금 10% 고정": stats(bt["fixed10_ret"]),
        "현금 20% 고정": stats(bt["fixed20_ret"]),
        "현금 30% 고정": stats(bt["fixed30_ret"]),
        "현금 50% 고정": stats(bt["fixed50_ret"]),
    }).T
    return bt, stats_df



# =========================================================
# PORTFOLIO OS EXTENSIONS
# =========================================================
import io
import re
from urllib.parse import quote
import requests

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1B9jNFFQW0dCqZUzJMoHb7sNpT-9k02G16A2PwJPUx9A/edit?usp=drivesdk"
DEFAULT_SHEET_NAME = "포트폴리오"

# 사용자가 지금 논의 중인 전술 포지션. 목표에 도달하면 자동으로 추가매수 제안이 사라집니다.
TACTICAL_PLAN = {
    "name": "SOL 반도체전공정",
    "code": "475300",
    "ticker": "475300.KS",
    "target_weight": 3.5,
    "funding": [
        ("SOL AI반도체소부장", 1.5),
        ("에이피알", 0.8),
        ("삼양식품", 0.7),
        ("신세계", 0.5),
    ],
}

# 비주식 자산은 일별 percentile이 아니라 "독립 폭락 이벤트의 재현주기"로 판단합니다.
# 같은 하락이 여러 날 이어져도 하나의 사건으로 묶고, 과거에 비슷하거나 더 심한 사건이
# 평균 몇 년에 한 번 있었는지를 계산합니다. 사용자의 비주식 자산 확신이 낮으므로
# 기본 알림 문턱은 1.5년(약 1~2년에 한 번 꼴)로 보수적으로 설정합니다.
# start_dd는 '폭락 이벤트 시작'을 인식하기 위한 자산별 최소 낙폭입니다.
CRASH_ASSETS = {
    "미국 장기채": {"ticker": "TLT", "group": "채권", "proxy": "TLT", "start_dd": -0.05},
    "금": {"ticker": "GC=F", "group": "원자재", "proxy": "금 선물", "start_dd": -0.075},
    "은": {"ticker": "SI=F", "group": "원자재", "proxy": "은 선물", "start_dd": -0.12},
    "구리": {"ticker": "HG=F", "group": "원자재", "proxy": "구리 선물", "start_dd": -0.12},
    "브렌트유": {"ticker": "BZ=F", "group": "원자재", "proxy": "브렌트 선물", "start_dd": -0.15},
    "BTC": {"ticker": "BTC-USD", "group": "코인", "proxy": "BTC", "start_dd": -0.20},
    "ETH": {"ticker": "ETH-USD", "group": "코인", "proxy": "ETH", "start_dd": -0.25},
}


# 8/16 캡처를 fallback으로 내장. Google Sheet를 못 읽을 때만 사용합니다.
FALLBACK_PORTFOLIO = [
    ("삼성전자", "005930", 349_987_500, 274_500, 1275),
    ("SOL AI반도체소부장", "455850", 129_524_685, 23_955, 5407),
    ("삼양식품", "003230", 63_500_000, 1_270_000, 50),
    ("KB금융", "105560", 60_997_000, 168_500, 362),
    ("메리츠금융지주", "138040", 59_166_900, 116_700, 507),
    ("신세계", "004170", 58_499_000, 427_000, 137),
    ("KT&G", "033780", 58_080_000, 176_000, 330),
    ("한화오션", "042660", 57_671_600, 95_800, 602),
    ("HD현대", "267250", 57_697_500, 235_500, 245),
    ("파마리서치", "214450", 48_556_000, 398_000, 122),
    ("HD현대중공업", "329180", 45_900_000, 510_000, 90),
    ("에이피알", "278470", 42_955_000, 390_500, 110),
    ("삼성전자우", "005935", 41_076_000, 195_600, 210),
    ("한국금융지주", "071050", 29_580_000, 204_000, 145),
    ("오라클", "ORCL", 10_660_353, 150.52, 50),
    ("마이크론", "MU", 8_257_963, 971.66, 6),
    ("블룸에너지", "BE", 8_142_578, 229.94, 25),
    ("루멘텀", "LITE", 7_871_097, 926.14, 6),
    ("코히어런트", "COHR", 6_922_926, 325.83, 15),
    ("VIP펀드", "", 31_850_000, 2171, 0),
    ("예수금", "", 38_420_000, np.nan, 0),
]


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().replace(",", "").replace("₩", "").replace("원", "")
    s = s.replace("%", "")
    if s in {"", "-", "—", "nan", "None", "#N/A", "#REF!", "#DIV/0!"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _clean_code(v):
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        s = s.split(".")[0]
        return s.zfill(6)
    return s.upper()


def fallback_portfolio_df():
    df = pd.DataFrame(FALLBACK_PORTFOLIO, columns=["종목", "코드", "평가금액", "현재가", "수량"])
    total = df["평가금액"].sum()
    df["비중"] = df["평가금액"] / total * 100
    df["데이터원"] = "8/16 fallback"
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_google_portfolio(sheet_url: str, sheet_name: str):
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", sheet_url)
    if not m:
        raise ValueError("Google Sheet URL에서 문서 ID를 찾지 못했습니다.")
    sid = m.group(1)
    csv_url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"
    r = requests.get(csv_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text
    if "accounts.google.com" in text.lower() or "sign in" in text.lower():
        raise PermissionError("시트가 링크 공개 상태가 아니어서 읽을 수 없습니다.")

    raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)
    header_row = None
    for i in range(min(len(raw), 20)):
        vals = [str(x).strip() for x in raw.iloc[i].tolist()]
        if "종목" in vals and "비중" in vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError("'종목'과 '비중' 헤더를 찾지 못했습니다. 시트 구조를 확인하세요.")

    hdr = [str(x).strip() for x in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)

    def idx(label):
        try:
            return hdr.index(label)
        except ValueError:
            return None

    name_i = idx("종목")
    code_i = idx("코드")
    weight_i = idx("비중")
    price_i = idx("현재가")
    qty_i = idx("수량")
    ret_i = idx("수익률")
    value_i = weight_i - 1 if weight_i is not None and weight_i > 0 else None
    if name_i is None or value_i is None:
        raise ValueError("포트폴리오 핵심 열을 찾지 못했습니다.")

    rows = []
    for _, row in body.iterrows():
        name = str(row.iloc[name_i]).strip() if name_i < len(row) else ""
        if not name or name.lower() == "nan":
            continue
        # 시트 하단 보조 계산영역/설명행 제외
        if name.lower() in {"pluto"} or name.startswith("cma+"):
            continue
        val = _num(row.iloc[value_i]) if value_i < len(row) else np.nan
        if pd.isna(val) or val <= 0:
            continue
        code = _clean_code(row.iloc[code_i]) if code_i is not None and code_i < len(row) else ""
        price = _num(row.iloc[price_i]) if price_i is not None and price_i < len(row) else np.nan
        qty = _num(row.iloc[qty_i]) if qty_i is not None and qty_i < len(row) else 0
        ret = _num(row.iloc[ret_i]) if ret_i is not None and ret_i < len(row) else np.nan
        rows.append({"종목": name, "코드": code, "평가금액": val, "현재가": price, "수량": qty, "수익률": ret})

    if not rows:
        raise ValueError("유효한 보유종목 행을 찾지 못했습니다.")
    df = pd.DataFrame(rows)
    total = df["평가금액"].sum()
    df["비중"] = df["평가금액"] / total * 100
    df["데이터원"] = f"Google Sheet/{sheet_name}"
    return df


def load_portfolio_with_fallback(sheet_url, sheet_name):
    try:
        df = load_google_portfolio(sheet_url, sheet_name)
        return df, None
    except Exception as e:
        return fallback_portfolio_df(), str(e)


def classify_holding(name: str, code: str):
    n = str(name).lower().replace(" ", "")
    c = str(code).upper()
    if "예수금" in n or n in {"cash", "현금"}:
        return "현금", "현금"
    if "vip" in n or "펀드" in n:
        return "펀드", "기타"
    if c in {"ORCL", "MU", "BE", "LITE", "COHR"}:
        return "미국 AI", "미국"
    if any(k in n for k in ["삼성전자", "반도체소부장", "반도체전공정"]):
        return "반도체", "한국"
    if any(k in n for k in ["kb금융", "메리츠금융", "한국금융"]):
        return "금융", "한국"
    if any(k in n for k in ["한화오션", "hd현대중공업", "hd현대"]):
        return "조선/산업재", "한국"
    if any(k in n for k in ["삼양식품", "신세계", "kt&g"]):
        return "소비재", "한국"
    if any(k in n for k in ["에이피알", "파마리서치"]):
        return "뷰티/헬스", "한국"
    if re.fullmatch(r"\d{6}", c):
        return "기타 한국주", "한국"
    return "기타", "기타"


def enrich_portfolio(df):
    out = df.copy()
    cats = out.apply(lambda r: classify_holding(r["종목"], r["코드"]), axis=1)
    out["분류"] = [x[0] for x in cats]
    out["지역"] = [x[1] for x in cats]
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_core_market(percentile_years=5):
    series = {}
    rows = []
    for name, ticker in ALL.items():
        try:
            s = fetch_close(ticker)
            series[name] = s
            m = calc_metrics(s, name, percentile_years)
            m["자산"] = name
            rows.append(m)
        except Exception:
            pass
    metrics = pd.DataFrame(rows).set_index("자산") if rows else pd.DataFrame()
    kospi_cash, kospi_reason, kospi_details = (np.nan, "데이터 없음", {})
    sp_cash, sp_reason, sp_details = (np.nan, "데이터 없음", {})
    if "KOSPI" in series:
        kospi_cash, kospi_reason, kospi_details = standalone_cash_for_asset(series["KOSPI"], "KOSPI", percentile_years)
    if "S&P500" in series:
        sp_cash, sp_reason, sp_details = standalone_cash_for_asset(series["S&P500"], "S&P500", percentile_years)
    return series, metrics, kospi_cash, kospi_reason, kospi_details, sp_cash, sp_reason, sp_details


def _event_grade(return_period_years: float) -> str:
    """Human-readable rarity grade. Signal threshold itself is user configurable."""
    if pd.isna(return_period_years):
        return "—"
    if np.isinf(return_period_years) or return_period_years >= 10:
        return "🔥 역사적 극단"
    if return_period_years >= 5:
        return "🔴 매우 드문 폭락"
    if return_period_years >= 2.5:
        return "🟠 큰 기회"
    if return_period_years >= 1.5:
        return "🟡 1~2년급 희귀 하락"
    if return_period_years >= 1.0:
        return "⚪ 관찰"
    return "—"


def _format_return_period(x):
    if pd.isna(x):
        return "—"
    if np.isinf(x):
        return "관측기간 내 없음"
    return f"{float(x):.1f}년"


def drawdown_event_metrics(close: pd.Series, start_dd: float, recovery_days: int = 15):
    """
    Treat one crash as one event.

    - Drawdown is measured versus the rolling 52-week high.
    - An event begins when drawdown falls through the asset-specific start_dd.
    - The event ends only after drawdown recovers above half of start_dd for
      recovery_days consecutive observations. This hysteresis prevents one crash
      from being counted over and over on consecutive days.
    - Historical recurrence = observation years / number of CLOSED historical
      events whose trough was at least as deep as today's drawdown.

    The current ongoing event is never included in the historical denominator.
    """
    s = close.dropna().astype(float).sort_index()
    if len(s) < 365 * 3:
        return {
            "현재 MDD": np.nan, "일별 폭락 percentile": np.nan, "ATH MDD": np.nan,
            "현재 이벤트": False, "이벤트 시작일": "—", "현재 이벤트 최저 MDD": np.nan,
            "과거 유사이상 이벤트": np.nan, "관측기간(년)": np.nan,
            "평균 재현주기(년)": np.nan, "희귀도": "—", "기준일": "—",
        }

    high = s.rolling("365D", min_periods=60).max()
    dd = (s / high - 1).dropna()
    if dd.empty:
        return {}

    # Recovery threshold is deliberately much shallower than the event-start level.
    # Examples: BTC -20% starts an event; it closes after staying above -10% for 15 observations.
    recover_dd = float(start_dd) * 0.50
    in_event = False
    start_date = None
    trough = np.nan
    trough_date = None
    recovery_count = 0
    events = []

    for dt, val in dd.items():
        val = float(val)
        if not in_event:
            if val <= start_dd:
                in_event = True
                start_date = dt
                trough = val
                trough_date = dt
                recovery_count = 0
            continue

        if val < trough:
            trough = val
            trough_date = dt

        if val >= recover_dd:
            recovery_count += 1
            if recovery_count >= recovery_days:
                events.append({
                    "start": start_date,
                    "trough_date": trough_date,
                    "trough_dd": float(trough),
                    "end": dt,
                })
                in_event = False
                start_date = None
                trough = np.nan
                trough_date = None
                recovery_count = 0
        else:
            recovery_count = 0

    cur = float(dd.iloc[-1])
    daily_pct = float((dd >= cur).mean() * 100)
    ath_dd = float((s / s.cummax() - 1).iloc[-1])
    years = max((dd.index[-1] - dd.index[0]).days / 365.25, 0.01)

    # Compare today's depth with closed historical event troughs only.
    historical_depths = np.array([e["trough_dd"] for e in events], dtype=float)
    if historical_depths.size:
        similar_or_worse = int(np.sum(historical_depths <= cur))
    else:
        similar_or_worse = 0

    if similar_or_worse == 0:
        return_period = np.inf if cur <= start_dd else np.nan
    else:
        return_period = years / similar_or_worse

    active = bool(in_event and cur <= recover_dd)
    # If the event has materially recovered, do not keep shouting even before the
    # 15-day closure confirmation is complete.
    if not active:
        return_period_for_signal = np.nan
    else:
        return_period_for_signal = return_period

    return {
        "현재 MDD": cur * 100,
        "일별 폭락 percentile": daily_pct,
        "ATH MDD": ath_dd * 100,
        "현재 이벤트": active,
        "이벤트 시작일": str(pd.Timestamp(start_date).date()) if in_event and start_date is not None else "—",
        "현재 이벤트 최저 MDD": float(trough) * 100 if in_event and pd.notna(trough) else np.nan,
        "과거 유사이상 이벤트": similar_or_worse,
        "관측기간(년)": years,
        "평균 재현주기(년)": return_period_for_signal,
        "희귀도": _event_grade(return_period_for_signal),
        "기준일": str(dd.index[-1].date()),
    }


def suggested_cross_asset_weight(return_period_years: float, each_cap: float) -> float:
    """Very conservative sizing because these are outside the user's core stock expertise."""
    if pd.isna(return_period_years):
        return 0.0
    if np.isinf(return_period_years) or return_period_years >= 8:
        raw = 1.5
    elif return_period_years >= 4:
        raw = 1.0
    elif return_period_years >= 1.5:
        raw = 0.5
    else:
        raw = 0.0
    return float(min(raw, each_cap))


@st.cache_data(ttl=1800, show_spinner=False)
def crash_radar_table():
    rows = []
    for name, meta in CRASH_ASSETS.items():
        try:
            s = fetch_close(meta["ticker"])
            x = drawdown_event_metrics(s, meta["start_dd"])
            x.update({
                "자산": name,
                "그룹": meta["group"],
                "신호 프록시": meta["proxy"],
                "이벤트 시작 기준": meta["start_dd"] * 100,
            })
            rows.append(x)
        except Exception:
            rows.append({
                "자산": name, "그룹": meta["group"], "신호 프록시": meta["proxy"],
                "이벤트 시작 기준": meta["start_dd"] * 100,
                "현재 MDD": np.nan, "일별 폭락 percentile": np.nan, "ATH MDD": np.nan,
                "현재 이벤트": False, "이벤트 시작일": "—", "현재 이벤트 최저 MDD": np.nan,
                "과거 유사이상 이벤트": np.nan, "관측기간(년)": np.nan,
                "평균 재현주기(년)": np.nan, "희귀도": "—", "기준일": "—",
            })
    return pd.DataFrame(rows).set_index("자산")


@st.cache_data(ttl=900, show_spinner=False)
def latest_price(ticker: str):
    s = fetch_close(ticker)
    return float(s.iloc[-1]), str(s.index[-1].date())


def portfolio_cash_target(portfolio, kospi_cash, sp_cash):
    kr = portfolio.loc[portfolio["지역"] == "한국", "평가금액"].sum()
    us = portfolio.loc[portfolio["지역"] == "미국", "평가금액"].sum()
    risk = kr + us
    if risk <= 0:
        return np.nan
    parts = 0.0
    denom = 0.0
    if pd.notna(kospi_cash) and kr > 0:
        parts += kr * kospi_cash
        denom += kr
    if pd.notna(sp_cash) and us > 0:
        parts += us * sp_cash
        denom += us
    return parts / denom if denom > 0 else np.nan


def find_row_by_name(df, wanted):
    wanted_key = wanted.lower().replace(" ", "")
    mask = df["종목"].astype(str).str.lower().str.replace(" ", "", regex=False).str.contains(wanted_key, regex=False)
    if mask.any():
        return df.loc[mask].iloc[0]
    return None


def tactical_orders(portfolio, total_value, target_price, min_trade_krw=5_000_000):
    target_code = TACTICAL_PLAN["code"]
    target_mask = (portfolio["코드"].astype(str) == target_code) | portfolio["종목"].astype(str).str.contains("반도체전공정", na=False)
    current_value = float(portfolio.loc[target_mask, "평가금액"].sum())
    current_weight = current_value / total_value * 100 if total_value > 0 else 0
    gap_pp = max(0.0, TACTICAL_PLAN["target_weight"] - current_weight)
    desired = total_value * gap_pp / 100
    if desired < min_trade_krw:
        return pd.DataFrame(), current_weight, desired

    funding_total_pp = sum(x[1] for x in TACTICAL_PLAN["funding"])
    orders = []
    proceeds = 0.0
    for name, cap_pp in TACTICAL_PLAN["funding"]:
        r = find_row_by_name(portfolio, name)
        if r is None:
            continue
        desired_sell = desired * cap_pp / funding_total_pp
        price = float(r["현재가"]) if pd.notna(r["현재가"]) and r["현재가"] > 0 else np.nan
        held_qty = int(r["수량"]) if pd.notna(r["수량"]) else 0
        if pd.notna(price) and held_qty > 0:
            shares = int(min(held_qty, np.floor(desired_sell / price)))
            amount = shares * price
        else:
            shares = 0
            amount = min(desired_sell, float(r["평가금액"]))
        if amount >= min_trade_krw / 4:
            proceeds += amount
            orders.append({"구분": "매도", "종목": r["종목"], "수량": shares if shares > 0 else "—", "예상금액": amount})

    if pd.notna(target_price) and target_price > 0 and proceeds > 0:
        buy_shares = int(np.floor(proceeds / target_price))
        buy_amount = buy_shares * target_price
    else:
        buy_shares = "—"
        buy_amount = proceeds
    if proceeds > 0:
        orders.append({"구분": "매수", "종목": TACTICAL_PLAN["name"], "수량": buy_shares, "예상금액": buy_amount})
    return pd.DataFrame(orders), current_weight, desired


def build_daily_advice(portfolio, kospi_cash, sp_cash, crash_df, min_return_period_years, cross_asset_each_pct, cross_asset_max_pct):
    total = float(portfolio["평가금액"].sum())
    cash_value = float(portfolio.loc[portfolio["분류"] == "현금", "평가금액"].sum())
    cash_pct = cash_value / total * 100 if total else np.nan
    target_cash = portfolio_cash_target(portfolio, kospi_cash, sp_cash)
    messages = []

    if pd.notna(target_cash) and pd.notna(cash_pct):
        gap = target_cash - cash_pct
        if gap >= 2.5:
            messages.append(("현금", "🔴", f"시장 엔진 기준 혼합 현금이 약 {target_cash:.1f}%인데 실제 예수금은 {cash_pct:.1f}%입니다. 약 {gap:.1f}%p 현금 확보가 우선입니다."))
        elif gap <= -2.5:
            messages.append(("현금", "🟢", f"시장 엔진 기준 혼합 현금은 약 {target_cash:.1f}%이고 실제 예수금은 {cash_pct:.1f}%입니다. 현금이 약 {-gap:.1f}%p 더 많아 신규매수 여력이 있습니다."))
        else:
            messages.append(("현금", "⚪", f"실제 예수금 {cash_pct:.1f}%는 시장 엔진 혼합 현금 {target_cash:.1f}%와 큰 차이가 없습니다."))

    # 집중도: 사용자가 반도체 고비중을 의도적으로 운용하므로 '강제 매도'가 아니라 경고만.
    single = portfolio[~portfolio["분류"].isin(["현금", "펀드"])].sort_values("비중", ascending=False).head(1)
    if not single.empty and float(single.iloc[0]["비중"]) >= 30:
        r = single.iloc[0]
        messages.append(("집중도", "🟡", f"{r['종목']} 단일종목 비중이 {r['비중']:.1f}%로 30%를 넘었습니다. 신규매수보다 다른 기회를 우선하는 편이 낫습니다."))

    semi_names = ["삼성전자", "삼성전자우", "SOL AI반도체소부장", "SOL 반도체전공정", "마이크론"]
    semi_mask = portfolio["종목"].astype(str).apply(lambda x: any(k.lower().replace(" ", "") in x.lower().replace(" ", "") for k in semi_names))
    semi_pct = portfolio.loc[semi_mask, "평가금액"].sum() / total * 100 if total else 0
    if semi_pct >= 50:
        messages.append(("집중도", "🔴", f"반도체 직접 노출이 약 {semi_pct:.1f}%입니다. 전공정 추가보다 반도체 총량 관리가 우선입니다."))
    elif semi_pct >= 45:
        messages.append(("집중도", "🟡", f"반도체 직접 노출이 약 {semi_pct:.1f}%입니다. 추가 편입은 50% 상한을 넘기지 않는 범위가 좋습니다."))

    if crash_df is not None and not crash_df.empty:
        rp = pd.to_numeric(crash_df["평균 재현주기(년)"], errors="coerce")
        active = crash_df["현재 이벤트"].fillna(False).astype(bool)
        triggered = crash_df[active & (rp >= min_return_period_years)].copy()
    else:
        triggered = pd.DataFrame()

    if not triggered.empty:
        allocs = []
        for _, r in triggered.iterrows():
            allocs.append(suggested_cross_asset_weight(r["평균 재현주기(년)"], cross_asset_each_pct))
        total_alloc = min(cross_asset_max_pct, sum(allocs))
        assets = ", ".join(triggered.index.tolist())
        messages.append(("폭락자산", "🔥", f"{assets}에서 독립 폭락 이벤트가 발생했습니다. 과거 재현주기가 최소 {min_return_period_years:.1f}년 이상인 구간만 잡은 신호이며, 비주식 자산은 합계 {total_alloc:.1f}%p 이내의 소액 분산만 검토합니다."))
    else:
        messages.append(("폭락자산", "⚪", f"채권·원자재·코인 중 평균 {min_return_period_years:.1f}년에 한 번 이하로 드문 독립 폭락 이벤트는 없습니다. → 아무것도 하지 않음"))
    return messages, target_cash, cash_pct, triggered


def money(x):
    if pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 100_000_000:
        return f"{x/100_000_000:.2f}억"
    return f"{x/10_000:.0f}만"


def render_portfolio_page(portfolio, market_pack, crash_df, settings):
    series, metrics, kospi_cash, kospi_reason, kospi_details, sp_cash, sp_reason, sp_details = market_pack
    total = float(portfolio["평가금액"].sum())
    advice, blend_cash, cash_pct, triggered = build_daily_advice(
        portfolio, kospi_cash, sp_cash, crash_df,
        settings["min_return_period_years"], settings["cross_asset_each_pct"], settings["cross_asset_max_pct"]
    )

    st.title("🧭 Portfolio OS")
    st.caption("Google Sheet의 최신 보유내역 + 기존 Market Distress Radar를 한 화면에서 연결합니다. 대부분의 날에는 '아무것도 안 함'이 정상입니다.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 포트", money(total))
    c2.metric("예수금", f"{cash_pct:.1f}%" if pd.notna(cash_pct) else "—")
    c3.metric("KOSPI 추천 현금", f"{kospi_cash:g}%" if pd.notna(kospi_cash) else "—")
    c4.metric("S&P500 추천 현금", f"{sp_cash:g}%" if pd.notna(sp_cash) else "—")

    st.subheader("오늘의 조언")
    for _, icon, msg in advice:
        if icon in {"🔴", "🔥"}:
            st.warning(f"{icon} {msg}")
        elif icon == "🟢":
            st.success(f"{icon} {msg}")
        else:
            st.info(f"{icon} {msg}")

    st.subheader("전술 리밸런싱 — SOL 반도체전공정 목표 3.5%")
    try:
        target_price, target_date = latest_price(TACTICAL_PLAN["ticker"])
    except Exception:
        target_price, target_date = np.nan, "—"
    orders, current_tw, desired = tactical_orders(portfolio, total, target_price, settings["min_trade_krw"])
    if orders.empty:
        st.success(f"현재 전공정 비중 {current_tw:.2f}% — 목표 3.5%와의 차이가 최소 거래기준보다 작아 추가 주문 없음.")
    else:
        st.caption(f"현재 전공정 {current_tw:.2f}% → 목표 3.5% · 목표 ETF 최근가 {target_price:,.0f}원 ({target_date})" if pd.notna(target_price) else f"현재 전공정 {current_tw:.2f}% → 목표 3.5%")
        show_orders = orders.copy()
        st.dataframe(show_orders.style.format({"예상금액": "{:,.0f}원"}), use_container_width=True, hide_index=True)
        st.caption("매일 시트 비중을 다시 읽어 목표에 가까워질수록 주문 제안이 자동으로 줄어듭니다. 최소 거래금액 이하이면 아무 거래도 제안하지 않습니다.")

    st.subheader("현재 포트")
    pshow = portfolio[["종목", "코드", "평가금액", "비중", "수량", "분류", "지역"]].sort_values("평가금액", ascending=False).copy()
    st.dataframe(pshow.style.format({"평가금액": "{:,.0f}원", "비중": "{:.2f}%", "수량": "{:,.0f}"}, na_rep="—"), use_container_width=True, hide_index=True)

    st.subheader("분류별 비중")
    cat = portfolio.groupby("분류", as_index=False)["평가금액"].sum()
    cat["비중"] = cat["평가금액"] / total * 100
    cat = cat.sort_values("비중", ascending=False)
    st.bar_chart(cat.set_index("분류")["비중"])
    st.dataframe(cat.style.format({"평가금액": "{:,.0f}원", "비중": "{:.1f}%"}), use_container_width=True, hide_index=True)

    if not triggered.empty:
        st.subheader("🔥 지금만 뜨는 비주식 폭락 기회")
        show = triggered[["그룹", "현재 MDD", "현재 이벤트 최저 MDD", "평균 재현주기(년)", "과거 유사이상 이벤트", "관측기간(년)", "희귀도", "신호 프록시", "이벤트 시작일"]].copy()
        show["권고 비중"] = [suggested_cross_asset_weight(x, settings["cross_asset_each_pct"]) for x in show["평균 재현주기(년)"]]
        st.dataframe(
            show.style.format({
                "현재 MDD": "{:.1f}%",
                "현재 이벤트 최저 MDD": "{:.1f}%",
                "평균 재현주기(년)": lambda x: _format_return_period(x),
                "관측기간(년)": "{:.1f}",
                "권고 비중": "{:.1f}%p",
            }, na_rep="—"),
            use_container_width=True,
        )
        st.caption(
            f"같은 폭락이 며칠 이어져도 하나의 이벤트로 묶습니다. 기본 알림은 평균 {settings['min_return_period_years']:.1f}년에 한 번 이하로 드문 사건부터. "
            f"비주식 전체 신규배분은 최대 {settings['cross_asset_max_pct']:.1f}%p까지만 허용합니다."
        )



def render_market_page(market_pack, percentile_years, freq_years):
    series, metrics, kospi_cash, kospi_reason, kospi_details, sp_cash, sp_reason, sp_details = market_pack
    st.title("📉 Market Distress Radar")
    st.caption("기존 v8 로직을 그대로 유지했습니다. KOSPI/S&P500 현금 엔진과 BTC·GOLD·KOSDAQ·M7 저가 판정은 서로 분리됩니다.")
    if metrics.empty:
        st.error("시장 데이터를 가져오지 못했습니다.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("### 🇰🇷 KOSPI")
        st.metric("추천 현금", f"{kospi_cash:g}%" if pd.notna(kospi_cash) else "—")
        st.info(kospi_reason)
    with right:
        st.markdown("### 🇺🇸 S&P500")
        st.metric("추천 현금", f"{sp_cash:g}%" if pd.notna(sp_cash) else "—")
        st.info(sp_reason)

    st.subheader("KOSPI / S&P500 현금 엔진 참고 신호")
    names = [x for x in ["KOSPI", "S&P500"] if x in metrics.index]
    market_show = metrics.loc[names, ["현재가", "52주 MDD", "50일 이격", "12개월 수익률", "C-score", "E-score", "판정", "기준일"]].copy()
    market_show = market_show.rename(columns={"판정": "낙폭 판정"})
    regime_map = {
        "KOSPI": regime_label(kospi_details.get("regime")) if kospi_details else "—",
        "S&P500": regime_label(sp_details.get("regime")) if sp_details else "—",
    }
    market_show["시장 상태"] = [regime_map.get(x, "—") for x in market_show.index]
    st.dataframe(market_show.style.format({"현재가": "{:,.2f}", "52주 MDD": "{:.1f}%", "50일 이격": "{:.1f}%", "12개월 수익률": "{:.1f}%", "C-score": "{:.1f}", "E-score": "{:.1f}"}, na_rep="—"), use_container_width=True)

    st.subheader("BTC · GOLD · KOSDAQ · M7 — 역사적 MDD 저가 판정")
    aux_rows = []
    for name in AUX_ASSETS:
        if name in series:
            x = all_history_drawdown_metrics(series[name])
            if x:
                x["자산"] = name
                aux_rows.append(x)
    aux_df = pd.DataFrame(aux_rows).set_index("자산") if aux_rows else pd.DataFrame()
    if not aux_df.empty:
        show = aux_df[["현재가", "ATH 대비 현재 MDD", "역사적 최대 MDD", "역사적 바닥 대비 괴리", "판정", "역사적 최대 MDD 날짜", "기준일"]].sort_values("역사적 바닥 대비 괴리")
        st.dataframe(show.style.format({"현재가": "{:,.2f}", "ATH 대비 현재 MDD": "{:.1f}%", "역사적 최대 MDD": "{:.1f}%", "역사적 바닥 대비 괴리": "+{:.1f}%"}, na_rep="—"), use_container_width=True)

    st.subheader("MDD 발생확률")
    freq_rows = []
    for name in list(MACRO.keys()) + list(M7.keys()):
        if name in series:
            freq_rows.append({"자산": name, **mdd_day_frequency(series[name], freq_years)})
    if freq_rows:
        freq_df = pd.DataFrame(freq_rows).set_index("자산")
        st.dataframe(freq_df.style.format("{:.2f}%"), use_container_width=True)

    st.subheader("MDD 진입빈도")
    entry_rows = []
    for name in list(MACRO.keys()) + list(M7.keys()):
        if name in series:
            entry_rows.append({"자산": name, **mdd_entry_frequency(series[name], freq_years)})
    if entry_rows:
        st.dataframe(pd.DataFrame(entry_rows).set_index("자산"), use_container_width=True)

    with st.expander("Walk-forward 백테스트 보기", expanded=False):
        for asset in ["KOSPI", "S&P500"]:
            if asset not in series:
                continue
            st.markdown(f"### {asset}")
            bt, stats = run_standalone_cash_backtest(series[asset], asset, percentile_years=percentile_years, trading_cost_bps=10)
            if not stats.empty:
                shown = stats.copy(); shown["CAGR"] *= 100; shown["MDD"] *= 100
                st.dataframe(shown.style.format({"CAGR": "{:.1f}%", "MDD": "{:.1f}%", "Sharpe": "{:.2f}", "Calmar": "{:.2f}"}), use_container_width=True)
                if not bt.empty:
                    st.line_chart(bt["cash"].tail(252))


def render_crash_page(crash_df, settings):
    st.title("🚨 비주식 폭락 이벤트 레이더")
    st.caption(
        "채권·원자재·코인은 일별 99 percentile 대신 독립적인 폭락 사건으로 묶습니다. "
        "현재 낙폭과 같거나 더 심했던 과거 사건이 평균 몇 년에 한 번 있었는지를 계산하고, "
        f"{settings['min_return_period_years']:.1f}년에 한 번 이하로 드문 사건부터만 🔔 신호를 냅니다."
    )

    df = crash_df.copy()
    rp = pd.to_numeric(df["평균 재현주기(년)"], errors="coerce")
    active = df["현재 이벤트"].fillna(False).astype(bool)
    df["신호"] = np.where(active & (rp >= settings["min_return_period_years"]), "🔔 검토", "—")
    # Infinity should sort first; NaN last.
    df["_sort"] = rp.replace(np.inf, 1e9)
    df = df.sort_values(["신호", "_sort"], ascending=[True, False]).drop(columns="_sort")

    show_cols = [
        "그룹", "신호 프록시", "현재 MDD", "현재 이벤트 최저 MDD", "일별 폭락 percentile",
        "과거 유사이상 이벤트", "관측기간(년)", "평균 재현주기(년)", "희귀도", "신호", "이벤트 시작일", "기준일"
    ]
    st.dataframe(
        df[show_cols].style.format({
            "현재 MDD": "{:.1f}%",
            "현재 이벤트 최저 MDD": "{:.1f}%",
            "일별 폭락 percentile": "{:.1f}",
            "관측기간(년)": "{:.1f}",
            "평균 재현주기(년)": lambda x: _format_return_period(x),
        }, na_rep="—"),
        use_container_width=True,
    )

    triggered = df[active & (rp >= settings["min_return_period_years"])]
    if triggered.empty:
        st.info(
            f"현재 평균 {settings['min_return_period_years']:.1f}년에 한 번 이하로 드문 비주식 폭락 이벤트가 없습니다. → 아무것도 하지 않음"
        )
    else:
        st.warning("신호가 켜져도 비주식 자산은 작은 비중만 허용합니다. 희귀할수록 비중을 조금 늘리되 한 자산 상한과 전체 상한을 지킵니다.")
        total_suggested = 0.0
        for name, r in triggered.iterrows():
            suggested = suggested_cross_asset_weight(r["평균 재현주기(년)"], settings["cross_asset_each_pct"])
            total_suggested += suggested
            rp_txt = _format_return_period(r["평균 재현주기(년)"])
            st.markdown(
                f"**{name}** · 현재 MDD {r['현재 MDD']:.1f}% · 과거 유사/이상 사건 {int(r['과거 유사이상 이벤트'])}회 / {r['관측기간(년)']:.1f}년 "
                f"→ **평균 {rp_txt}에 한 번** · {r['희귀도']} · **총자산 {suggested:.1f}%p 검토**"
            )
        st.caption(
            f"동시 신호 합산 제안 {min(total_suggested, settings['cross_asset_max_pct']):.1f}%p, 전체 상한 {settings['cross_asset_max_pct']:.1f}%p. "
            "원자재는 선물가격을 신호로만 사용하며 PTP/세금/롤오버 때문에 실제 상품은 자동 지정하지 않습니다."
        )

    st.markdown("""
**이벤트 계산 방식**  
- 자산별 최소 낙폭을 넘으면 폭락 이벤트 시작  
- 이후 충분히 회복한 상태가 15거래일(코인은 15관측일) 이어져야 이벤트 종료  
- 같은 하락장에서 며칠씩 99 percentile이 반복되어도 **한 사건으로만 계산**  
- 현재 낙폭과 같거나 더 깊었던 **종료된 과거 사건 수**로 평균 재현주기를 계산  
- 현재 진행 중인 사건은 과거 횟수에 포함하지 않음

`일별 폭락 percentile`은 참고용으로만 남겨두고 **알림 여부에는 사용하지 않습니다.**
""")


def render_settings_page(settings):
    st.title("⚙️ 설정 / 규칙")
    st.markdown(f"""
### 포트 데이터
- Google Sheet: `{settings['sheet_url']}`
- 탭 이름: `{settings['sheet_name']}`
- 5분 캐시 후 자동 재조회

### 비주식 폭락 이벤트 규칙
- 알림 문턱: **평균 {settings['min_return_period_years']:.1f}년에 한 번 이하로 드문 사건부터**
- 같은 하락이 여러 날 이어져도 **1개의 이벤트로 묶음**
- 이벤트 종료: 충분한 회복 상태가 **15관측일 연속** 확인될 때
- 일별 MDD percentile은 참고용이며 **알림 조건에는 사용하지 않음**
- 1.5~4년급: 기본 **0.5%p** 검토
- 4~8년급: 기본 **1.0%p** 검토
- 8년+ 또는 관측기간 내 전례 없음: 기본 **1.5%p** 검토
- 한 자산 최대 **{settings['cross_asset_each_pct']:.1f}%p**
- 동시 신호 전체 최대 **{settings['cross_asset_max_pct']:.1f}%p**
- 원자재는 선물가격을 신호로만 사용하고 실제 매수상품 자동 추천은 하지 않음

### 포트 리밸런싱 규칙
- 최소 거래금액: **{money(settings['min_trade_krw'])}**
- SOL 반도체전공정 전술 목표: **3.5%**
- 목표 조달원: SOL AI소부장 1.5%p / APR 0.8%p / 삼양식품 0.7%p / 신세계 0.5%p
- 반도체 총 노출 45%부터 경고, 50%부터 추가편입 제한 경고

### 기존 Market Distress Radar
- KOSPI와 S&P500 현금 엔진은 v8 로직 그대로
- C-score = 절대 MDD
- E-score = 12개월 수익률 percentile 60% + 50일 이격 percentile 40%
- BTC/GOLD/KOSDAQ/M7 저가판정은 현금비중에 영향 없음
""")



# =========================================================
# UI — LEFT NAVIGATION
# =========================================================
with st.sidebar:
    st.title("🧭 Portfolio OS")
    page = st.radio("메뉴", ["🏠 내 포트", "📉 시장 레이더", "🚨 폭락 자산", "⚙️ 설정/설명"], label_visibility="collapsed")
    st.divider()
    percentile_years = st.selectbox("E-score 비교기간", [3, 5, 10], index=1, format_func=lambda x: f"최근 {x}년")
    freq_choice = st.selectbox("MDD 빈도표", ["최근 5년", "최근 10년", "전체 가용기간"], index=2)
    freq_years = {"최근 5년": 5, "최근 10년": 10, "전체 가용기간": None}[freq_choice]
    st.divider()
    sheet_url = st.text_input("포트 Google Sheet", value=DEFAULT_SHEET_URL)
    sheet_name = st.text_input("포트 탭 이름", value=DEFAULT_SHEET_NAME)
    min_return_period_years = st.slider("비주식 알림 최소 재현주기", 1.0, 10.0, 1.5, 0.5, format="%.1f년")
    cross_asset_each_pct = st.slider("비주식 1자산 최대 비중", 0.5, 2.0, 1.5, 0.5)
    cross_asset_max_pct = st.slider("비주식 신호 전체 최대 비중", 1.0, 6.0, 3.0, 0.5)
    min_trade_krw = st.select_slider("최소 리밸런싱 금액", options=[1_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000], value=5_000_000, format_func=lambda x: f"{x/10_000:.0f}만원")
    if st.button("🔄 모든 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

settings = {
    "sheet_url": sheet_url,
    "sheet_name": sheet_name,
    "min_return_period_years": min_return_period_years,
    "cross_asset_each_pct": cross_asset_each_pct,
    "cross_asset_max_pct": cross_asset_max_pct,
    "min_trade_krw": min_trade_krw,
}

portfolio_raw, portfolio_error = load_portfolio_with_fallback(sheet_url, sheet_name)
portfolio = enrich_portfolio(portfolio_raw)
market_pack = load_core_market(percentile_years)
crash_df = crash_radar_table()

if portfolio_error:
    st.warning(f"Google Sheet 자동 읽기 실패 → 8/16 fallback 포트를 사용 중입니다. 원인: {portfolio_error}")
else:
    st.caption(f"✅ 포트 데이터: {portfolio['데이터원'].iloc[0]} · 최신 조회")

if page == "🏠 내 포트":
    render_portfolio_page(portfolio, market_pack, crash_df, settings)
elif page == "📉 시장 레이더":
    render_market_page(market_pack, percentile_years, freq_years)
elif page == "🚨 폭락 자산":
    render_crash_page(crash_df, settings)
else:
    render_settings_page(settings)

st.divider()
st.caption("투자 의사결정 보조용 개인 대시보드입니다. 가격 데이터 오류·지연, Google Sheet 접근권한, 세금·슬리피지·환율·상품구조를 실제 주문 전 별도로 확인하세요.")
