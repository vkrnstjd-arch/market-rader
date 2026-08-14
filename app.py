
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Market Distress Radar", page_icon="📉", layout="wide")

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
# UI
# =========================================================
st.title("📉 Market Distress Radar")
st.caption(
    "KOSPI와 S&P500은 서로 다른 상태·경로형 현금 엔진을 사용합니다. "
    "BTC·GOLD·KOSDAQ·M7은 현금비중에 영향을 주지 않고, 각 자산의 전체 가용 역사에서 ATH 대비 최대 MDD와 현재 MDD를 비교해 저가 구간만 판정합니다."
)

with st.sidebar:
    st.header("설정")

    percentile_years = st.selectbox(
        "E-score percentile 비교기간",
        [3, 5, 10],
        index=1,
        format_func=lambda x: f"최근 {x}년"
    )

    freq_choice = st.selectbox(
        "MDD 빈도표 비교기간",
        ["최근 5년", "최근 10년", "전체 가용기간"],
        index=2
    )

    freq_years = {
        "최근 5년": 5,
        "최근 10년": 10,
        "전체 가용기간": None
    }[freq_choice]

    st.divider()
    st.markdown("""
**절대 MDD 기반 C-score (아래는 KOSPI 기준)**

- **-10% → C 70**: 본격 조정
- **-15% → C 80**: 큰 조정
- **-20% → C 90**: 깊은 조정
- **-25% → C 95**: 대폭락
- **-30% → C 97.5**: 매우 큰 폭락
- **-35% → C 99**: 위기 수준
- **-40% 이하 → C 100**: 역사적 위기

C-score에는 **50일 이격도나 추정 밸류에이션을 넣지 않습니다.**  
S&P500은 별도 기준: **-7.5%=70 / -10%=80 / -15%=90 / -20%=95 / -25%=98 / -30%=100**.
""")

    st.markdown("""
**BTC · GOLD · KOSDAQ · M7 보수적 저가 판정**

현재 ATH 대비 MDD를 **전체 가용 역사상 최대 MDD의 실제 바닥가격**과 비교합니다.
MDD %p 차이가 아니라 가격 차이입니다.

- 역사적 최악 바닥 대비 **+5% 이내 → 🔥 극단적 매수**
- **+10% 이내 → 매우 싸다**
- **+20% 이내 → 싸다**
- **+35% 이내 → 관심**
- 그 이상 → 보통

예: 역사적 최대 MDD -50%, 현재 -45%라면 바닥 50 대비 현재 55이므로 **+10%**입니다.
""")

    st.caption("GOLD는 Yahoo Finance의 금 선물(GC=F)을 가격 프록시로 사용합니다.")

    st.caption("KOSPI와 S&P500은 독립 계좌이며 세부 파라미터도 다릅니다. KOSPI는 -15%부터, S&P500은 -10%부터 하락 모드가 발동합니다. 현재 MDD가 아니라 사이클 최저 MDD를 기억하고, 박스권·회복 시에는 옛 전고점을 자동으로 리셋할 수 있습니다.")


if st.button("🔄 최신 데이터 다시 받기"):
    st.cache_data.clear()
    st.rerun()


series = {}
rows = []

with st.spinner("시장 데이터를 불러오고 계산 중..."):
    for name, ticker in ALL.items():
        try:
            s = fetch_close(ticker)
            series[name] = s
            m = calc_metrics(s, name, percentile_years)
            m["자산"] = name
            rows.append(m)
        except Exception as e:
            st.warning(f"{name} 데이터를 불러오지 못했습니다: {e}")

if not rows:
    st.error("시장 데이터를 가져오지 못했습니다. 잠시 뒤 다시 시도하세요.")
    st.stop()

metrics = pd.DataFrame(rows).set_index("자산")
macro_names = [x for x in MACRO if x in metrics.index]
m7_names = [x for x in M7 if x in metrics.index]

macro_df = metrics.loc[macro_names].copy()
m7_df = metrics.loc[m7_names].copy()

aux_rows = []
for name in AUX_ASSETS:
    if name in series:
        x = all_history_drawdown_metrics(series[name])
        if x:
            x["자산"] = name
            aux_rows.append(x)
aux_df = pd.DataFrame(aux_rows).set_index("자산") if aux_rows else pd.DataFrame()


# SUMMARY — TWO INDEPENDENT CASH SLEEVES
kospi_cash, kospi_reason, kospi_details = standalone_cash_for_asset(series["KOSPI"], "KOSPI", percentile_years) if "KOSPI" in series else (np.nan, "데이터 없음", {})
sp_cash, sp_reason, sp_details = standalone_cash_for_asset(series["S&P500"], "S&P500", percentile_years) if "S&P500" in series else (np.nan, "데이터 없음", {})

st.subheader("오늘의 독립 현금비중 추천")

left, right = st.columns(2)

with left:
    st.markdown("### 🇰🇷 KOSPI 계좌")
    st.metric("추천 현금", f"{kospi_cash:g}%" if pd.notna(kospi_cash) else "—")
    if "KOSPI" in metrics.index:
        r = metrics.loc["KOSPI"]
        st.caption(
            f"C-score {r['C-score']:.1f} · E-score {r['E-score']:.1f} · "
            f"MDD {r['52주 MDD']:.1f}% · 50일 이격 {r['50일 이격']:.1f}%"
        )
        if kospi_details:
            st.caption(f"낙폭 판정 {r['판정']} · 시장 상태 {regime_label(kospi_details.get('regime'))}")
    st.info(kospi_reason)

with right:
    st.markdown("### 🇺🇸 S&P500 계좌")
    st.metric("추천 현금", f"{sp_cash:g}%" if pd.notna(sp_cash) else "—")
    if "S&P500" in metrics.index:
        r = metrics.loc["S&P500"]
        st.caption(
            f"C-score {r['C-score']:.1f} · E-score {r['E-score']:.1f} · "
            f"MDD {r['52주 MDD']:.1f}% · 50일 이격 {r['50일 이격']:.1f}%"
        )
        if sp_details:
            st.caption(f"낙폭 판정 {r['판정']} · 시장 상태 {regime_label(sp_details.get('regime'))}")
    st.info(sp_reason)

st.caption(
    "※ 두 숫자는 서로 완전히 독립적입니다. KOSPI용 자금 100, S&P500용 자금 100이 각각 있다고 가정합니다. "
    "BTC·GOLD·KOSDAQ·M7의 움직임은 이 두 현금비중 계산에 영향을 주지 않습니다."
)


# MARKET CASH-ENGINE SIGNALS
st.subheader("1) KOSPI / S&P500 — 현금 엔진 참고 신호")
st.caption("※ 낙폭 판정은 C-score(MDD 깊이)이고, 시장 상태는 MA200 기울기·지속성으로 계산한 별도 Regime입니다. '하락 추세'라는 표현은 시장 상태에만 사용합니다.")
market_names = [x for x in ["KOSPI", "S&P500"] if x in metrics.index]
market_cols = ["현재가","52주 MDD","50일 이격","12개월 수익률","C-score","E-score","판정","기준일"]
market_show = metrics.loc[market_names, market_cols].copy()

# C-score describes drawdown severity only. Regime is a separate trend-state signal.
market_show = market_show.rename(columns={"판정": "낙폭 판정"})
regime_map = {
    "KOSPI": regime_label(kospi_details.get("regime")) if kospi_details else "—",
    "S&P500": regime_label(sp_details.get("regime")) if sp_details else "—",
}
market_show.insert(7, "시장 상태", [regime_map.get(x, "—") for x in market_show.index])
market_show = market_show[["현재가","52주 MDD","50일 이격","12개월 수익률","C-score","E-score","낙폭 판정","시장 상태","기준일"]]

st.dataframe(
    market_show.style.format({
        "현재가": "{:,.2f}",
        "52주 MDD": "{:.1f}%",
        "50일 이격": "{:.1f}%",
        "12개월 수익률": "{:.1f}%",
        "C-score": "{:.1f}",
        "E-score": "{:.1f}",
    }, na_rep="—"),
    use_container_width=True
)

st.subheader("2) BTC · GOLD · KOSDAQ · M7 — 역사적 MDD 저가 판정")
st.caption(
    "현금 추천에는 영향을 주지 않습니다. '역사적 바닥 대비 괴리'는 MDD %p 차이가 아니라, "
    "각 drawdown의 직전 ATH를 100으로 놓았을 때 역사적 최악 바닥가격 대비 현재 가격이 몇 % 위인지 계산합니다."
)
if not aux_df.empty:
    aux_show = aux_df[["현재가","ATH 대비 현재 MDD","역사적 최대 MDD","역사적 바닥 대비 괴리","판정","역사적 최대 MDD 날짜","기준일"]].copy()
    aux_show = aux_show.sort_values("역사적 바닥 대비 괴리", ascending=True)
    st.dataframe(
        aux_show.style.format({
            "현재가": "{:,.2f}",
            "ATH 대비 현재 MDD": "{:.1f}%",
            "역사적 최대 MDD": "{:.1f}%",
            "역사적 바닥 대비 괴리": "+{:.1f}%",
        }, na_rep="—"),
        use_container_width=True
    )
else:
    st.warning("보조자산 MDD 데이터를 계산하지 못했습니다.")


# DAY FREQUENCY
st.subheader("3) MDD 발생확률 — 해당 하락폭에 있었던 거래일 비중")
st.caption(
    "예: -30%가 2%라면 선택한 기간 중 약 2%의 날에 가격이 최근 52주 고점보다 30% 이상 낮았습니다. "
    "자산별 변동성 차이를 머릿속에 익히기 위한 표입니다."
)

freq_rows = []
for name in list(MACRO.keys()) + list(M7.keys()):
    if name in series:
        freq_rows.append({"자산": name, **mdd_day_frequency(series[name], freq_years)})

freq_df = pd.DataFrame(freq_rows).set_index("자산")
st.dataframe(freq_df.style.format("{:.2f}%"), use_container_width=True)


# ENTRY FREQUENCY
st.subheader("4) MDD 진입빈도 — 대략 얼마나 자주 그 선을 깨는가?")
st.caption(
    "같은 약세장에서 며칠 동안 -30% 아래에 머문 것을 매일 새 사건으로 세지 않고, "
    "각 기준선을 위에서 아래로 새로 돌파한 횟수를 이용한 체감 빈도입니다."
)

entry_rows = []
for name in list(MACRO.keys()) + list(M7.keys()):
    if name in series:
        entry_rows.append({"자산": name, **mdd_entry_frequency(series[name], freq_years)})

entry_df = pd.DataFrame(entry_rows).set_index("자산")
st.dataframe(entry_df, use_container_width=True)


# BACKTEST
st.subheader("5) KOSPI / S&P500 독립 현금룰 — 일별 Walk-forward 백테스트")
st.caption(
    "두 시장을 완전히 분리해서 검증합니다. KOSPI는 -15/-20/-30/-40%, S&P500은 -10/-15/-20/-30%에서 "
    "1:2:3:4 누적 비율로 준비 현금을 투입합니다. MA200은 하루 돌파가 아니라 기울기·20일 지속성으로 레짐을 판정합니다. "
    "신호는 매일 종가 기준, 다음 거래일에 적용하며 거래비용 10bp를 반영합니다."
)

for asset in ["KOSPI", "S&P500"]:
    if asset not in series:
        continue

    st.markdown(f"### {'🇰🇷' if asset == 'KOSPI' else '🇺🇸'} {asset}")

    with st.spinner(f"{asset} 독립 현금룰 백테스트 계산 중..."):
        bt_one, stats_one = run_standalone_cash_backtest(
            series[asset],
            asset,
            percentile_years=percentile_years,
            trading_cost_bps=10
        )

    if not stats_one.empty:
        shown = stats_one.copy()
        shown["CAGR"] *= 100
        shown["MDD"] *= 100

        st.dataframe(
            shown.style.format({
                "CAGR": "{:.1f}%",
                "MDD": "{:.1f}%",
                "Sharpe": "{:.2f}",
                "Calmar": "{:.2f}",
            }),
            use_container_width=True
        )

        if not bt_one.empty:
            recent = bt_one.dropna().tail(252)
            st.caption(f"{asset} 최근 약 1년 일별 추천 현금비중")
            st.line_chart(recent["cash"])

            # Useful debug table: lets the user inspect historical turning points.
            st.caption(f"{asset} 최근 20거래일 신호")
            debug = recent[["cash", "reserve_cash", "used_frac", "mdd_pct", "c", "e"]].tail(20).copy()
            debug["used_frac"] *= 100
            debug.columns = ["추천 현금", "시작 현금", "누적 투입률", "MDD", "C-score", "E-score"]
            st.dataframe(
                debug.style.format({
                    "추천 현금": "{:.1f}%",
                    "시작 현금": "{:.1f}%",
                    "누적 투입률": "{:.0f}%",
                    "MDD": "{:.1f}%",
                    "C-score": "{:.1f}",
                    "E-score": "{:.1f}",
                }),
                use_container_width=True
            )
    else:
        st.warning(f"{asset} 백테스트 데이터가 충분하지 않습니다.")

st.markdown("""
**시장별 독립 현금룰**

**KOSPI**
- 상승/박스/하락 기본 현금: **10% / 15% / 20%**
- 하락 모드 시작: **로컬 고점 대비 -15%**
- 준비 현금 투입: **-15 / -20 / -30 / -40% → 누적 10 / 30 / 60 / 100%**
- 회복 확인: 50일선 회복 + 50일선 상승 + 저점 대비 5% 이상 반등 + 하락 레짐 아님
- 박스 안정화가 이어지면 전고점 회복 전에도 새 로컬 고점으로 사이클 리셋

**S&P500**
- 상승/박스/하락 기본 현금: **5% / 10% / 15%**
- 하락 모드 시작: **로컬 고점 대비 -10%**
- 준비 현금 투입: **-10 / -15 / -20 / -30% → 누적 10 / 30 / 60 / 100%**
- E-score가 매우 높아질 때도 KOSPI보다 더 높은 임계값을 요구해 현금 드래그를 줄임
- 회복/현금 재축적 속도도 KOSPI보다 느리게 설정

**공통 구조**
- 현재 MDD가 아니라 **이번 사이클 최저 MDD**를 기억하는 래칫 방식
- 하락 중 반등했다고 이미 투자한 돈을 다시 현금으로 되돌리지 않음
- MA200은 단순 상·하회가 아니라 **60일 기울기 + 최근 20일 하회 일수**로 상승/박스/하락 레짐 판정
- E가 식어도 현금을 하루 만에 확 줄이지 않고 **2.5%p씩 천천히 해제**
- 폭락 후 박스권에서는 시장별 현금 바닥까지 **2.5%p씩 천천히 재축적**
- 깊은 MDD에서는 최종적으로 **현금 0%까지 허용**
""")

# EXPLANATION
st.subheader("6) 해석")
st.markdown("""
**C-score**  
절대 MDD 기반의 낙폭 점수이며 **KOSPI와 S&P500 현금 엔진용 참고지표**입니다. 가치평가가 아니라 하락 깊이를 봅니다.  
KOSPI는 기존 절대 MDD 기준을 유지하고, **S&P500은 더 낮은 변동성을 반영해 같은 MDD를 더 높은 C-score로 평가**합니다.

**BTC · GOLD · KOSDAQ · M7 저가 판정**  
C-score를 쓰지 않습니다. 전체 가용 가격 역사에서 `현재 ATH MDD`와 `역사적 최대 ATH MDD`를 구한 뒤, **역사적 최악 바닥가격 대비 현재 가격의 실제 괴리율**로 판정합니다.  
보수적으로 **+5% 이내만 극단적 매수**, +10% 이내 매우 싸다, +20% 이내 싸다, +35% 이내 관심으로 표시합니다. 이 신호들은 KOSPI/S&P500 현금비중에 영향을 주지 않습니다.

**E-score**  
`12개월 수익률 percentile × 60% + 50일 이격 percentile × 40%`입니다. 과열 시 탄약을 만드는 보조 신호입니다.  
KOSPI와 S&P500의 E→현금 매핑도 다르며, S&P500은 더 극단적인 E에서만 큰 현금비중을 권합니다.

**Regime**  
200일선을 하루 이틀 깼는지는 중요하게 보지 않습니다. **200일선 60일 기울기와 최근 20일 동안 200일선 아래에 있었던 일수**를 함께 사용해 상승/박스/하락을 구분합니다.

**Path-dependent MDD**  
현금 추천은 오늘의 52주 MDD만 보고 매일 초기화하지 않습니다. 한 번 하락 모드가 시작되면 **그 사이클에서 실제로 찍었던 최저 MDD**를 기억합니다. 바닥 -30% 후 현재 -23%로 반등해도 -30% 단계에서 이미 집행한 매수는 되돌리지 않습니다.

**Recovery / Reset**  
50일선 회복·상승과 저점 대비 반등이 확인되면 남은 현금 일부를 추가 투자합니다. 또한 전고점을 끝내 회복하지 못해도 박스권 안정화가 충분히 지속되면 옛 전고점을 버리고 새로운 로컬 고점을 기준으로 다음 사이클을 시작합니다.

**현금 재축적**  
폭락 후 현금이 거의 0%가 된 상태에서 박스권으로 넘어가도 현금을 하루 만에 10~20%로 만들지 않습니다. 시장별 속도로 **2.5%p씩 점진적으로** 탄약을 다시 만듭니다.

추천값은 기계적 기준선입니다. 실제 운용에서는 포트폴리오 상황과 시장 해석에 따라 수동 조정할 수 있습니다.
""")

st.warning(
    "이 대시보드는 투자 의사결정 보조도구입니다. "
    "Yahoo Finance 데이터 오류·지연 가능성이 있으며, 백테스트는 세금·슬리피지·실제 ETF 비용을 완전히 반영하지 않습니다."
)
