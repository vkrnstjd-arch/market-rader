
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


# =========================================================
# DATA
# =========================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_close(ticker: str) -> pd.Series:
    df = yf.download(
        ticker,
        start="2010-01-01",
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


def rating_label(score):
    if pd.isna(score):
        return "—"
    if score >= 95:
        return "🔴 극단적"
    if score >= 90:
        return "🟠 매우 싸다"
    if score >= 80:
        return "🟡 싸다"
    if score >= 70:
        return "🟢 관심"
    return "⚪ 평범"


def calc_metrics(close: pd.Series, asset: str, years: int):
    dd, sep, ret12 = indicator_series(close, asset)

    cur_dd = dd.iloc[-1]
    cur_sep = sep.iloc[-1]
    cur_ret = ret12.iloc[-1]

    p_dd = distress_percentile(dd, cur_dd, years)
    p_sep = distress_percentile(sep, cur_sep, years)

    p_ret_up = euphoria_percentile(ret12, cur_ret, years)
    p_sep_up = euphoria_percentile(sep, cur_sep, years)

    c = 0.60 * p_dd + 0.40 * p_sep
    e = 0.60 * p_ret_up + 0.40 * p_sep_up

    return {
        "현재가": close.iloc[-1],
        "52주 MDD": cur_dd * 100,
        "50일 이격": cur_sep * 100,
        "12개월 수익률": cur_ret * 100 if pd.notna(cur_ret) else np.nan,
        "MDD 극단도": p_dd,
        "이격 극단도": p_sep,
        "C-score": c,
        "판정": rating_label(c),
        "E-score": e,
        "기준일": str(close.index[-1].date()),
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
# CASH RULE
# =========================================================

def quantize_cash(value, step=2.5):
    """Round target cash to 2.5%p increments and clamp to 0~60%."""
    value = float(np.clip(value, 0, 60))
    return round(value / step) * step


def target_cash_from_scores(c_scores: pd.Series, e_scores: pd.Series):
    """
    Shared live/backtest cash engine.

    Distress has priority. The second-highest Macro C-score is used so one
    unusually volatile asset cannot by itself force the whole portfolio to deploy cash.

    The old 10%p anchors are preserved, but values BETWEEN anchors are interpolated
    and rounded to 2.5%p increments:

      C2 70 -> cash 30
      C2 80 -> cash 20
      C2 90 -> cash 10
      C2 95 -> cash  0

    When no broad distress signal exists, broad euphoria raises cash gradually:
      E median <85 -> cash 30
      85 -> 30
      90 -> 40
      94 -> 50
      97 -> 60

    Final cash is always one of:
      0, 2.5, 5, 7.5, ... , 60
    """
    cs = pd.Series(c_scores).dropna().sort_values(ascending=False)
    second_c = cs.iloc[1] if len(cs) >= 2 else (cs.iloc[0] if len(cs) else np.nan)

    # BUY / distress side: deploy cash progressively.
    if pd.notna(second_c) and second_c >= 70:
        if second_c >= 95:
            raw_cash = 0.0
        elif second_c >= 90:
            # 90 -> 10, 95 -> 0
            raw_cash = 10 - 2 * (second_c - 90)
        else:
            # 70 -> 30, 80 -> 20, 90 -> 10
            raw_cash = 100 - second_c

        cash = quantize_cash(raw_cash)
        return cash, second_c, np.nan, "distress"

    # SELL / euphoria side: rebuild cash progressively.
    es = pd.Series(e_scores).dropna()
    e_med = es.median() if len(es) else np.nan

    if pd.isna(e_med) or e_med < 85:
        raw_cash = 30.0
    elif e_med < 90:
        # 85 -> 30, 90 -> 40
        raw_cash = 30 + 2 * (e_med - 85)
    elif e_med < 94:
        # 90 -> 40, 94 -> 50
        raw_cash = 40 + 2.5 * (e_med - 90)
    elif e_med < 97:
        # 94 -> 50, 97 -> 60
        raw_cash = 50 + (10/3) * (e_med - 94)
    else:
        raw_cash = 60.0

    cash = quantize_cash(raw_cash)
    return cash, second_c, e_med, "euphoria"


def standalone_cash_rule(c_score, e_score):
    """
    Standalone cash rule for ONE market sleeve.

    Think of KOSPI and S&P500 as two independent 100-unit portfolios.
    Each portfolio decides its own cash ratio from its own C/E scores only.

    Distress (buying opportunity) has priority:
      C < 70: use euphoria/neutral rule
      C 70 -> cash 30
      C 80 -> cash 20
      C 90 -> cash 10
      C 95+ -> cash 0

    Between anchors, interpolate and round to 2.5%p.

    When C < 70, euphoria rebuilds cash:
      E < 85 -> cash 30
      E 85 -> 30
      E 90 -> 40
      E 94 -> 50
      E 97+ -> 60

    Final range: 0~60%, step 2.5%p.
    """
    if pd.notna(c_score) and c_score >= 70:
        if c_score >= 95:
            raw_cash = 0.0
        elif c_score >= 90:
            raw_cash = 10 - 2 * (c_score - 90)  # 90->10, 95->0
        else:
            raw_cash = 100 - c_score             # 70->30, 80->20, 90->10

        cash = quantize_cash(raw_cash)
        return cash, f"C-score {c_score:.1f} → 하락구간에서 현금 투입"

    if pd.isna(e_score) or e_score < 85:
        raw_cash = 30.0
        reason = "중립"
    elif e_score < 90:
        raw_cash = 30 + 2 * (e_score - 85)       # 85->30, 90->40
        reason = f"E-score {e_score:.1f} → 수익실현 시작"
    elif e_score < 94:
        raw_cash = 40 + 2.5 * (e_score - 90)     # 90->40, 94->50
        reason = f"E-score {e_score:.1f} → 과열"
    elif e_score < 97:
        raw_cash = 50 + (10/3) * (e_score - 94)  # 94->50, 97->60
        reason = f"E-score {e_score:.1f} → 강한 과열"
    else:
        raw_cash = 60.0
        reason = f"E-score {e_score:.1f} → 극단적 과열"

    cash = quantize_cash(raw_cash)
    return cash, reason


def standalone_cash_for_asset(metrics_df, asset):
    if asset not in metrics_df.index:
        return np.nan, "데이터 없음"
    row = metrics_df.loc[asset]
    return standalone_cash_rule(row["C-score"], row["E-score"])


def macro_cash_rule(macro_df: pd.DataFrame):
    c_scores = macro_df["C-score"]

    # BTC excluded from broad euphoria median because its normal volatility is much larger.
    e_assets = [x for x in ["KOSPI", "KOSDAQ", "S&P500", "GOLD"] if x in macro_df.index]
    e_scores = macro_df.loc[e_assets, "E-score"] if e_assets else pd.Series(dtype=float)

    cash, second_c, e_med, mode = target_cash_from_scores(c_scores, e_scores)

    if mode == "distress":
        if second_c >= 95:
            reason = f"두 번째로 높은 Macro C-score가 {second_c:.1f} → 극단적 할인"
        elif second_c >= 90:
            reason = f"두 번째로 높은 Macro C-score가 {second_c:.1f} → 매우 싼 구간"
        elif second_c >= 80:
            reason = f"두 번째로 높은 Macro C-score가 {second_c:.1f} → 싼 구간"
        else:
            reason = f"두 번째로 높은 Macro C-score가 {second_c:.1f} → 관심 구간"

        return cash, reason + f" → 현금 {cash:g}%"

    if pd.notna(e_med) and e_med >= 85:
        return cash, f"광범위 E-score 중앙값 {e_med:.1f} → 기계적 수익실현 → 현금 {cash:g}%"

    return cash, f"중립 구간 → 현금 {cash:g}%"


def m7_opportunity(m7_df: pd.DataFrame):
    if m7_df.empty:
        return "특별한 M7 기회 없음"

    row = m7_df.sort_values("C-score", ascending=False).iloc[0]
    name = m7_df.sort_values("C-score", ascending=False).index[0]
    c = row["C-score"]

    if c >= 95:
        return f"🔴 {name}: 극단적 개별 매수기회 (C {c:.1f})"
    if c >= 90:
        return f"🟠 {name}: 매우 드문 할인 (C {c:.1f})"
    if c >= 80:
        return f"🟡 {name}: 싸다 (C {c:.1f})"
    if c >= 70:
        return f"🟢 {name}: 관심구간 (C {c:.1f})"
    return "⚪ 현재 M7에는 강한 개별 할인 신호 없음"


# =========================================================
# DAILY WALK-FORWARD BACKTEST
# =========================================================
def build_daily_signal_frame(close: pd.Series, asset: str):
    """
    Daily signal frame. C/E scores are computed using only information
    available up to that date. The final percentile window is ~5 business years.
    """
    dd, sep, ret12 = indicator_series(close, asset)

    f = pd.DataFrame({
        "price": close,
        "dd": dd,
        "sep": sep,
        "ret12": ret12,
    }).sort_index()

    return f


@st.cache_data(ttl=1800, show_spinner=False)
def run_cash_backtest_daily(series_dict, trading_cost_bps=10):
    """
    DAILY walk-forward cash timing backtest.

    - Signals are calculated at close on day t using only data known by t.
    - The resulting cash target is applied to t -> t+1 return.
    - Macro 5 prices are aligned to a common business-day calendar and
      forward-filled over local holidays.
    - The invested portion is an equal-weight Macro 5 basket so the test
      isolates CASH TIMING rather than asset-selection skill.
    - Trading cost is charged only when target cash weight changes.
      Default: 10 bps on the absolute portfolio weight changed.

    This is deliberately a cash-rule test, not a full portfolio optimizer.
    """
    frames = {}
    for asset in MACRO:
        if asset in series_dict:
            frames[asset] = build_daily_signal_frame(series_dict[asset], asset)

    if len(frames) < 4:
        return pd.DataFrame(), pd.DataFrame()

    # Common business-day calendar.
    start_date = max(df.index.min() for df in frames.values())
    end_date = min(df.index.max() for df in frames.values())
    cal = pd.date_range(start_date, end_date, freq="B")

    aligned = {}
    for asset, df in frames.items():
        x = df.reindex(cal).ffill()

        # Recompute returns after alignment so BTC/weekend moves flow into
        # the next business-day return instead of disappearing.
        x["ret1"] = x["price"].pct_change()

        # Rolling 5-year (~1260 business days) percentile, no future data.
        win = 1260
        minp = 504  # require about 2 years before trusting a score

        rank_dd = x["dd"].rolling(win, min_periods=minp).rank(pct=True)
        rank_sep = x["sep"].rolling(win, min_periods=minp).rank(pct=True)
        rank_ret = x["ret12"].rolling(win, min_periods=minp).rank(pct=True)

        # Distress: lower raw value = higher percentile score
        x["c"] = 0.60 * ((1 - rank_dd) * 100) + 0.40 * ((1 - rank_sep) * 100)

        # Euphoria: higher return / higher positive separation = higher score
        x["e"] = 0.60 * (rank_ret * 100) + 0.40 * (rank_sep * 100)

        aligned[asset] = x

    # Build equal-weight Macro basket daily return.
    ret_df = pd.DataFrame({a: x["ret1"] for a, x in aligned.items()})
    basket_ret = ret_df.mean(axis=1, skipna=True)

    c_df = pd.DataFrame({a: x["c"] for a, x in aligned.items()})
    e_df = pd.DataFrame({a: x["e"] for a, x in aligned.items()})

    rows = []
    prev_cash = None

    for i in range(len(cal) - 1):
        dt = cal[i]
        nxt = cal[i + 1]

        cs = c_df.loc[dt].dropna()
        if len(cs) < 4:
            continue

        e_subset = e_df.loc[
            dt,
            [x for x in ["KOSPI","KOSDAQ","S&P500","GOLD"] if x in e_df.columns]
        ].dropna()

        cash, _, _, _ = target_cash_from_scores(cs, e_subset)

        next_ret = basket_ret.loc[nxt]
        if pd.isna(next_ret):
            continue

        invested = 1 - cash / 100

        # Cost only when the target cash allocation changes.
        turnover = 0 if prev_cash is None else abs(cash - prev_cash) / 100
        cost = turnover * (trading_cost_bps / 10000)

        strategy_ret = invested * next_ret - cost

        rows.append({
            "date": dt,
            "cash": cash,
            "basket_ret_next": next_ret,
            "strategy_ret": strategy_ret,
            "fixed0_ret": next_ret,
            "fixed30_ret": 0.70 * next_ret,
            "fixed50_ret": 0.50 * next_ret,
            "turnover": turnover,
            "cost": cost,
        })

        prev_cash = cash

    bt = pd.DataFrame(rows).set_index("date")
    if bt.empty:
        return bt, pd.DataFrame()

    def stats(ret):
        ret = ret.dropna()
        if len(ret) < 252:
            return {"CAGR": np.nan, "MDD": np.nan, "Sharpe": np.nan, "Calmar": np.nan}

        wealth = (1 + ret).cumprod()
        years = len(ret) / 252
        cagr = wealth.iloc[-1] ** (1 / years) - 1

        dd = wealth / wealth.cummax() - 1
        mdd = dd.min()

        ann_vol = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
        calmar = cagr / abs(mdd) if mdd < 0 else np.nan

        return {
            "CAGR": cagr,
            "MDD": mdd,
            "Sharpe": sharpe,
            "Calmar": calmar,
        }

    stats_df = pd.DataFrame({
        "C-score 일별 현금룰": stats(bt["strategy_ret"]),
        "현금 0% 고정": stats(bt["fixed0_ret"]),
        "현금 30% 고정": stats(bt["fixed30_ret"]),
        "현금 50% 고정": stats(bt["fixed50_ret"]),
    }).T

    return bt, stats_df



@st.cache_data(ttl=1800, show_spinner=False)
def run_standalone_cash_backtest(close: pd.Series, asset: str, trading_cost_bps=10):
    """
    Daily walk-forward backtest for one market only.

    - Uses only that market's own C/E score.
    - Signal at day t close -> applied to t+1 return.
    - 5-year rolling percentile, ~2-year minimum warmup.
    - Cash target 0~60%, 2.5%p increments.
    - Trading cost charged when target cash changes.
    """
    dd, sep, ret12 = indicator_series(close, asset)

    df = pd.DataFrame({
        "price": close,
        "dd": dd,
        "sep": sep,
        "ret12": ret12,
    }).dropna(subset=["price"]).sort_index()

    # Use business-day calendar for stocks.
    cal = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(cal).ffill()
    df["ret1"] = df["price"].pct_change()

    win = 1260
    minp = 504

    rank_dd = df["dd"].rolling(win, min_periods=minp).rank(pct=True)
    rank_sep = df["sep"].rolling(win, min_periods=minp).rank(pct=True)
    rank_ret = df["ret12"].rolling(win, min_periods=minp).rank(pct=True)

    df["c"] = 0.60 * ((1 - rank_dd) * 100) + 0.40 * ((1 - rank_sep) * 100)
    df["e"] = 0.60 * (rank_ret * 100) + 0.40 * (rank_sep * 100)

    rows = []
    prev_cash = None

    for i in range(len(df) - 1):
        dt = df.index[i]
        nxt = df.index[i + 1]

        c = df.loc[dt, "c"]
        e = df.loc[dt, "e"]
        next_ret = df.loc[nxt, "ret1"]

        if pd.isna(c) or pd.isna(e) or pd.isna(next_ret):
            continue

        cash, _ = standalone_cash_rule(c, e)

        turnover = 0 if prev_cash is None else abs(cash - prev_cash) / 100
        cost = turnover * (trading_cost_bps / 10000)

        invested = 1 - cash / 100
        strategy_ret = invested * next_ret - cost

        rows.append({
            "date": dt,
            "cash": cash,
            "c": c,
            "e": e,
            "market_ret_next": next_ret,
            "strategy_ret": strategy_ret,
            "fixed0_ret": next_ret,
            "fixed30_ret": 0.70 * next_ret,
            "fixed50_ret": 0.50 * next_ret,
            "turnover": turnover,
            "cost": cost,
        })

        prev_cash = cash

    bt = pd.DataFrame(rows).set_index("date")
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
        f"{asset} C/E 현금룰": stats(bt["strategy_ret"]),
        "현금 0% 고정": stats(bt["fixed0_ret"]),
        "현금 30% 고정": stats(bt["fixed30_ret"]),
        "현금 50% 고정": stats(bt["fixed50_ret"]),
    }).T

    return bt, stats_df


# =========================================================
# UI
# =========================================================
st.title("📉 Market Distress Radar")
st.caption(
    "절대 하락률이 아니라 각 자산의 자기 역사에서 얼마나 이례적인 하락인지 비교합니다. "
    "C-score = MDD 극단도 60% + 50일선 이격 극단도 40%."
)

with st.sidebar:
    st.header("설정")

    percentile_years = st.selectbox(
        "현재 C-score 비교기간",
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
**C-score 한글 판정**

- ⚪ **평범**: 0~69.9
- 🟢 **관심**: 70~79.9
- 🟡 **싸다**: 80~89.9
- 🟠 **매우 싸다**: 90~94.9
- 🔴 **극단적**: 95~100

점수가 높을수록 **그 자산 자신의 역사에 비해** 이례적으로 눌려 있다는 뜻입니다.
""")

    st.caption("GOLD는 Yahoo Finance의 금 선물(GC=F)을 가격 프록시로 사용합니다.")

    st.caption("KOSPI와 S&P500의 현금비중 추천은 서로 독립적이며 0~60% 범위에서 **2.5%p 단위**로 움직입니다.")


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


# SUMMARY — TWO INDEPENDENT CASH SLEEVES
kospi_cash, kospi_reason = standalone_cash_for_asset(metrics, "KOSPI")
sp_cash, sp_reason = standalone_cash_for_asset(metrics, "S&P500")

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
    st.info(sp_reason)

st.caption(
    "※ 두 숫자는 서로 완전히 독립적입니다. KOSPI용 자금 100, S&P500용 자금 100이 각각 있다고 가정합니다. "
    "BTC·GOLD·KOSDAQ·M7의 움직임은 이 두 현금비중 계산에 영향을 주지 않습니다."
)


# MACRO 5
st.subheader("1) Macro 5 — 현재 C-score")
cols = ["현재가","52주 MDD","50일 이격","12개월 수익률","MDD 극단도","이격 극단도","C-score","판정","기준일"]
macro_show = macro_df[cols].sort_values("C-score", ascending=False)

st.dataframe(
    macro_show.style.format({
        "현재가": "{:,.2f}",
        "52주 MDD": "{:.1f}%",
        "50일 이격": "{:.1f}%",
        "12개월 수익률": "{:.1f}%",
        "MDD 극단도": "{:.1f}",
        "이격 극단도": "{:.1f}",
        "C-score": "{:.1f}",
    }, na_rep="—"),
    use_container_width=True
)


# M7
st.subheader("2) Magnificent 7 — 개별 C-score")
m7_show = m7_df[cols].sort_values("C-score", ascending=False)

st.dataframe(
    m7_show.style.format({
        "현재가": "${:,.2f}",
        "52주 MDD": "{:.1f}%",
        "50일 이격": "{:.1f}%",
        "12개월 수익률": "{:.1f}%",
        "MDD 극단도": "{:.1f}",
        "이격 극단도": "{:.1f}",
        "C-score": "{:.1f}",
    }, na_rep="—"),
    use_container_width=True
)


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
    "두 시장을 완전히 분리해서 검증합니다. 각 시장은 자기 자신의 MDD·50일 이격·12개월 수익률만 사용합니다. "
    "다른 자산이 싸거나 비싸도 해당 시장의 현금비중에는 영향을 주지 않습니다. "
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
            debug = recent[["cash", "c", "e"]].tail(20).copy()
            debug.columns = ["추천 현금", "C-score", "E-score"]
            st.dataframe(
                debug.style.format({
                    "추천 현금": "{:.1f}%",
                    "C-score": "{:.1f}",
                    "E-score": "{:.1f}",
                }),
                use_container_width=True
            )
    else:
        st.warning(f"{asset} 백테스트 데이터가 충분하지 않습니다.")

st.markdown("""
**독립 현금룰**
- 각 시장마다 별도의 100짜리 계좌가 있다고 가정
- 추천 현금: **0 / 2.5 / 5 / … / 60%**
- 해당 시장 C-score가 70을 넘으면 하락폭이 커질수록 현금을 점진적으로 투입
- C=70 → 현금 약 30%, C=80 → 20%, C=90 → 10%, C=95+ → 0%
- C<70일 때는 해당 시장 E-score로 수익실현
- E=85 → 약 30%, E=90 → 40%, E=94 → 50%, E=97+ → 60%
- **KOSPI 현금에는 S&P/BTC/Gold 등이 전혀 영향을 주지 않음**
- **S&P500 현금에는 KOSPI/BTC/Gold 등이 전혀 영향을 주지 않음**
""")


# EXPLANATION
st.subheader("6) 해석")
st.markdown("""
**52주 MDD**  
현재 가격이 최근 365일 최고점에서 얼마나 내려와 있는지.

**MDD 극단도**  
현재 MDD가 해당 자산 자신의 최근 3/5/10년 역사에서 얼마나 드문지.  
예를 들어 BTC -35%와 S&P500 -35%를 같은 사건으로 취급하지 않습니다.

**50일 이격 극단도**  
현재 가격이 50일 평균선 아래로 떨어진 정도가 자기 역사에서 얼마나 드문지.

**C-score**  
`MDD 극단도 × 60% + 50일 이격 극단도 × 40%`

따라서 C-score 95는 “95% 하락했다”는 뜻이 아니라,  
**그 자산 자신의 역사와 비교했을 때 매우 드문 스트레스 상태**라는 뜻입니다.

**GOLD / BTC / KOSDAQ / M7**  
계속 자기 역사 대비 C-score와 MDD 빈도를 보여주지만, **KOSPI와 S&P500의 현금비중 추천에는 섞지 않습니다.** 이들은 어느 자산군이 상대적으로 매력적인지 판단하기 위한 참고판입니다.
""")

st.warning(
    "이 대시보드는 투자 의사결정 보조도구입니다. "
    "Yahoo Finance 데이터 오류·지연 가능성이 있으며, 백테스트는 세금·슬리피지·실제 ETF 비용을 완전히 반영하지 않습니다."
)
