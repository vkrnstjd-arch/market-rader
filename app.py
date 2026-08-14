
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


def c_score_from_mdd(mdd_pct):
    """Absolute-MDD C-score. No percentile / valuation estimates are used."""
    if pd.isna(mdd_pct):
        return np.nan

    # x is positive drawdown magnitude: -20% MDD -> x=20
    x = max(0.0, -float(mdd_pct))
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
        return "🟠 베어마켓"
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

    # C-score: ABSOLUTE 52-week MDD only.
    c = c_score_from_mdd(cur_dd * 100)

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


def e_cash_target(e_score):
    """
    Cash to BUILD while the market is not in a >=10% drawdown.

    Neutral-market cash floor is 10%. E-score raises it up to 60%:
      E <= 70 -> 10%
      E 75 -> 15%
      E 80 -> 20%
      E 85 -> 30%
      E 90 -> 40%
      E 94 -> 50%
      E 97+ -> 60%

    Between anchors, interpolate and round to 2.5%p.
    The total system can still reach 0% cash after deep-MDD deployment.
    """
    if pd.isna(e_score):
        return 10.0

    e = float(np.clip(e_score, 0, 100))
    xp = np.array([0, 70, 75, 80, 85, 90, 94, 97, 100], dtype=float)
    fp = np.array([10, 10, 15, 20, 30, 40, 50, 60, 60], dtype=float)
    return quantize_cash(np.interp(e, xp, fp))


def drawdown_deployment(mdd_pct):
    """
    Conservative 1:2:3:4 deployment of the cash reserve.

    -10%: deploy 10% of reserve (1/10)
    -20%: deploy another 20% (cumulative 30%)
    -30%: deploy another 30% (cumulative 60%)
    -40%: deploy remaining 40% (cumulative 100%)
    """
    if pd.isna(mdd_pct):
        return 0.0, "대기", None

    mdd = float(mdd_pct)
    if mdd <= -40:
        return 1.00, "-40%: 잔여 40% 투입", None
    if mdd <= -30:
        return 0.60, "-30%: 누적 60% 투입", -40
    if mdd <= -20:
        return 0.30, "-20%: 누적 30% 투입", -30
    if mdd <= -10:
        return 0.10, "-10%: 누적 10% 투입", -20
    return 0.0, "E-score로 현금 적립", -10


def build_e_score_series(close: pd.Series, asset: str, years=5):
    """Walk-forward E-score series using only information available on each date."""
    dd, sep, ret12 = indicator_series(close, asset)
    df = pd.DataFrame({"price": close, "dd": dd, "sep": sep, "ret12": ret12}).sort_index()

    ppy = periods_per_year(asset)
    win = max(int(ppy * years), ppy)
    minp = max(int(ppy * min(years, 2)), int(ppy * 0.75))

    rank_sep = df["sep"].rolling(win, min_periods=minp).rank(pct=True)
    rank_ret = df["ret12"].rolling(win, min_periods=minp).rank(pct=True)
    df["e"] = 0.60 * (rank_ret * 100) + 0.40 * (rank_sep * 100)
    df["mdd_pct"] = df["dd"] * 100
    df["c"] = df["mdd_pct"].map(c_score_from_mdd)
    return df


def cash_path_from_signals(df: pd.DataFrame):
    """
    Stateful engine.

    Above -10% MDD: E-score can only BUILD cash, never reduce it.
    At the first -10% breach: freeze the cash then on hand as the reserve.
    Deeper drawdowns deploy that frozen reserve in 1:2:3:4 proportions.
    After recovery above -10%, keep remaining cash; later E-score can rebuild it higher.
    """
    rows = []
    prev_cash = None
    reserve_cash = None
    in_deploy = False

    for dt, r in df.iterrows():
        e = r.get("e", np.nan)
        mdd_pct = r.get("mdd_pct", np.nan)
        c = r.get("c", np.nan)

        if pd.isna(mdd_pct):
            continue

        used_frac, stage_text, next_level = drawdown_deployment(mdd_pct)

        if mdd_pct > -10:
            e_target = e_cash_target(e)
            # E-score is an accumulation signal: never spend cash merely because E cools.
            if prev_cash is None:
                cash = e_target
            else:
                cash = max(float(prev_cash), float(e_target))
            reserve_cash = cash
            in_deploy = False
            mode = "E 적립" if e_target > (prev_cash if prev_cash is not None else -1) else "현금 유지"
        else:
            if not in_deploy or reserve_cash is None:
                # Freeze the actual cash available just before the drawdown trigger.
                reserve_cash = prev_cash if prev_cash is not None else e_cash_target(e)
                in_deploy = True
            cash = round(float(reserve_cash * (1 - used_frac)), 1)
            mode = "MDD 투입"

        rows.append({
            "date": dt,
            "cash": cash,
            "reserve_cash": reserve_cash,
            "used_frac": used_frac,
            "stage": stage_text,
            "next_level": next_level,
            "c": c,
            "e": e,
            "mdd_pct": mdd_pct,
            "mode": mode,
        })
        prev_cash = cash

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def standalone_cash_for_asset(close: pd.Series, asset: str, years: int):
    sig = build_e_score_series(close, asset, years)
    path = cash_path_from_signals(sig)
    if path.empty:
        return np.nan, "데이터 없음", {}

    r = path.iloc[-1]
    cash = float(r["cash"])
    reserve = float(r["reserve_cash"]) if pd.notna(r["reserve_cash"]) else np.nan
    used_pct = float(r["used_frac"] * 100)
    mdd = float(r["mdd_pct"])
    next_level = r["next_level"]

    if r["mode"] == "E 적립":
        reason = f"E-score {r['e']:.1f} → 현금 적립 구간 → 현금 {cash:g}%"
    else:
        nxt = f" · 다음 추가매수 {next_level:.0f}%" if pd.notna(next_level) else " · 전액 투입 단계"
        reason = (
            f"MDD {mdd:.1f}% → 시작 현금 {reserve:g}% 중 누적 {used_pct:.0f}% 투입"
            f" → 현금 {cash:g}%{nxt}"
        )

    details = {
        "reserve_cash": reserve,
        "used_pct": used_pct,
        "next_level": next_level,
        "mode": r["mode"],
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
        return f"🟠 {name}: 베어마켓급 낙폭 (C {c:.1f})"
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
    - Above -10% MDD, E-score builds 10~60% cash (neutral floor 10%).
    - Once -10% is breached, cash available at that point is frozen as the reserve.
    - Reserve is deployed at -10/-20/-30/-40% in cumulative 10/30/60/100% steps.
    - Signal at t close is applied to t+1 return.
    """
    sig = build_e_score_series(close, asset, percentile_years).copy()

    # Stocks: business-day calendar. Forward-fill local holidays.
    cal = pd.date_range(sig.index.min(), sig.index.max(), freq="B")
    sig = sig.reindex(cal).ffill()
    sig["ret1"] = sig["price"].pct_change()

    path = cash_path_from_signals(sig)
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
    "C-score는 52주 고점 대비 절대 MDD만 사용하고, E-score는 12개월 수익률 60% + 50일 이격 40%의 역사적 percentile로 계산합니다. "
    "E-score로 현금을 쌓고, MDD -10/-20/-30/-40%에서 1:2:3:4로 현금을 투입합니다."
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
**절대 MDD 기반 C-score**

- **-10% → C 70**: 본격 조정
- **-15% → C 80**: 큰 조정
- **-20% → C 90**: 베어마켓
- **-25% → C 95**: 대폭락
- **-30% → C 97.5**: 매우 큰 폭락
- **-35% → C 99**: 위기 수준
- **-40% 이하 → C 100**: 역사적 위기

C-score에는 **50일 이격도나 추정 밸류에이션을 넣지 않습니다.**
""")

    st.caption("GOLD는 Yahoo Finance의 금 선물(GC=F)을 가격 프록시로 사용합니다.")

    st.caption("KOSPI와 S&P500은 독립 계좌입니다. 평상시 중립 현금은 10%이며, E-score가 높아질수록 10~60% 범위에서 2.5%p 단위로 현금을 늘립니다. 하락장에서는 시작 현금을 -10/-20/-30/-40% MDD에서 정확히 1:2:3:4 비율로 투입하며, 깊은 하락에서는 현금 0%까지 내려갈 수 있습니다.")


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
cols = ["현재가","52주 MDD","50일 이격","12개월 수익률","C-score","E-score","판정","기준일"]
macro_show = macro_df[cols].sort_values("C-score", ascending=False)

st.dataframe(
    macro_show.style.format({
        "현재가": "{:,.2f}",
        "52주 MDD": "{:.1f}%",
        "50일 이격": "{:.1f}%",
        "12개월 수익률": "{:.1f}%",
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
    "두 시장을 완전히 분리해서 검증합니다. C는 절대 MDD, E는 12개월 수익률·50일 이격 percentile만 사용합니다. "
    "E로 확보한 현금을 -10/-20/-30/-40% MDD에서 1:2:3:4 비율로 투입합니다. "
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
**독립 현금룰**
- 각 시장마다 별도의 100짜리 계좌가 있다고 가정
- **C-score = 절대 52주 MDD만 사용**
- **E-score = 12개월 수익률 percentile 60% + 50일 이격 percentile 40%**
- MDD가 -10%보다 얕을 때는 **중립 현금 10%**를 기본으로 두고 E-score로 최대 60%까지 적립
- E≤70 → 10%, E=75 → 15%, E=80 → 20%, E=85 → 30%, E=90 → 40%, E=94 → 50%, E=97+ → 60%
- 따라서 박스권·중립장에서는 대체로 현금 10%를 유지하고, 깊은 MDD에서만 0%까지 내려갈 수 있음
- MDD가 **-10%에 처음 진입하는 순간의 현금**을 이번 하락장의 '시작 현금'으로 고정
- **-10%: 시작 현금의 10% 투입**
- **-20%: 추가 20% 투입 (누적 30%)**
- **-30%: 추가 30% 투입 (누적 60%)**
- **-40%: 남은 40% 전부 투입 (누적 100%)**
- 중간 구간에서는 현금을 더 쓰지 않고 다음 MDD 단계까지 대기
- **KOSPI와 S&P500은 서로 완전히 독립적으로 계산**
""")


# EXPLANATION
st.subheader("6) 해석")
st.markdown("""
**52주 MDD**  
현재 가격이 최근 365일 최고점에서 얼마나 내려와 있는지입니다.

**C-score**  
절대 MDD만 점수화합니다. 역사적 percentile이나 Forward PBR 같은 추정치는 사용하지 않습니다.  
핵심 기준은 **-10%=70 / -15%=80 / -20%=90 / -25%=95 / -30%=97.5 / -35%=99 / -40%=100**입니다.

**E-score**  
`12개월 수익률 percentile × 60% + 50일 이격 percentile × 40%`  
시장이 얼마나 과열됐는지 보고 평상시 **중립 현금 10%**에서 시작해 최대 60%까지 적립하는 용도입니다. E가 낮다고 해서 평상시 현금을 0%로 만들지는 않으며, 현금 0%는 깊은 MDD에서 준비한 현금을 모두 투입했을 때 도달할 수 있습니다.

**현금 투입 원칙**  
E-score는 현금을 **쌓는 신호**로만 사용하며, E-score가 낮아졌다는 이유만으로 현금을 다시 주식에 투입하지 않습니다. 실제 현금 투입은 MDD 단계에서만 실행합니다.

E-score로 쌓은 현금을 작은 조정에서 한꺼번에 쓰지 않습니다. -10% MDD 진입 시점의 현금을 기준으로 **1:2:3:4** 비율로 더 깊은 하락에 더 많이 배치합니다.

**수동 판단 여지**  
추천 현금비중은 기준선일 뿐이며, 실제 운용에서는 시장 상황·개별 포트폴리오 판단에 따라 사용자가 수동으로 조정할 수 있습니다.

**GOLD / BTC / KOSDAQ / M7**  
C-score와 E-score, MDD 빈도는 참고용으로 계속 보여주지만 **KOSPI와 S&P500의 현금비중에는 영향을 주지 않습니다.**
""")

st.warning(
    "이 대시보드는 투자 의사결정 보조도구입니다. "
    "Yahoo Finance 데이터 오류·지연 가능성이 있으며, 백테스트는 세금·슬리피지·실제 ETF 비용을 완전히 반영하지 않습니다."
)
