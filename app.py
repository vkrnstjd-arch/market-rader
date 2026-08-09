
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Market Distress Radar",
    page_icon="📉",
    layout="wide",
)

MACRO = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "S&P500": "^GSPC",
    "BTC": "BTC-USD",
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
    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    return close.astype(float)


def pct_rank_distress(series: pd.Series, current: float, years: int = 5) -> float:
    """High score = current value is unusually LOW versus its own history."""
    if series.empty or pd.isna(current):
        return np.nan
    cutoff = series.index.max() - pd.DateOffset(years=years)
    hist = series.loc[series.index >= cutoff].dropna()
    if hist.empty:
        return np.nan
    return float((hist >= current).mean() * 100)


def pct_rank_euphoria(series: pd.Series, current: float, years: int = 5) -> float:
    """High score = current value is unusually HIGH versus its own history."""
    if series.empty or pd.isna(current):
        return np.nan
    cutoff = series.index.max() - pd.DateOffset(years=years)
    hist = series.loc[series.index >= cutoff].dropna()
    if hist.empty:
        return np.nan
    return float((hist <= current).mean() * 100)


def calculate_metrics(close: pd.Series, percentile_years: int = 5) -> dict:
    roll_high_252 = close.rolling(252, min_periods=60).max()
    dd52 = close / roll_high_252 - 1

    ma50 = close.rolling(50, min_periods=30).mean()
    ma_sep = close / ma50 - 1

    ret12 = close.pct_change(252)

    latest = close.iloc[-1]
    curr_dd = dd52.iloc[-1]
    curr_sep = ma_sep.iloc[-1]
    curr_ret12 = ret12.iloc[-1]

    mdd_pct = pct_rank_distress(dd52, curr_dd, percentile_years)
    ma_distress_pct = pct_rank_distress(ma_sep, curr_sep, percentile_years)

    ret_euphoria_pct = pct_rank_euphoria(ret12, curr_ret12, percentile_years)
    ma_euphoria_pct = pct_rank_euphoria(ma_sep, curr_sep, percentile_years)

    c_score = 0.60 * mdd_pct + 0.40 * ma_distress_pct
    e_score = 0.60 * ret_euphoria_pct + 0.40 * ma_euphoria_pct

    return {
        "Last": latest,
        "52W MDD": curr_dd * 100,
        "50D Sep": curr_sep * 100,
        "12M Return": curr_ret12 * 100 if pd.notna(curr_ret12) else np.nan,
        "MDD Pctl": mdd_pct,
        "50D Pctl": ma_distress_pct,
        "C-score": c_score,
        "E-score": e_score,
        "Date": close.index[-1].date(),
    }


def drawdown_frequency(close: pd.Series, years: int | None = 10) -> dict:
    dd = close / close.rolling(252, min_periods=60).max() - 1

    if years is not None:
        cutoff = dd.index.max() - pd.DateOffset(years=years)
        dd = dd.loc[dd.index >= cutoff]

    dd = dd.dropna()
    levels = [10, 20, 30, 40, 50, 60, 70, 80]
    out = {}
    for x in levels:
        out[f"-{x}%"] = float((dd <= -x / 100).mean() * 100)
    return out


def cash_recommendation(macro_df: pd.DataFrame, m7_df: pd.DataFrame):
    """
    Transparent heuristic draft.
    This is deliberately separated from the metric engine so the rule can be
    replaced after a full historical backtest.
    """
    top2_c = macro_df["C-score"].nlargest(2).mean()
    macro_e = macro_df["E-score"].median()

    if top2_c >= 96:
        base_cash = 0
        macro_reason = "매크로 자산 중 2개 이상이 자기 역사상 극단적 스트레스 구간"
    elif top2_c >= 90:
        base_cash = 10
        macro_reason = "매크로 스트레스가 매우 높음"
    elif top2_c >= 80:
        base_cash = 20
        macro_reason = "매크로 스트레스가 높음"
    else:
        if macro_e >= 99:
            base_cash = 60
            macro_reason = "매크로 자산 전반이 극단적 과열"
        elif macro_e >= 97:
            base_cash = 50
            macro_reason = "매크로 자산 전반이 매우 과열"
        elif macro_e >= 93:
            base_cash = 40
            macro_reason = "매크로 자산 전반이 과열"
        else:
            base_cash = 30
            macro_reason = "중립 구간"

    max_m7_c = m7_df["C-score"].max()
    if max_m7_c >= 98:
        override = -15
        m7_reason = "M7에 C-score 98+ 개별 기회 존재"
    elif max_m7_c >= 95:
        override = -10
        m7_reason = "M7에 C-score 95+ 개별 기회 존재"
    elif max_m7_c >= 90:
        override = -5
        m7_reason = "M7에 C-score 90+ 개별 기회 존재"
    else:
        override = 0
        m7_reason = "M7 극단적 개별 할인 없음"

    final_cash = int(np.clip(base_cash + override, 0, 60))
    return final_cash, base_cash, override, macro_reason, m7_reason


def style_score(v):
    if pd.isna(v):
        return ""
    if v >= 95:
        return "background-color: rgba(255, 80, 80, 0.22); font-weight: 700;"
    if v >= 90:
        return "background-color: rgba(255, 180, 80, 0.20); font-weight: 600;"
    if v >= 80:
        return "background-color: rgba(255, 230, 120, 0.16);"
    return ""


st.title("📉 Market Distress Radar")
st.caption(
    "각 자산을 자기 역사와 비교합니다. "
    "C-score = 52주 MDD percentile 60% + 50일선 이격 percentile 40%. "
    "100에 가까울수록 '이 자산치고 비정상적으로 많이 빠진 상태'입니다."
)

with st.sidebar:
    st.header("설정")
    percentile_years = st.selectbox(
        "C-score percentile 기준",
        options=[3, 5, 10],
        index=1,
        format_func=lambda x: f"최근 {x}년",
    )
    freq_years_label = st.selectbox(
        "MDD 빈도표 기준",
        options=["최근 5년", "최근 10년", "전체 가용기간"],
        index=1,
    )
    freq_years = {"최근 5년": 5, "최근 10년": 10, "전체 가용기간": None}[freq_years_label]

    st.divider()
    st.markdown(
        """
        **해석**
        - C-score 80+: 꽤 이례적
        - C-score 90+: 드문 할인
        - C-score 95+: 매우 드문 할인
        - C-score 98+: 극단적 구간

        현금 추천 규칙은 아직 **백테스트 전 초안**입니다.
        """
    )

if st.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

series = {}
rows = []

with st.spinner("시장 데이터를 불러오는 중..."):
    for name, ticker in ALL.items():
        try:
            s = fetch_close(ticker)
            series[name] = s
            m = calculate_metrics(s, percentile_years)
            m["Asset"] = name
            rows.append(m)
        except Exception as e:
            st.warning(f"{name} ({ticker}) 데이터를 불러오지 못했습니다: {e}")

metrics = pd.DataFrame(rows).set_index("Asset")

if metrics.empty:
    st.error("데이터를 불러오지 못했습니다. 잠시 뒤 다시 시도하세요.")
    st.stop()

macro_df = metrics.loc[[x for x in MACRO if x in metrics.index]].copy()
m7_df = metrics.loc[[x for x in M7 if x in metrics.index]].copy()

if len(macro_df) >= 2 and len(m7_df) >= 1:
    final_cash, base_cash, override, macro_reason, m7_reason = cash_recommendation(macro_df, m7_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("추천 현금 비중", f"{final_cash}%")
    c2.metric("매크로 기본 현금", f"{base_cash}%")
    c3.metric("M7 Opportunity Override", f"{override}%p")
    c4.metric("최고 C-score", f"{metrics['C-score'].max():.1f}")

    st.info(f"**매크로:** {macro_reason}  ·  **M7:** {m7_reason}")
else:
    st.warning("현금 추천 계산에 필요한 데이터가 일부 부족합니다.")

st.subheader("1) Macro 4")
display_cols = [
    "Last", "52W MDD", "50D Sep", "12M Return",
    "MDD Pctl", "50D Pctl", "C-score", "E-score", "Date"
]
macro_show = macro_df[display_cols].copy()

macro_styled = (
    macro_show.style
    .format({
        "Last": "{:,.2f}",
        "52W MDD": "{:.1f}%",
        "50D Sep": "{:.1f}%",
        "12M Return": "{:.1f}%",
        "MDD Pctl": "{:.1f}",
        "50D Pctl": "{:.1f}",
        "C-score": "{:.1f}",
        "E-score": "{:.1f}",
    }, na_rep="—")
    .map(style_score, subset=["C-score"])
)
st.dataframe(macro_styled, use_container_width=True)

st.subheader("2) Magnificent 7 개별 C-score")
m7_show = m7_df[display_cols].sort_values("C-score", ascending=False).copy()
m7_styled = (
    m7_show.style
    .format({
        "Last": "${:,.2f}",
        "52W MDD": "{:.1f}%",
        "50D Sep": "{:.1f}%",
        "12M Return": "{:.1f}%",
        "MDD Pctl": "{:.1f}",
        "50D Pctl": "{:.1f}",
        "C-score": "{:.1f}",
        "E-score": "{:.1f}",
    }, na_rep="—")
    .map(style_score, subset=["C-score"])
)
st.dataframe(m7_styled, use_container_width=True)

st.subheader("3) Macro 4 — 52주 고점 대비 MDD가 얼마나 흔한가?")
st.caption(
    f"{freq_years_label}의 '거래일 기준 발생 비중'입니다. "
    "예: -30%가 2%라면 관측일의 약 2%에서 52주 고점보다 30% 이상 낮았습니다. "
    "한 번의 하락장을 한 사건으로 세는 '에피소드 빈도'와는 다른 지표입니다."
)

freq_rows = []
for name in MACRO:
    if name not in series:
        continue
    row = {"Asset": name, **drawdown_frequency(series[name], freq_years)}
    freq_rows.append(row)

freq_df = pd.DataFrame(freq_rows).set_index("Asset")
st.dataframe(
    freq_df.style.format("{:.2f}%"),
    use_container_width=True,
)

st.subheader("4) M7 — 현재 MDD를 자기 역사와 비교")
m7_compact = m7_df[["52W MDD", "MDD Pctl", "50D Sep", "50D Pctl", "C-score"]].sort_values(
    "C-score", ascending=False
)
st.dataframe(
    m7_compact.style
    .format({
        "52W MDD": "{:.1f}%",
        "MDD Pctl": "{:.1f}",
        "50D Sep": "{:.1f}%",
        "50D Pctl": "{:.1f}",
        "C-score": "{:.1f}",
    })
    .map(style_score, subset=["C-score"]),
    use_container_width=True,
)

st.divider()
st.markdown(
    """
    ### 계산 정의
    - **52W MDD**: 현재 종가 ÷ 최근 252거래일 최고 종가 − 1
    - **50D Sep**: 현재 종가 ÷ 50일 이동평균 − 1
    - **MDD Pctl**: 현재 MDD가 해당 자산 자신의 최근 N년 역사에서 얼마나 드물게 낮은지
    - **50D Pctl**: 현재 50일선 이격이 해당 자산 자신의 최근 N년 역사에서 얼마나 드물게 낮은지
    - **C-score**: `MDD percentile × 0.6 + 50D percentile × 0.4`
    - **E-score**: 12개월 수익률과 상방 50일 이격을 이용한 과열 참고지표

    **주의:** 자동 현금비중 규칙은 현재 전략 설계를 화면에 옮긴 *초안*입니다.
    지표 계산 자체와 현금비중 매매 규칙을 분리해 두었기 때문에,
    이후 백테스트 결과에 맞춰 현금 룰만 교체할 수 있습니다.
    """
)
