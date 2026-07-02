"""
IDX Fear & Greed Index — Streamlit dashboard.

Bloomberg-terminal-styled: true black, amber primary, mono type.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io, connect the repo
    3. Select app.py as the entry point
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from indicators import (
    IDX80_TICKERS,
    JCI,
    USDIDR,
    classify,
    compute_all,
    compute_history,
    fetch_bond_proxy,
    fetch_history,
)
from news import aggregate_sentiment, fetch_headlines

WIB = ZoneInfo("Asia/Jakarta")

# ---------------------------------------------------------------------------
# Bloomberg palette
# ---------------------------------------------------------------------------

BLACK = "#000000"
PANEL = "#0c0c0c"
LINE = "#1f1f1f"
AMBER = "#ff8000"          # Bloomberg signature orange-amber
AMBER_SOFT = "#ffb000"
WHITE = "#e6e6e6"
GRAY = "#8a8a8a"
FAINT = "#555555"
RED = "#ff433d"
ORANGE = "#ff8000"
YELLOW = "#ffd60a"
GREEN_SOFT = "#7ee081"
GREEN = "#00d97e"


st.set_page_config(
    page_title="IDX Fear & Greed",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }
    .stApp, .main {
        background-color: #000000;
    }
    .block-container {
        padding-top: 1.6rem;
        max-width: 1200px;
    }
    h1 {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #ff8000 !important;
        font-size: 1.6rem !important;
    }
    .stButton>button {
        background: #0c0c0c;
        color: #ff8000;
        border: 1px solid #ff8000;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.1em;
    }
    .stButton>button:hover {
        background: #ff8000;
        color: #000;
        border-color: #ff8000;
    }
    .tape {
        display: flex; gap: 2.2rem; flex-wrap: wrap;
        background: #0c0c0c; border: 1px solid #1f1f1f;
        border-top: 2px solid #ff8000;
        padding: 0.55rem 1rem; margin-bottom: 1.4rem;
        font-size: 0.78rem;
    }
    .tape .k { color: #ff8000; letter-spacing: 0.12em; }
    .tape .v { color: #e6e6e6; font-weight: 700; }
    .sec-head {
        display: flex; justify-content: space-between; align-items: baseline;
        background: #0c0c0c; border: 1px solid #1f1f1f;
        border-left: 3px solid #ff8000;
        padding: 0.5rem 0.9rem; margin: 1.6rem 0 0.8rem;
    }
    .sec-title {
        color: #ff8000; font-size: 0.8rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.22em;
    }
    .sec-meta {
        color: #8a8a8a; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.15em;
    }
    .headline-label {
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.2em; color: #8a8a8a; margin-bottom: 0.2rem;
    }
    .headline-value {
        font-family: 'Fraunces', serif;
        font-size: 4.5rem; font-weight: 800; line-height: 1; margin: 0;
    }
    .headline-class {
        font-size: 1rem; text-transform: uppercase;
        letter-spacing: 0.15em; margin-top: 0.5rem;
    }
    .indicator-card {
        background: #0c0c0c;
        border: 1px solid #1f1f1f;
        border-left: 2px solid #333;
        padding: 0.9rem 1.1rem 1.1rem;
        margin-bottom: 0.55rem;
    }
    .indicator-name {
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.18em; color: #8a8a8a;
    }
    .indicator-score {
        font-family: 'Fraunces', serif;
        font-size: 1.9rem; font-weight: 600;
    }
    .indicator-desc {
        font-size: 0.78rem; color: #999; margin-top: 0.15rem;
    }
    .chip {
        font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase;
        border: 1px solid; border-radius: 2px;
        padding: 0.1rem 0.45rem; margin-left: 0.6rem;
        vertical-align: middle;
    }
    .track {
        position: relative; height: 5px; margin-top: 0.7rem;
        background: linear-gradient(90deg,
            #ff433d 0%, #ff8000 30%, #ffd60a 50%, #7ee081 70%, #00d97e 100%);
        border-radius: 3px; opacity: 0.85;
    }
    .track .dot {
        position: absolute; top: -4.5px;
        width: 13px; height: 13px; border-radius: 50%;
        background: #e6e6e6; border: 3px solid #000;
        transform: translateX(-50%);
        box-shadow: 0 0 0 1px #444;
    }
    .track-labels {
        display: flex; justify-content: space-between;
        font-size: 0.6rem; color: #555; letter-spacing: 0.12em;
        margin-top: 0.35rem; text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def color_for_score(score: float) -> str:
    """Red-to-green across the fear-greed spectrum, terminal tones."""
    if score < 25:
        return RED
    if score < 45:
        return ORANGE
    if score < 55:
        return YELLOW
    if score < 75:
        return GREEN_SOFT
    return GREEN


def section(title: str, meta: str = "") -> None:
    st.markdown(
        f'<div class="sec-head"><span class="sec-title">{title}</span>'
        f'<span class="sec-meta">{meta}</span></div>',
        unsafe_allow_html=True,
    )


def spectrum_bar(score: float) -> str:
    """Full-width fear-greed spectrum with a marker at the current reading."""
    return f"""
    <div class="track" style="height:8px; margin-top:0.4rem;">
        <div class="dot" style="left:{score:.1f}%; top:-3.5px;"></div>
    </div>
    <div class="track-labels">
        <span>Extreme Fear</span><span>Fear</span><span>Neutral</span>
        <span>Greed</span><span>Extreme Greed</span>
    </div>
    """


def fmt_delta(pct: float) -> str:
    color = GREEN if pct >= 0 else RED
    arrow = "▲" if pct >= 0 else "▼"
    return f'<span style="color:{color};">{arrow} {pct:+.2f}%</span>'


def build_gauge(score: float, label: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "font": {"family": "Fraunces, serif", "size": 72, "color": WHITE},
                "valueformat": ".0f",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": FAINT,
                    "tickfont": {"family": "JetBrains Mono", "size": 11, "color": GRAY},
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                },
                "bar": {"color": color_for_score(score), "thickness": 0.25},
                "bgcolor": BLACK,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#3d120f"},
                    {"range": [25, 45], "color": "#3d2408"},
                    {"range": [45, 55], "color": "#3d3305"},
                    {"range": [55, 75], "color": "#1c3d18"},
                    {"range": [75, 100], "color": "#0e3d24"},
                ],
                "threshold": {
                    "line": {"color": AMBER, "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        height=340,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


def build_history_chart(hist: pd.DataFrame) -> go.Figure:
    """JCI level (left axis, filled area) vs Fear & Greed index (right axis,
    0-100 line) over the last year, daily. Amber line on black, terminal style."""
    months = hist.index.to_period("M").unique().to_timestamp()[::2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["jci"], name="JCI LEVEL",
        line=dict(color="#9a9a9a", width=1),
        fill="tozeroy", fillcolor="rgba(150,150,150,0.14)",
        hovertemplate="JCI %{y:,.0f}<extra></extra>",
    ))
    # extreme fear / extreme greed guides on the index axis
    for lvl in (25, 75):
        fig.add_trace(go.Scatter(
            x=[hist.index[0], hist.index[-1]], y=[lvl, lvl], yaxis="y2",
            mode="lines", line=dict(color="#3a3a3a", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["fng"], name="FEAR & GREED INDEX", yaxis="y2",
        line=dict(color=AMBER, width=1.8),
        hovertemplate="F&G %{y:.0f}<extra></extra>",
    ))
    jlo, jhi = float(hist["jci"].min()), float(hist["jci"].max())
    pad = (jhi - jlo) * 0.10
    fig.update_layout(
        paper_bgcolor=BLACK, plot_bgcolor=BLACK, height=430,
        margin=dict(l=10, r=10, t=24, b=10),
        font=dict(family="JetBrains Mono, monospace", size=11, color=GRAY),
        yaxis=dict(range=[jlo - pad, jhi + pad], gridcolor="#141414",
                   tickformat=",.0f", zeroline=False),
        yaxis2=dict(overlaying="y", side="right", range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100], showgrid=False,
                    zeroline=False),
        xaxis=dict(gridcolor="#111111", zeroline=False, tickangle=0,
                   tickfont=dict(size=9),
                   tickvals=months,
                   ticktext=[d.strftime("%b-%y").upper() for d in months]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14,
                    font=dict(size=10, color="#bbb")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#111", bordercolor=AMBER,
                        font=dict(family="JetBrains Mono", color=WHITE)),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 2)  # 2 hours — reduce load on upstream sources
def load_market_data():
    # 3 years so the 1-year history chart has a full percentile window
    # behind its earliest displayed day
    jci = fetch_history([JCI], period="3y")[JCI]
    idx80 = fetch_history(IDX80_TICKERS, period="3y")
    usdidr = fetch_history([USDIDR], period="3y")[USDIDR]
    bond_series, bond_source = fetch_bond_proxy()
    history = compute_history(jci, idx80, usdidr, days=252)
    return jci, idx80, usdidr, bond_series, bond_source, history


@st.cache_data(ttl=60 * 60 * 2)
def load_news():
    headlines = fetch_headlines()
    score, count = aggregate_sentiment(headlines)
    return headlines, score, count


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Header
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown("# IDX Fear & Greed")
    st.markdown(
        '<div style="color:#8a8a8a; font-size:0.75rem; letter-spacing:0.14em; '
        'text-transform:uppercase; margin-top:-0.4rem; margin-bottom:0.9rem;">'
        "Sentiment gauge · Indonesian stock market"
        "</div>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("↻ REFRESH", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Load
with st.spinner("Fetching market data..."):
    try:
        jci, idx80, usdidr, bond_series, bond_source, history = load_market_data()
        indicators, headline_score = compute_all(jci, idx80, usdidr, bond_series, bond_source)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load market data: {e}")
        st.stop()

# Terminal tape: market context at a glance
jci_last = float(jci.iloc[-1])
jci_chg = float(jci.pct_change().iloc[-1]) * 100 if len(jci) > 1 else 0.0
usd_last = float(usdidr.iloc[-1])
usd_chg = float(usdidr.pct_change().iloc[-1]) * 100 if len(usdidr) > 1 else 0.0
now_wib = datetime.now(WIB).strftime("%d %b %Y · %H:%M WIB")
st.markdown(
    f"""
    <div class="tape">
        <span><span class="k">JCI</span>&nbsp; <span class="v">{jci_last:,.0f}</span>
              &nbsp;{fmt_delta(jci_chg)}</span>
        <span><span class="k">USD/IDR</span>&nbsp; <span class="v">{usd_last:,.0f}</span>
              &nbsp;{fmt_delta(usd_chg)}</span>
        <span><span class="k">INDEX</span>&nbsp;
              <span class="v" style="color:{color_for_score(headline_score)};">
              {headline_score:.0f} · {classify(headline_score).upper()}</span></span>
        <span style="margin-left:auto; color:#555;">{now_wib}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main row: gauge + headline
col_gauge, col_numbers = st.columns([3, 2])

with col_gauge:
    st.plotly_chart(build_gauge(headline_score, classify(headline_score)),
                    use_container_width=True, config={"displayModeBar": False})

with col_numbers:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="headline-label">Current reading</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="headline-value" style="color:{color_for_score(headline_score)}">'
        f'{headline_score:.0f}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="headline-class" style="color:{color_for_score(headline_score)}">'
        f'{classify(headline_score)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:#8a8a8a; font-size:0.72rem; margin-top:0.9rem; letter-spacing:0.1em;">'
        f'0 = MAXIMUM FEAR &nbsp;·&nbsp; 100 = MAXIMUM GREED'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Bond data source badge
    source_label = {
        "xbnd": "BOND: XBND.JK (Bahana Bond ETF)",
        "none": "BOND: unavailable · FX only",
    }.get(bond_source, "BOND: unknown")
    source_color = {
        "xbnd": GREEN,
        "none": ORANGE,
    }.get(bond_source, GRAY)
    st.markdown(
        f'<div style="color:{source_color}; font-size:0.65rem; margin-top:0.4rem; '
        f'letter-spacing:0.1em;">{source_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(spectrum_bar(headline_score), unsafe_allow_html=True)

# Indicator breakdown
section("Component breakdown", "5 indicators · equal weight · 1y percentile")

for ind in indicators:
    color = color_for_score(ind.score)
    label = classify(ind.score).upper()
    st.markdown(
        f"""
        <div class="indicator-card" style="border-left-color:{color};">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div class="indicator-name">{ind.name}
                    <span class="chip" style="color:{color}; border-color:{color};">{label}</span>
                </div>
                <div class="indicator-score" style="color:{color};">{ind.score:.0f}</div>
            </div>
            <div class="indicator-desc">{ind.description}</div>
            <div class="track"><div class="dot" style="left:{ind.score:.1f}%;"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Historical context: JCI level vs the index, 1 year daily
section("JCI · Fear & Greed", "1Y · daily")
st.markdown(
    '<div style="color:#8a8a8a; font-size:0.75rem; margin-bottom:0.5rem;">'
    "Each day is scored with the same five indicators, using only data "
    "available on that day. Dotted lines mark extreme fear (25) and extreme greed (75)."
    "</div>",
    unsafe_allow_html=True,
)
if len(history) >= 30:
    st.plotly_chart(build_history_chart(history), use_container_width=True,
                    config={"displayModeBar": False})
else:
    st.caption("Not enough history yet for the 1-year chart.")

# News panel — aggregate only, no headline reproduction (ToS hygiene)
section("News sentiment", "48h · Google News RSS · VADER")
with st.spinner("Analysing headlines..."):
    try:
        headlines, news_score, news_count = load_news()
    except Exception as e:  # noqa: BLE001
        st.warning(f"News feed unavailable: {e}")
        headlines, news_score, news_count = [], 50.0, 0

news_color = color_for_score(news_score)

# Compute distribution stats without displaying any individual headline
if headlines:
    pos_count = sum(1 for h in headlines if h.score > 0.05)
    neg_count = sum(1 for h in headlines if h.score < -0.05)
    neu_count = news_count - pos_count - neg_count
else:
    pos_count = neg_count = neu_count = 0

st.markdown(
    f"""
    <div style="display:flex; gap:2.5rem; align-items:baseline; margin-bottom:1rem; flex-wrap:wrap;">
        <div>
            <div class="indicator-name">Sentiment score</div>
            <div class="indicator-score" style="color:{news_color};">{news_score:.0f}</div>
        </div>
        <div>
            <div class="indicator-name">Headlines analysed</div>
            <div class="indicator-score" style="color:{WHITE};">{news_count}</div>
        </div>
        <div>
            <div class="indicator-name">Distribution</div>
            <div style="font-size:0.9rem; color:#ccc; margin-top:0.3rem;">
                <span style="color:{GREEN};">▲ {pos_count}</span> &nbsp;
                <span style="color:{GRAY};">· {neu_count}</span> &nbsp;
                <span style="color:{RED};">▼ {neg_count}</span>
            </div>
        </div>
    </div>
    <div style="color:#8a8a8a; font-size:0.72rem; max-width:700px; margin-bottom:0.5rem;">
        Aggregated VADER sentiment across Indonesian market headlines from the last 48 hours.
        Reference only, not folded into the headline gauge. VADER is English-biased;
        Indonesian-language items under-weighted.
    </div>
    <div style="color:#666; font-size:0.72rem;">
        → <a href="https://news.google.com/search?q=Indonesia+stock+market" target="_blank"
             style="color:{AMBER}; text-decoration:none;">View headlines on Google News</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="color:#555; font-size:0.68rem; text-transform:uppercase;
                letter-spacing:0.15em; border-top:1px solid #1f1f1f; padding-top:1rem;">
        Data: Yahoo Finance · Google News RSS &nbsp;·&nbsp;
        Not investment advice &nbsp;·&nbsp;
        Indicators inspired by CNN Fear &amp; Greed, adapted for IDX constraints
    </div>
    """,
    unsafe_allow_html=True,
)
