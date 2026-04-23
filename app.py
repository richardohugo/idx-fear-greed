"""
IDX Fear & Greed Index — Streamlit dashboard.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io, connect the repo
    3. Select app.py as the entry point
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from indicators import (
    IDX80_TICKERS,
    JCI,
    USDIDR,
    classify,
    compute_all,
    fetch_bond_proxy,
    fetch_history,
)
from news import aggregate_sentiment, fetch_headlines


# ---------------------------------------------------------------------------
# Page config + theming
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IDX Fear & Greed",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Bloomberg-terminal-inspired dark theme. Restrained, editorial, data-forward.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }
    .main {
        background-color: #0a0a0a;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .headline-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #888;
        margin-bottom: 0.2rem;
    }
    .headline-value {
        font-family: 'Fraunces', serif;
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }
    .headline-class {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 0.5rem;
    }
    .indicator-card {
        background: #111;
        border-left: 2px solid #333;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .indicator-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #888;
    }
    .indicator-score {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 600;
    }
    .indicator-desc {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #aaa;
    }
    .news-item {
        border-bottom: 1px solid #222;
        padding: 0.6rem 0;
        font-size: 0.85rem;
    }
    .news-meta {
        color: #666;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def color_for_score(score: float) -> str:
    """Red-to-green gradient across the fear-greed spectrum."""
    if score < 25:
        return "#d73027"  # deep red
    if score < 45:
        return "#fc8d59"  # orange
    if score < 55:
        return "#fee08b"  # amber
    if score < 75:
        return "#91cf60"  # light green
    return "#1a9850"      # deep green


def build_gauge(score: float, label: str) -> go.Figure:
    """Speedometer-style gauge using Plotly."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "font": {"family": "Fraunces, serif", "size": 72, "color": "#f5f5f5"},
                "suffix": "",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#444",
                    "tickfont": {"family": "JetBrains Mono", "size": 11, "color": "#888"},
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                },
                "bar": {"color": color_for_score(score), "thickness": 0.25},
                "bgcolor": "#0a0a0a",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#3a1414"},
                    {"range": [25, 45], "color": "#3a2614"},
                    {"range": [45, 55], "color": "#333014"},
                    {"range": [55, 75], "color": "#1f3a14"},
                    {"range": [75, 100], "color": "#143a1d"},
                ],
                "threshold": {
                    "line": {"color": "#f5f5f5", "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        height=360,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 2)  # 2 hours — reduce load on upstream sources
def load_market_data():
    jci = fetch_history([JCI], period="2y")[JCI]
    idx80 = fetch_history(IDX80_TICKERS, period="2y")
    usdidr = fetch_history([USDIDR], period="2y")[USDIDR]
    bond_series, bond_source = fetch_bond_proxy()
    return jci, idx80, usdidr, bond_series, bond_source


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
        '<div style="color:#666; font-size:0.85rem; letter-spacing:0.08em; '
        'text-transform:uppercase; margin-top:-0.5rem;">'
        "Real-time sentiment gauge for the Indonesian stock market"
        "</div>",
        unsafe_allow_html=True,
    )
with col_refresh:
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Load
with st.spinner("Fetching market data..."):
    try:
        jci, idx80, usdidr, bond_series, bond_source = load_market_data()
        indicators, headline_score = compute_all(jci, idx80, usdidr, bond_series, bond_source)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load market data: {e}")
        st.stop()

# Main row: gauge + headline
col_gauge, col_numbers = st.columns([3, 2])

with col_gauge:
    st.plotly_chart(build_gauge(headline_score, classify(headline_score)),
                    use_container_width=True, config={"displayModeBar": False})

with col_numbers:
    st.markdown("<br><br>", unsafe_allow_html=True)
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
        f'<div style="color:#666; font-size:0.75rem; margin-top:1rem; letter-spacing:0.1em;">'
        f'AS OF {datetime.now().strftime("%d %b %Y, %H:%M WIB")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Bond data source badge
    source_label = {
        "xbnd": "BOND: XBND.JK (Bahana Bond ETF)",
        "none": "BOND: unavailable — FX only",
    }.get(bond_source, "BOND: unknown")
    source_color = {
        "xbnd": "#1a9850",
        "none": "#fc8d59",
    }.get(bond_source, "#888")
    st.markdown(
        f'<div style="color:{source_color}; font-size:0.65rem; margin-top:0.3rem; '
        f'letter-spacing:0.1em;">{source_label}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Indicator breakdown
st.markdown("## Component breakdown")
st.markdown(
    '<div style="color:#666; font-size:0.8rem; margin-bottom:1rem;">'
    "Five indicators, equal-weighted. Each normalised against its own 2-year history."
    "</div>",
    unsafe_allow_html=True,
)

for ind in indicators:
    color = color_for_score(ind.score)
    st.markdown(
        f"""
        <div class="indicator-card" style="border-left-color:{color};">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div class="indicator-name">{ind.name}</div>
                <div class="indicator-score" style="color:{color};">{ind.score:.0f}</div>
            </div>
            <div class="indicator-desc">{ind.description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# News panel — aggregate only, no headline reproduction (ToS hygiene)
st.markdown("## News sentiment")
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
            <div class="indicator-score">{news_count}</div>
        </div>
        <div>
            <div class="indicator-name">Distribution</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#ccc; margin-top:0.3rem;">
                <span style="color:#1a9850;">↑ {pos_count}</span> &nbsp;
                <span style="color:#888;">· {neu_count}</span> &nbsp;
                <span style="color:#d73027;">↓ {neg_count}</span>
            </div>
        </div>
    </div>
    <div style="color:#888; font-size:0.75rem; max-width:700px; margin-bottom:0.5rem;">
        Aggregated VADER sentiment across Indonesian market headlines from the last 48 hours.
        Reference only — not folded into the headline gauge. VADER is English-biased;
        Indonesian-language items under-weighted.
    </div>
    <div style="color:#666; font-size:0.75rem;">
        → <a href="https://news.google.com/search?q=Indonesia+stock+market" target="_blank"
             style="color:#91cf60; text-decoration:none;">View headlines on Google News</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="color:#555; font-size:0.7rem; text-transform:uppercase;
                letter-spacing:0.15em; border-top:1px solid #222; padding-top:1rem;">
        Data: Yahoo Finance · Google News RSS &nbsp;·&nbsp;
        Not investment advice &nbsp;·&nbsp;
        Indicators inspired by CNN Fear &amp; Greed, adapted for IDX constraints
    </div>
    """,
    unsafe_allow_html=True,
)
