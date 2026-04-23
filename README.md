# IDX Fear & Greed Index

A real-time sentiment gauge for the Indonesian stock market (IDX), inspired by CNN's Fear & Greed Index but adapted for IDX data constraints. Built in Python + Streamlit, designed to deploy for free on Streamlit Community Cloud directly from GitHub.

![Dashboard preview](https://via.placeholder.com/800x400/0a0a0a/1a9850?text=IDX+Fear+%26+Greed)

## What it measures

Five indicators, equal-weighted, each normalised 0–100 against its own 2-year history:

| # | Indicator | What it captures | Data |
|---|-----------|------------------|------|
| 1 | **Price Momentum** | JCI vs 125-day moving average | `^JKSE` |
| 2 | **Price Strength** | % of IDX80 near 52w highs vs lows | IDX80 constituents |
| 3 | **Price Breadth** | 10-day EMA of advance/decline ratio | IDX80 constituents |
| 4 | **Market Volatility** | EWMA vol (λ=0.94 RiskMetrics), inverted | `^JKSE` |
| 5 | **Safe-Haven Demand** | 80% JCI–XBND return spread + 20% USD/IDR move | `^JKSE`, `XBND.JK`, `IDR=X` |

### v2 quant improvements

- **EWMA volatility** (RiskMetrics λ=0.94) replaces rolling 30-day std. Responds to regime shifts without the mechanical "forgetting" at day 31.
- **1-year rolling percentile** (was 2-year). Responsive to regime change — motivated by Farrell & O'Connor (2025) finding that FG-Index predictive power is time-varying.
- **Winsorisation at 2.5/97.5** before ranking. One black-swan day no longer pins percentiles at 0 for months.
- **Safe-haven redesigned**: 80% bond leg (JCI vs XBND.JK ETF return spread) + 20% USD/IDR. Falls back to FX-only if XBND data unavailable. A source badge in the UI shows which path is live.

### Compliance-conscious design

This app deliberately uses **only yfinance + Google News RSS** for data. It does **not** use:

- TradingView scrapers (`tvDatafeed`, `tvdatafeed`) — their ToS explicitly prohibits automated access and violations can result in account bans. Bad trade-off for a working finance professional.
- Investing.com scrapers (`investpy`) — blocks third-party clients via Cloudflare, returns 403.

This means the bond signal is weaker than the ideal INDOGB 10Y yield (XBND.JK is a price-return proxy, not a yield signal), but the app won't compromise any personal account.

The news panel shows **aggregate sentiment only** — no individual headlines are reproduced, respecting Google News ToS concerns around derivative works. Users are linked out to Google News for the actual headlines.

News sentiment from Google News RSS is shown alongside for reference but **not folded into the headline gauge** — VADER is English-biased and would systematically under-weight Indonesian-language headlines.

## Why only 5 indicators (not CNN's 7)?

IDX lacks liquid analogs for two of CNN's components:

- **Put/Call ratio** — IDX options (KBIE/LQ45 options) are too thinly traded and unavailable on yfinance.
- **Junk bond demand** — No liquid HY vs IG spread index on Indonesian corporate bonds comparable to US HYG/LQD.

Rather than fabricate weak proxies, the index sticks to signals that can be computed reliably from free data.

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/idx-fear-greed.git
cd idx-fear-greed
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repo and branch, set the main file to `app.py`.
4. Click **Deploy**. You'll get a public URL like `https://idx-fear-greed.streamlit.app`.

No API keys or secrets needed — everything uses free public data.

## Methodology notes

- **Self-calibrating normalisation.** Rather than hard-coded thresholds, each indicator is percentile-ranked against its own rolling 2-year distribution. This avoids the common failure mode of Fear & Greed clones drifting as the underlying volatility regime shifts.
- **Data cached for 30 minutes.** yfinance rate-limits aggressively; the cache prevents a refresh button from hammering the endpoint.
- **IDX80 list is static.** IDX reviews IDX80 constituents quarterly (Feb / May / Aug / Nov). The list in `src/indicators.py` should be refreshed after each review. Current list is effective 2 Feb – 30 Apr 2026 (source: Kontan).

## Known limitations (and what to fix next)

1. **News sentiment is English-biased.** VADER was trained on English social media. Replace with a multilingual model like `cardiffnlp/twitter-xlm-roberta-base-sentiment` or Indonesian FinBERT once you need news to count toward the headline score.
2. **No volume component.** Indonesian retail flow is a dominant market force; a turnover-based indicator (e.g. JCI volume vs 50-day average) would capture it. Deferred to v2.
3. **No Factiva integration yet.** The `src/news.py` module has a clean interface (`fetch_headlines() -> list[Headline]`) — a Factiva fetcher can be dropped in alongside RSS.
4. **Volatility is realised, not implied.** IDX has no VIX. Realised vol is the honest substitute; users should read it as "how bumpy was it" not "how bumpy does the market expect it to be."

## Project structure

```
idx-fear-greed/
├── app.py                  # Streamlit UI
├── src/
│   ├── indicators.py       # 5 market indicators + normalisation
│   └── news.py             # Google News RSS + VADER
├── requirements.txt
├── .streamlit/config.toml  # Dark theme
└── README.md
```

## License

MIT. Not investment advice.
