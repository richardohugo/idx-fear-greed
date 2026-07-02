"""
IDX Fear & Greed indicators — v2.2.

Changes from v2.1:
    * IDX80 constituents updated to the BEI evaluation effective 4 May 2026
      (in: BKSL, CBDK, DEWA, GGRM, TPIA; out: BREN, BTPS, DSSA, MTEL, NCKL)
    * compute_history(): point-in-time daily history of the headline index,
      powering the JCI vs Fear & Greed 1-year chart

Changes from v2:
    * Bond proxy disabled — IDX bond ETFs on yfinance (XBND, XISB, XMGB)
      are too thin/gappy to produce reliable daily signals. Safe-haven
      runs FX-only until a clean INDOGB 10Y yield source is wired in
      (Bloomberg CSV export, PHEI API, etc).

Kept from v2:
    * Volatility uses EWMA (lambda=0.94 RiskMetrics), not rolling std
    * Normalisation uses 1-year rolling percentile (not 2-year) — responsive to regime
    * Values are winsorised at [2.5%, 97.5%] before ranking

Design note — no TradingView/Investing.com scrapers:
    TradingView's ToS explicitly prohibits automated access. Investing.com
    blocks third-party clients via Cloudflare. We deliberately stick to
    yfinance only to avoid ban risk on the user's personal accounts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# IDX80 constituents used for breadth / strength calculations.
# Source: BEI evaluation effective 4 May 2026 (next evaluation August 2026);
# in: BKSL, CBDK, DEWA, GGRM, TPIA / out: BREN, BTPS, DSSA, MTEL, NCKL.
# https://www.bareksa.com/berita/saham/2026-04-24/idx30-lq45-hingga-idx80-dirombak-bei-ini-daftar-saham-masuk-dan-keluar-mei-2026
IDX80_TICKERS = [
    "AADI.JK", "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK",
    "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK", "BBNI.JK",
    "BBRI.JK", "BBTN.JK", "BKSL.JK", "BMRI.JK", "BRMS.JK", "BRPT.JK",
    "BSDE.JK", "BUKA.JK", "BUMI.JK", "CBDK.JK", "CMRY.JK", "CPIN.JK",
    "CTRA.JK", "CUAN.JK", "DEWA.JK", "DSNG.JK", "ELSA.JK", "EMTK.JK",
    "ENRG.JK", "ERAA.JK", "ESSA.JK", "EXCL.JK", "GGRM.JK", "GOTO.JK",
    "HEAL.JK", "HRTA.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK",
    "INDY.JK", "INKP.JK", "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK",
    "JSMR.JK", "KIJA.JK", "KLBF.JK", "KPIG.JK", "MAPA.JK", "MAPI.JK",
    "MBMA.JK", "MDKA.JK", "MEDC.JK", "MIKA.JK", "MYOR.JK", "PANI.JK",
    "PGAS.JK", "PGEO.JK", "PNLF.JK", "PTBA.JK", "PTRO.JK", "PWON.JK",
    "RAJA.JK", "RATU.JK", "SCMA.JK", "SIDO.JK", "SMGR.JK", "SMRA.JK",
    "SSIA.JK", "TAPG.JK", "TLKM.JK", "TOWR.JK", "TPIA.JK", "UNTR.JK",
    "UNVR.JK", "WIFI.JK",
]

JCI = "^JKSE"
USDIDR = "IDR=X"

LOOKBACK_DAYS = 252
WINSOR_LOW = 0.025
WINSOR_HIGH = 0.975
EWMA_LAMBDA = 0.94


@dataclass
class Indicator:
    name: str
    value: float
    score: float
    description: str


def _winsorise(s: pd.Series, low: float = WINSOR_LOW, high: float = WINSOR_HIGH) -> pd.Series:
    s = s.dropna()
    if len(s) < 20:
        return s
    lo, hi = s.quantile(low), s.quantile(high)
    return s.clip(lower=lo, upper=hi)


def _percentile_score(series: pd.Series, current: float,
                      lookback: int = LOOKBACK_DAYS) -> float:
    window = series.iloc[-lookback:].dropna()
    if len(window) < 20:
        return 50.0
    winsorised = _winsorise(window)
    lo, hi = winsorised.min(), winsorised.max()
    clipped = np.clip(current, lo, hi)
    rank = (winsorised < clipped).sum() / len(winsorised)
    return float(np.clip(rank * 100, 0, 100))


def _invert(score: float) -> float:
    return 100.0 - score


def fetch_history(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    data = yf.download(
        tickers, period=period, interval="1d",
        progress=False, auto_adjust=True, threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close.dropna(how="all")


def fetch_bond_proxy() -> tuple[pd.Series, str]:
    """Bond proxy deliberately disabled.

    IDX-listed bond ETFs (XBND, XISB, XMGB) are too thinly traded on
    yfinance to produce a reliable daily signal — prices are often stale
    prints between sporadic trades, and at least one (XISB) showed a
    spurious ~200% move from a data artifact in March 2026. The safe-haven
    indicator runs FX-only until a clean INDOGB 10Y yield source can be
    sourced from Bloomberg or PHEI.
    """
    return pd.Series(dtype=float), "none"


def momentum_indicator(jci_close: pd.Series) -> Indicator:
    ma125 = jci_close.rolling(125).mean()
    gap = (jci_close - ma125) / ma125 * 100
    current = gap.iloc[-1]
    score = _percentile_score(gap, current)
    return Indicator(
        name="Price Momentum",
        value=current,
        score=score,
        description=f"JCI is {current:+.2f}% vs its 125-day moving average",
    )


def strength_indicator(idx80_close: pd.DataFrame) -> Indicator:
    window = idx80_close.iloc[-252:]
    today = window.iloc[-1]
    high_52w = window.max()
    low_52w = window.min()
    near_high = (today >= high_52w * 0.98).sum()
    near_low = (today <= low_52w * 1.02).sum()
    total = len(today.dropna())
    if total == 0:
        return Indicator("Price Strength", 0, 50, "No IDX80 data")
    net = (near_high - near_low) / total
    score = float(np.clip((net + 1) / 2 * 100, 0, 100))
    return Indicator(
        name="Price Strength",
        value=near_high - near_low,
        score=score,
        description=f"{near_high} stocks near 52w highs, {near_low} near 52w lows (of {total})",
    )


def breadth_indicator(idx80_close: pd.DataFrame) -> Indicator:
    returns = idx80_close.pct_change()
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)
    total = advances + declines
    ad_ratio = (advances - declines) / total.replace(0, np.nan)
    ad_smoothed = ad_ratio.ewm(span=10, adjust=False).mean()
    current = ad_smoothed.iloc[-1]
    score = _percentile_score(ad_smoothed, current)
    return Indicator(
        name="Price Breadth",
        value=current,
        score=score,
        description=f"10-day smoothed A/D ratio: {current:+.3f}",
    )


def volatility_indicator(jci_close: pd.Series) -> Indicator:
    """EWMA vol per RiskMetrics, lambda=0.94."""
    returns = jci_close.pct_change().dropna()
    ewma_var = returns.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).var()
    ewma_vol = np.sqrt(ewma_var * 252) * 100
    current = ewma_vol.iloc[-1]
    raw = _percentile_score(ewma_vol, current)
    score = _invert(raw)
    return Indicator(
        name="Market Volatility",
        value=current,
        score=score,
        description=f"JCI EWMA vol (λ=0.94): {current:.2f}% annualised (higher = more fear)",
    )


def safe_haven_indicator(jci_close: pd.Series,
                         usdidr_close: pd.Series,
                         bond_series: pd.Series,
                         bond_source: str) -> Indicator:
    """Safe-haven = USD/IDR 20d move, inverted.

    Rising USD/IDR (weakening rupiah) = domestic capital flight into USD =
    FEAR. Falling USD/IDR (strengthening rupiah) = capital returning =
    GREED.

    Bond leg disabled — see fetch_bond_proxy() docstring for rationale.
    The bond_series and bond_source arguments are accepted but ignored;
    kept in the signature for forward compatibility when a proper INDOGB
    10Y feed is wired in.
    """
    fx_ret20 = usdidr_close.pct_change(20) * 100
    fx_current = fx_ret20.iloc[-1]
    fx_raw = _percentile_score(fx_ret20, fx_current)
    fx_score = _invert(fx_raw)

    desc = f"USD/IDR 20d: {fx_current:+.2f}% [FX only — bond leg disabled]"

    return Indicator(
        name="Safe-Haven Demand",
        value=fx_score,
        score=fx_score,
        description=desc,
    )


def compute_all(jci: pd.Series,
                idx80: pd.DataFrame,
                usdidr: pd.Series,
                bond_series: pd.Series,
                bond_source: str) -> tuple[list[Indicator], float]:
    inds = [
        momentum_indicator(jci),
        strength_indicator(idx80),
        breadth_indicator(idx80),
        volatility_indicator(jci),
        safe_haven_indicator(jci, usdidr, bond_series, bond_source),
    ]
    headline = float(np.mean([i.score for i in inds]))
    return inds, headline


def classify(score: float) -> str:
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


# ---------------------------------------------------------------------------
# Historical series (for the JCI vs Fear & Greed chart)
# ---------------------------------------------------------------------------

def _percentile_series(s: pd.Series, lookback: int = LOOKBACK_DAYS) -> pd.Series:
    """Rolling version of _percentile_score: for each day, the score of that
    day's value inside its own trailing window, winsorised the same way.
    Each day uses only data available on that day, so there is no look-ahead."""
    def f(win: np.ndarray) -> float:
        if np.isnan(win[-1]):
            return np.nan
        w = win[~np.isnan(win)]
        if len(w) < 20:
            return np.nan
        lo, hi = np.quantile(w, WINSOR_LOW), np.quantile(w, WINSOR_HIGH)
        w = np.clip(w, lo, hi)
        cur = np.clip(win[-1], lo, hi)
        return float(np.clip((w < cur).sum() / len(w) * 100, 0, 100))
    return s.rolling(lookback, min_periods=20).apply(f, raw=True)


def compute_history(jci: pd.Series, idx80: pd.DataFrame, usdidr: pd.Series,
                    days: int = 252) -> pd.DataFrame:
    """Daily history of the headline index over the last `days` trading days.

    Applies the same five formulas as the live indicators, day by day,
    point-in-time. Needs roughly 3 years of price history to produce a full
    1-year series (125d MA plus a 252d percentile window behind the earliest
    displayed day). Returns a DataFrame with columns ['jci', 'fng'].
    """
    # 1. momentum: JCI gap vs 125d MA, percentile-ranked
    ma125 = jci.rolling(125).mean()
    gap = (jci - ma125) / ma125 * 100
    s_mom = _percentile_series(gap)

    # 2. strength: 52w highs vs lows across IDX80 (directly scored, no rank)
    hi52 = idx80.rolling(252, min_periods=1).max()
    lo52 = idx80.rolling(252, min_periods=1).min()
    near_hi = (idx80 >= hi52 * 0.98).sum(axis=1)
    near_lo = (idx80 <= lo52 * 1.02).sum(axis=1)
    total = idx80.notna().sum(axis=1).replace(0, np.nan)
    s_str = (((near_hi - near_lo) / total) + 1) / 2 * 100

    # 3. breadth: 10d smoothed advance/decline ratio, percentile-ranked
    rets = idx80.pct_change()
    adv = (rets > 0).sum(axis=1)
    dec = (rets < 0).sum(axis=1)
    ad = (adv - dec) / (adv + dec).replace(0, np.nan)
    s_brd = _percentile_series(ad.ewm(span=10, adjust=False).mean())

    # 4. volatility: JCI EWMA vol, percentile-ranked and inverted
    r = jci.pct_change()
    evol = np.sqrt(r.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).var() * 252) * 100
    s_vol = 100 - _percentile_series(evol)

    # 5. safe haven: USD/IDR 20d move, inverted, aligned to JCI trading days
    fx = usdidr.pct_change(20) * 100
    s_fx = (100 - _percentile_series(fx)).reindex(jci.index).ffill()

    scores = pd.concat(
        {"momentum": s_mom, "strength": s_str, "breadth": s_brd,
         "volatility": s_vol, "safe_haven": s_fx}, axis=1,
    ).reindex(jci.index)
    fng = scores.mean(axis=1, skipna=False)      # require all five components
    out = pd.DataFrame({"jci": jci, "fng": fng}).dropna()
    return out.tail(days)