import uuid
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from src.models.hypothesis import Hypothesis, HypothesisStatus
from src.models.governance import GovernanceRecord, GovernanceStage
from src.models.experiment_queue import ExperimentJob, JobStatus
from src.data.edgar_client import (fetch_company_facts, extract_eps_history,
                                    compute_eps_surprise_proxy, KNOWN_EDGAR_GAPS)
# ── SP500 sample universe (top 50 by liquidity) ──────────────────────────────
SP500_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","LLY",
    "V","UNH","XOM","AVGO","MA","PG","HD","COST","JNJ","MRK",
    "ABBV","BAC","WMT","NFLX","CRM","AMD","PEP","KO","TMO","ORCL",
    "ACN","MCD","LIN","CSCO","DHR","NKE","TXN","NEE","PM","QCOM",
    "BMY","UPS","MS","INTC","RTX","AMGN","HON","LOW","IBM","GS"
]

def fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        volume = raw["Volume"]
    else:
        close = raw[["Close"]]
        volume = raw[["Volume"]]
    close = close.ffill().dropna(how="all", axis=1)
    volume = volume.ffill().dropna(how="all", axis=1)
    return close, volume

def compute_momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    if len(close) < lookback:
        return pd.Series(dtype=float)
    returns = close.iloc[-1] / close.iloc[-(lookback - skip)] - 1
    return returns.dropna()

def compute_earnings_revision_proxy(close: pd.DataFrame, volume: pd.DataFrame,
                                     lookback: int = 63) -> pd.Series:
    """
    Proxy for earnings revision using price-volume divergence.
    Real implementation would use analyst estimate data (future: EDGAR/Refinitiv).
    Proxy: abnormal volume on up days vs down days over lookback window.
    """
    if len(close) < lookback:
        return pd.Series(dtype=float)
    price_change = close.diff()
    up_vol = (volume.where(price_change > 0, 0)).iloc[-lookback:].sum()
    down_vol = (volume.where(price_change < 0, 0)).iloc[-lookback:].sum()
    total_vol = up_vol + down_vol
    revision_proxy = ((up_vol - down_vol) / total_vol.replace(0, np.nan)).dropna()
    return revision_proxy

def compute_volatility_compression(close: pd.DataFrame,
                                    short_window: int = 21,
                                    long_window: int = 63) -> pd.Series:
    if len(close) < long_window:
        return pd.Series(dtype=float)
    returns = close.pct_change().dropna()
    short_vol = returns.iloc[-short_window:].std() * np.sqrt(252)
    long_vol = returns.iloc[-long_window:].std() * np.sqrt(252)
    compression = long_vol - short_vol   # positive = vol compressing
    return compression.dropna()

def compute_institutional_accumulation(close: pd.DataFrame,
                                        volume: pd.DataFrame,
                                        lookback: int = 20) -> pd.Series:
    if len(close) < lookback + 1:
        return pd.Series(dtype=float)
    price_change = close.diff()
    signed_vol = volume * np.sign(price_change)
    obv_slope = signed_vol.iloc[-lookback:].sum()
    avg_vol = volume.iloc[-252:].mean().replace(0, np.nan)
    normalized = (obv_slope / avg_vol).dropna()
    return normalized

def compute_real_earnings_revision(tickers: list[str], verbose: bool = False) -> pd.Series:
    """
    Real earnings-trend signal sourced from SEC EDGAR XBRL data, replacing
    the price-volume proxy. Returns trailing-4-quarter EPS growth as a
    cross-sectional signal across the given tickers. Tickers with known
    structural reporting gaps (see KNOWN_EDGAR_GAPS) are skipped explicitly.
    """
    import time
    values = {}
    skipped = []
    for ticker in tickers:
        if ticker in KNOWN_EDGAR_GAPS:
            skipped.append((ticker, KNOWN_EDGAR_GAPS[ticker]))
            continue
        facts = fetch_company_facts(ticker)
        if not facts:
            skipped.append((ticker, "no EDGAR facts retrieved (CIK mapping or API failure)"))
            continue
        eps_history = extract_eps_history(facts)
        surprise = compute_eps_surprise_proxy(eps_history)
        if "error" not in surprise:
            values[ticker] = surprise["eps_growth_yoy_proxy"]
        else:
            skipped.append((ticker, surprise["error"]))
        time.sleep(0.15)  # stay well under SEC's 10 req/sec limit

    if verbose and skipped:
        print(f"  Earnings signal: {len(skipped)} ticker(s) excluded:")
        for t, reason in skipped:
            print(f"    {t}: {reason}")

    return pd.Series(values)

def build_composite_signal(close: pd.DataFrame,
                            volume: pd.DataFrame,
                            signal_components: list[dict],
                            use_real_earnings_data: bool = True) -> pd.Series:
    signals = {}
    for comp in signal_components:
        factor = comp["factor"]
        direction = 1 if comp["direction"] == "positive" else -1
        lb = comp.get("lookback_days", 63)

        if "momentum" in factor and "institutional" not in factor:
            s = compute_momentum(close, lookback=lb)
        elif "earnings" in factor or "revision" in factor:
            if use_real_earnings_data:
                s = compute_real_earnings_revision(close.columns.tolist())
            else:
                s = compute_earnings_revision_proxy(close, volume, lookback=lb)
        elif "volatility" in factor or "low_vol" in factor:
            s = compute_volatility_compression(close, short_window=21, long_window=63)
        elif "institutional" in factor or "flow" in factor:
            s = compute_institutional_accumulation(close, volume, lookback=lb)
        else:
            print(f"  UNHANDLED FACTOR: '{factor}' (no keyword match, skipping)")
            continue

        # Z-score each signal and apply direction, then store in the signals
        # dict for averaging into the composite. This step was missing —
        # s was being computed but never written anywhere.
        # Z-score each signal and apply direction, then store in the signals
        # dict for averaging into the composite. This step was missing —
        # s was being computed but never written anywhere.
        if s is None or len(s) == 0:
            print(f"  '{factor}': empty or None series, dropped")
        else:
            std = s.std()
            if std > 0:
                signals[factor] = direction * (s - s.mean()) / std
            else:
                print(f"  '{factor}': zero variance (std={std}), dropped — "
                      f"likely degenerate window (short_window == long_window?)")

    if not signals:
        print("  WARNING: no signal components matched, returning empty series")
        return pd.Series(dtype=float)

    signal_df = pd.DataFrame(signals)
    print(f"  Signal components before dropna: {list(signals.keys())}, shape={signal_df.shape}")
    signal_df = signal_df.dropna()
    print(f"  Signal components after dropna: shape={signal_df.shape}")

    if signal_df.empty:
        print("  WARNING: all rows dropped by dropna(), returning empty series")
        return pd.Series(dtype=float)

    composite = signal_df.mean(axis=1)
    return composite

def run_backtest(composite_signal: pd.Series,
                 close: pd.DataFrame,
                 holding_days: int = 21,
                 top_quantile: float = 0.2,
                 backtest_start: str = "2018-01-01") -> dict:
    """
    Simple long-only quantile backtest.
    Each month, go long top quintile by composite signal.
    Hold for holding_days, then rebalance.
    """
    tickers = composite_signal.index.tolist()
    if len(tickers) < 10:
        return {"error": "insufficient universe after signal computation"}

    # Select top quintile
    n_long = max(1, int(len(tickers) * top_quantile))
    long_tickers = composite_signal.nlargest(n_long).index.tolist()

    # Get forward returns for holding period
    available = [t for t in long_tickers if t in close.columns]
    if not available:
        return {"error": "no valid tickers after filtering"}

    portfolio_close = close[available]

    # Sample only at non-overlapping rebalance points (every holding_days)
    # to avoid compounding the same overlapping window repeatedly.
    rebalance_idx = portfolio_close.index[::holding_days]
    rebalance_close = portfolio_close.loc[rebalance_idx]

    fwd_returns = rebalance_close.pct_change().dropna()

    if len(fwd_returns) < 12:
        return {"error": "insufficient return history"}

    # Equal weight portfolio returns at each non-overlapping holding period
    portfolio_returns = fwd_returns.mean(axis=1)

    # Compute metrics
    n_periods_per_year = 252 / holding_days
    ann_return = portfolio_returns.mean() * n_periods_per_year
    ann_vol = portfolio_returns.std() * np.sqrt(n_periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside.std() * np.sqrt(n_periods_per_year) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    win_rate = (portfolio_returns > 0).mean()

    # OOS split (last 20% of data)
    split = int(len(portfolio_returns) * 0.8)
    oos_returns = portfolio_returns.iloc[split:]
    oos_ann_return = oos_returns.mean() * n_periods_per_year
    oos_ann_vol = oos_returns.std() * np.sqrt(n_periods_per_year)
    oos_sharpe = oos_ann_return / oos_ann_vol if oos_ann_vol > 0 else 0.0

    return {
        "sharpe_ratio": float(round(sharpe, 4)),
        "sortino_ratio": float(round(sortino, 4)),
        "calmar_ratio": float(round(calmar, 4)),
        "max_drawdown": float(round(max_dd, 4)),
        "annualized_return": float(round(ann_return, 4)),
        "win_rate": float(round(win_rate, 4)),
        "oos_sharpe": float(round(oos_sharpe, 4)),
        "n_long_positions": int(n_long),
        "backtest_start": backtest_start,
        "backtest_end": datetime.now(timezone.utc).strftime("%Y-%m-%d")
    }

def run_hypothesis_backtest(db: Session, hypothesis: Hypothesis) -> dict:
    print(f"  Fetching price data for {len(SP500_UNIVERSE)} tickers...")
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    close, volume = fetch_price_data(SP500_UNIVERSE, start_date, end_date)
    print(f"  Data fetched: {close.shape[0]} days × {close.shape[1]} tickers")

    print(f"  Computing composite signal...")
    composite = build_composite_signal(close, volume, hypothesis.signal_components)
    print(f"  Signal computed for {len(composite)} tickers")

    print(f"  Running backtest...")
    results = run_backtest(
        composite,
        close,
        holding_days=hypothesis.expected_holding_days or 21
    )

    if "error" in results:
        print(f"  Backtest failed: {results['error']}")
        return results

    # Write results back to hypothesis
    hypothesis.sharpe_ratio = results["sharpe_ratio"]
    hypothesis.sortino_ratio = results["sortino_ratio"]
    hypothesis.calmar_ratio = results["calmar_ratio"]
    hypothesis.max_drawdown = results["max_drawdown"]
    hypothesis.annualized_return = results["annualized_return"]
    hypothesis.win_rate = results["win_rate"]
    hypothesis.status = HypothesisStatus.VALIDATED
    hypothesis.backtest_period_start = datetime.strptime(results["backtest_start"], "%Y-%m-%d")
    hypothesis.backtest_period_end = datetime.strptime(results["backtest_end"], "%Y-%m-%d")

    # Advance governance stage
    gov = db.query(GovernanceRecord).filter_by(hypothesis_id=hypothesis.id).first()
    if gov:
        gov.advance_stage(
            GovernanceStage.STATISTICAL_REVIEW,
            notes=f"Backtest complete. Sharpe={results['sharpe_ratio']}, "
                  f"OOS Sharpe={results['oos_sharpe']}, "
                  f"MaxDD={results['max_drawdown']}"
        )

    # Update experiment job
    job = db.query(ExperimentJob).filter_by(hypothesis_id=hypothesis.id).first()
    if job:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(hypothesis)
    return results

def fetch_vix_data(start: str, end: str) -> pd.Series:
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_close = vix_raw["Close"].iloc[:, 0]
    else:
        vix_close = vix_raw["Close"]
    return vix_close.ffill()

def run_backtest_with_circuit_breaker(composite_signal: pd.Series,
                                        close: pd.DataFrame,
                                        vix: pd.Series,
                                        holding_days: int = 10,
                                        top_quantile: float = 0.2,
                                        vix_exit_threshold: float = 18.0,
                                        vix_reentry_threshold: float = 16.0,
                                        backtest_start: str = "2018-01-01") -> dict:
    """
    Simulates the actual circuit-breaker mechanism: enters positions at scheduled
    rebalance points, but checks VIX daily during the holding period. If VIX closes
    >= vix_exit_threshold on any day, liquidates at next-day open instead of waiting
    for the scheduled rebalance.
    """
    tickers = composite_signal.index.tolist()
    if len(tickers) < 10:
        return {"error": "insufficient universe after signal computation"}

    n_long = max(1, int(len(tickers) * top_quantile))
    long_tickers = composite_signal.nlargest(n_long).index.tolist()
    available = [t for t in long_tickers if t in close.columns]
    if not available:
        return {"error": "no valid tickers after filtering"}

    portfolio_close = close[available]

    # Align VIX to the same trading calendar
    vix_aligned = vix.reindex(portfolio_close.index).ffill()

    rebalance_idx = portfolio_close.index[::holding_days]

    period_returns = []
    circuit_breaker_triggers = 0
    total_periods = 0

    for i in range(len(rebalance_idx) - 1):
        entry_date = rebalance_idx[i]
        scheduled_exit_date = rebalance_idx[i + 1]

        # Get the trading day slice for this holding period
        period_mask = (portfolio_close.index > entry_date) & (portfolio_close.index <= scheduled_exit_date)
        period_dates = portfolio_close.index[period_mask]

        if len(period_dates) == 0:
            continue
        total_periods += 1

        # Check VIX daily within this holding period for a breach
        actual_exit_date = scheduled_exit_date
        triggered = False
        for d in period_dates:
            if d in vix_aligned.index and not pd.isna(vix_aligned.loc[d]):
                if vix_aligned.loc[d] >= vix_exit_threshold:
                    actual_exit_date = d
                    triggered = True
                    circuit_breaker_triggers += 1
                    break

        entry_prices = portfolio_close.loc[entry_date]
        exit_prices = portfolio_close.loc[actual_exit_date]
        period_return = ((exit_prices / entry_prices) - 1).mean()
        period_returns.append(period_return)

    if len(period_returns) < 12:
        return {"error": "insufficient return history"}

    portfolio_returns = pd.Series(period_returns)

    n_periods_per_year = 252 / holding_days
    ann_return = portfolio_returns.mean() * n_periods_per_year
    ann_vol = portfolio_returns.std() * np.sqrt(n_periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    downside = portfolio_returns[portfolio_returns < 0]
    downside_vol = downside.std() * np.sqrt(n_periods_per_year) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (portfolio_returns > 0).mean()

    return {
        "sharpe_ratio": float(round(sharpe, 4)),
        "sortino_ratio": float(round(sortino, 4)),
        "calmar_ratio": float(round(calmar, 4)),
        "max_drawdown": float(round(max_dd, 4)),
        "annualized_return": float(round(ann_return, 4)),
        "win_rate": float(round(win_rate, 4)),
        "n_long_positions": int(n_long),
        "total_periods": int(total_periods),
        "circuit_breaker_triggers": int(circuit_breaker_triggers),
        "circuit_breaker_trigger_rate": float(round(circuit_breaker_triggers / total_periods, 4)) if total_periods > 0 else 0.0,
        "backtest_start": backtest_start,
        "backtest_end": datetime.now(timezone.utc).strftime("%Y-%m-%d")
    }