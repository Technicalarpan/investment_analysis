import pandas as pd
import numpy as np
from data_engine import _detect_patterns

def run_backtest(df: pd.DataFrame, signal_threshold: float = 0.15, hold_days: int = 20) -> dict:
    """
    Backtest: Apply signal rules on real historical price data (yfinance).
    Scans every 10 bars, looks ahead `hold_days` bars to check if signal was correct.
    """
    if len(df) < 100:
        return {"error": "Need at least 100 bars for backtesting"}

    # Work on a copy with indicators already computed
    df = df.copy().dropna(subset=["Close"])
    total_bars = len(df)

    # Calculate approximate years of data
    try:
        days_span = (df.index[-1] - df.index[0]).days
        years_span = round(days_span / 365, 1)
        period_label = f"{years_span} year{'s' if years_span != 1.0 else ''}"
    except Exception:
        period_label = "the analysed period"

    trades = []
    # Scan every 10 bars (denser than before for more signals)
    window = 10

    for i in range(80, total_bars - hold_days, window):
        slice_df = df.iloc[:i].copy()
        if len(slice_df) < 60:
            continue

        # Use already-computed indicators — just detect patterns on slice
        try:
            ta_slice = _detect_patterns(slice_df)
        except Exception:
            continue

        score    = ta_slice.get("composite_score", 0.0)
        price_in = float(slice_df.iloc[-1]["Close"])

        if abs(score) < signal_threshold or price_in <= 0:
            continue  # no clear signal

        direction = "BUY" if score > 0 else "SELL"

        # Outcome: compare entry price to price after hold_days bars
        price_out = float(df.iloc[i + hold_days]["Close"])
        if price_out <= 0:
            continue

        ret = (price_out - price_in) / price_in
        # A trade is a "win" if BUY and price rose >0.5%, or SELL and price fell >0.5%
        success = (ret > 0.005 and direction == "BUY") or (ret < -0.005 and direction == "SELL")

        trades.append({
            "direction":  direction,
            "score":      round(score, 4),
            "price_in":   round(price_in, 2),
            "price_out":  round(price_out, 2),
            "return_pct": round(ret * 100, 2),
            "success":    success,
            "date_in":    str(slice_df.index[-1].date()),
        })

    if not trades:
        return {"error": "Not enough clear signals found in historical data. Try a longer period (2y or more)."}

    total       = len(trades)
    wins        = sum(1 for t in trades if t["success"])
    buy_trades  = [t for t in trades if t["direction"] == "BUY"]
    sell_trades = [t for t in trades if t["direction"] == "SELL"]
    avg_ret     = round(float(np.mean([t["return_pct"] for t in trades])), 2)
    win_rate    = round(wins / total * 100, 1)

    # Pick last 8 trades as sample (most recent)
    sample = trades[-8:]

    # Cumulative return of all trades
    cum_return = round(sum(t["return_pct"] for t in trades), 2)

    quality = "reliable" if win_rate >= 60 else "moderate" if win_rate >= 45 else "weak"

    return {
        "total_signals":  total,
        "winning_signals": wins,
        "win_rate":        win_rate,
        "avg_return_pct":  avg_ret,
        "cum_return_pct":  cum_return,
        "buy_signals":     len(buy_trades),
        "sell_signals":    len(sell_trades),
        "buy_win_rate":    round(sum(1 for t in buy_trades if t["success"]) / max(len(buy_trades), 1) * 100, 1),
        "sell_win_rate":   round(sum(1 for t in sell_trades if t["success"]) / max(len(sell_trades), 1) * 100, 1),
        "sample_trades":   sample,
        "period_label":    period_label,
        "hold_days":       hold_days,
        "message": (
            f"This signal worked {win_rate}% of the time over {period_label} "
            f"({total} trades, avg return {avg_ret:+.2f}% per {hold_days}-day hold). "
            f"Signal quality: {quality.upper()}."
        ),
    }