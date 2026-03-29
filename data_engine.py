import time
import numpy as np
import pandas as pd
from typing import Optional

# optional imports
try:
    import yfinance as yf
    YF_OK = True
except:
    YF_OK = False

try:
    import pandas_ta as ta
    TA_OK = True
except:
    TA_OK = False
    
_MAX_RETRIES = 3
_MIN_ROWS    = 50

# Full NIFTY 50 stocks for scanner
NIFTY_50 = {
    "Reliance Industries":  "RELIANCE.NS",
    "TCS":                  "TCS.NS",
    "HDFC Bank":            "HDFCBANK.NS",
    "Infosys":              "INFY.NS",
    "ICICI Bank":           "ICICIBANK.NS",
    "Kotak Mahindra Bank":  "KOTAKBANK.NS",
    "Axis Bank":            "AXISBANK.NS",
    "SBI":                  "SBIN.NS",
    "Bajaj Finance":        "BAJFINANCE.NS",
    "LT":                   "LT.NS",
    "HUL":                  "HINDUNILVR.NS",
    "Wipro":                "WIPRO.NS",
    "HCL Technologies":     "HCLTECH.NS",
    "Tata Motors":          "TATAMOTORS.NS",
    "Maruti Suzuki":        "MARUTI.NS",
    "Asian Paints":         "ASIANPAINT.NS",
    "Sun Pharma":           "SUNPHARMA.NS",
    "Adani Enterprises":    "ADANIENT.NS",
    "ITC":                  "ITC.NS",
    "Bharti Airtel":        "BHARTIARTL.NS",
    "M&M":                  "M&M.NS",
    "Power Grid":           "POWERGRID.NS",
    "NTPC":                 "NTPC.NS",
    "UltraTech Cement":     "ULTRACEMCO.NS",
    "Titan":                "TITAN.NS",
}

# Subset for scanner (quick scan, configurable)
SCANNER_STOCKS = {k: v for k, v in list(NIFTY_50.items())[:15]}


def _stub_dataframe(n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    ret   = rng.normal(0.0004, 0.015, n)
    close = 2800.0 * np.exp(np.cumsum(ret))
    high  = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = close * (1 + rng.normal(0, 0.006, n))
    vol   = rng.integers(5_000_000, 30_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df.index.name = "Date"
    return df


def _fetch_yfinance(ticker: str, period: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    if not YF_OK:
        return None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                return df
            raise ValueError("Empty response")
        except Exception:
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]

    # MACD
    if TA_OK:
        try:
            macd_df = ta.macd(c, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                cols = macd_df.columns.tolist()
                df["MACD"]        = macd_df[cols[0]].values
                df["MACD_hist"]   = macd_df[cols[1]].values
                df["MACD_signal"] = macd_df[cols[2]].values
        except Exception:
            pass

    if "MACD" not in df.columns:
        ema_f = c.ewm(span=12, adjust=False).mean()
        ema_s = c.ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema_f - ema_s
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # RSI
    if TA_OK:
        try:
            rsi_s = ta.rsi(c, length=14)
            if rsi_s is not None:
                df["RSI"] = rsi_s.values
        except Exception:
            pass

    if "RSI" not in df.columns:
        delta  = c.diff()
        gain   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger Bands
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_middle"]= sma20
    df["BB_lower"] = sma20 - 2 * std20
    bw             = df["BB_upper"] - df["BB_lower"]
    df["BB_pct"]   = (c - df["BB_lower"]) / bw.replace(0, np.nan)

    # SMAs
    df["SMA_50"]  = c.rolling(50).mean()
    df["SMA_200"] = c.rolling(200).mean()

    # ATR
    hl           = df["High"] - df["Low"]
    hc           = (df["High"] - c.shift(1)).abs()
    lc           = (df["Low"]  - c.shift(1)).abs()
    true_range   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"]    = true_range.ewm(alpha=1/14, adjust=False).mean()

    # Volume MA
    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    # Stochastic (new)
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # OBV (new)
    obv = (np.sign(c.diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv

    return df


def _detect_patterns(df: pd.DataFrame) -> dict:
    if len(df) < _MIN_ROWS:
        return {
            "signal_label": "NEUTRAL", "composite_score": 0.0,
            "confidence_pct": 0.0, "patterns": [],
            "current_price": None, "atr": None,
            "suggested_stop_loss": None, "suggested_take_profit": None,
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]
    pats: list[dict] = []

    def g(row, col):
        v = row.get(col, np.nan)
        return np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    # 1. MACD crossover
    mn, ms = g(last,"MACD"), g(last,"MACD_signal")
    pm, ps = g(prev,"MACD"), g(prev,"MACD_signal")
    if not any(np.isnan(x) for x in [mn, ms, pm, ps]):
        if   pm < ps and mn >= ms: sc, det = 0.8,  f"Bullish MACD crossover — MACD ({mn:.3f}) just crossed above Signal ({ms:.3f}). Upward momentum building."
        elif pm > ps and mn <= ms: sc, det = -0.8, f"Bearish MACD crossover — MACD ({mn:.3f}) just crossed below Signal ({ms:.3f}). Downward momentum."
        else:
            direction = "above" if mn > ms else "below"
            sc, det = (0.2 if mn > ms else -0.2), f"MACD ({mn:.3f}) is {direction} Signal ({ms:.3f}). No fresh crossover today."
    else:
        sc, det = 0.0, "MACD data unavailable."
    pats.append({"name": "MACD Crossover", "score": sc, "detail": det})

    # 2. RSI
    rsi = g(last, "RSI")
    if not np.isnan(rsi):
        if   rsi <= 35: sc, det = min(1.0, 0.5+(35-rsi)/20),  f"RSI OVERSOLD at {rsi:.1f} — stock may bounce from mean-reversion buying."
        elif rsi >= 65: sc, det = -min(1.0, 0.5+(rsi-65)/20), f"RSI OVERBOUGHT at {rsi:.1f} — short-term selling pressure likely."
        else:           sc, det = 0.0, f"RSI at {rsi:.1f} — neutral zone, no extreme reading."
    else:
        sc, det = 0.0, "RSI data unavailable."
    pats.append({"name": "RSI Extreme", "score": sc, "detail": det})

    # 3. Golden / Death Cross
    s50, s200 = g(last,"SMA_50"), g(last,"SMA_200")
    p50, p200 = g(prev,"SMA_50"), g(prev,"SMA_200")
    if not any(np.isnan(x) for x in [s50, s200, p50, p200]):
        cur_above  = s50 > s200
        prev_above = p50 > p200
        if   not prev_above and cur_above:  sc, det = 1.0,  f"FRESH GOLDEN CROSS — SMA-50 ({s50:.1f}) just crossed above SMA-200 ({s200:.1f}). Strong bull signal."
        elif prev_above and not cur_above:  sc, det = -1.0, f"FRESH DEATH CROSS — SMA-50 ({s50:.1f}) just crossed below SMA-200 ({s200:.1f}). Strong bear signal."
        elif cur_above:                     sc, det = 0.4,  f"SMA-50 ({s50:.1f}) is above SMA-200 ({s200:.1f}). Bullish long-term structure intact."
        else:                               sc, det = -0.4, f"SMA-50 ({s50:.1f}) is below SMA-200 ({s200:.1f}). Bearish long-term structure."
    else:
        sc, det = 0.0, "SMA-200 unavailable — need more data."
    pats.append({"name": "Golden/Death Cross", "score": sc, "detail": det})

    # 4. Bollinger Band breakout
    cl, bu, bl, bm = g(last,"Close"), g(last,"BB_upper"), g(last,"BB_lower"), g(last,"BB_middle")
    vm, vma        = g(last,"Volume"), g(last,"Volume_MA")
    if not any(np.isnan(x) for x in [cl, bu, bl, bm]):
        bw_ratio = (bu - bl) / bm if bm != 0 else 1.0
        vr       = vm / vma if (not np.isnan(vma) and vma > 0) else 1.0
        if   bw_ratio < 0.04: sc, det = 0.0, f"BOLLINGER SQUEEZE (band width {bw_ratio:.2%}) — volatility is coiling. Large directional move imminent."
        elif cl > bu:
            if vr >= 1.5:  sc, det = 0.9, f"CONFIRMED BREAKOUT above upper BB ₹{bu:.1f} with {vr:.1f}× volume. High-conviction bullish signal."
            else:           sc, det = 0.5, f"Price above upper BB ₹{bu:.1f} but volume ({vr:.1f}×) is weak — needs confirmation."
        elif cl < bl:
            if vr >= 1.5:  sc, det = -0.9, f"CONFIRMED BREAKDOWN below lower BB ₹{bl:.1f} with {vr:.1f}× volume. High-conviction bearish."
            else:           sc, det = -0.5, f"Price below lower BB ₹{bl:.1f} but volume is low — potential oversold bounce."
        else:               sc, det = 0.0, f"Price (₹{cl:.1f}) is within Bollinger Bands. No breakout."
    else:
        sc, det = 0.0, "Bollinger Band data unavailable."
    pats.append({"name": "BB Breakout", "score": sc, "detail": det})

    # 5. Volume spike
    pc = g(prev, "Close")
    if not any(np.isnan(x) for x in [cl, pc, vm, vma]) and vma > 0:
        vr2  = vm / vma
        chg  = (cl - pc) / pc * 100
        if vr2 >= 1.5:
            sc  = 0.6 if chg > 0 else -0.6
            det = f"Volume spike {vr2:.1f}× the 20-day average on a {chg:+.2f}% price move. Likely institutional activity."
        else:
            sc, det = 0.0, f"Volume is normal ({vr2:.1f}× 20-day avg). No unusual activity."
    else:
        sc, det = 0.0, "Volume data unavailable."
    pats.append({"name": "Volume Spike", "score": sc, "detail": det})

    # Weighted composite
    weights = {
        "MACD Crossover":   0.30,
        "RSI Extreme":      0.20,
        "Golden/Death Cross": 0.20,
        "BB Breakout":      0.20,
        "Volume Spike":     0.10,
    }
    comp  = sum(p["score"] * weights.get(p["name"], 0.1) for p in pats) / sum(weights.values())
    label = "BULLISH" if comp >= 0.3 else ("BEARISH" if comp <= -0.3 else "NEUTRAL")

    price = g(last, "Close")
    atr   = g(last, "ATR")
    sl = tp = None
    if not any(np.isnan([price, atr])) and atr > 0:
        if label == "BULLISH":
            sl = round(price - 1.5 * atr, 2)
            tp = round(price + 3.0 * atr, 2)
        elif label == "BEARISH":
            sl = round(price + 1.5 * atr, 2)
            tp = round(price - 3.0 * atr, 2)

    return {
        "signal_label":          label,
        "composite_score":       round(comp, 4),
        "confidence_pct":        round(abs(comp) * 100, 1),
        "current_price":         round(price, 2) if not np.isnan(price) else None,
        "atr":                   round(atr, 2)   if not np.isnan(atr)   else None,
        "suggested_stop_loss":   sl,
        "suggested_take_profit": tp,
        "patterns":              pats,
    }


def run_data_agent(ticker: str, period: str, interval: str = "1d", use_stub: bool = False, seed: int = 42) -> dict:
    if use_stub or not YF_OK:
        raw_df = _stub_dataframe(seed=seed)
    else:
        raw_df = _fetch_yfinance(ticker, period, interval)
        if raw_df is None:
            raw_df = _stub_dataframe(seed=seed)

    raw_df.dropna(subset=["Close"], inplace=True)
    enriched_df = _compute_indicators(raw_df.copy())
    ta_signals  = _detect_patterns(enriched_df)

    date_range = None
    if len(enriched_df) > 0:
        date_range = (str(enriched_df.index[0].date()), str(enriched_df.index[-1].date()))

    return {
        "raw_df":     enriched_df,
        "ta_signals": ta_signals,
        "meta": {"row_count": len(enriched_df), "date_range": date_range},
    }

