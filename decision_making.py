def compute_decision(ta_signals: dict, news_score: float = 0.0) -> dict:
    """
    Core Decision Engine: Convert technical + news signals into BUY/SELL/HOLD.
    Returns human-readable decision with explanation.
    """
    comp  = ta_signals.get("composite_score", 0.0)
    price = ta_signals.get("current_price", 0.0) or 0.0
    sl    = ta_signals.get("suggested_stop_loss")
    tp    = ta_signals.get("suggested_take_profit")
    atr   = ta_signals.get("atr", 0.0) or 0.0
    pats  = ta_signals.get("patterns", [])

    # Blend TA (80%) + News (20%)
    blended = comp * 0.80 + news_score * 0.20

    # Decision thresholds
    if blended >= 0.25:
        action = "BUY"
        confidence = min(95, int(abs(blended) * 90 + 25))
    elif blended <= -0.25:
        action = "SELL"
        confidence = min(95, int(abs(blended) * 90 + 25))
    else:
        action = "HOLD"
        confidence = max(30, 60 - int(abs(blended) * 30))

    # Risk level
    rsi = next((g for p in pats for g in [p.get("score", 0)] if p["name"] == "RSI Extreme"), 0)
    vol_score = abs(next((p["score"] for p in pats if p["name"] == "Volume Spike"), 0))
    volatility = atr / price if price > 0 else 0
    if volatility > 0.025 or abs(blended) < 0.15:
        risk = "HIGH"
    elif volatility > 0.012:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Plain-English reasons
    positive_reasons = []
    negative_reasons = []
    for p in pats:
        sc = p["score"]
        name = p["name"]
        if sc > 0.1:
            if name == "MACD Crossover":
                positive_reasons.append("Buying momentum is building up — trend turning positive")
            elif name == "RSI Extreme":
                positive_reasons.append("Stock was in oversold zone — good potential buying opportunity")
            elif name == "Golden/Death Cross":
                positive_reasons.append("Long-term trend is strongly bullish (Golden Cross confirmed)")
            elif name == "BB Breakout":
                positive_reasons.append("Price broke above resistance with strong buying activity")
            elif name == "Volume Spike":
                positive_reasons.append("Unusually high trading activity — possible institutional buying")
        elif sc < -0.1:
            if name == "MACD Crossover":
                negative_reasons.append("Selling pressure increasing — trend turning negative")
            elif name == "RSI Extreme":
                negative_reasons.append("Stock is overvalued in short term — caution advised")
            elif name == "Golden/Death Cross":
                negative_reasons.append("Long-term trend is bearish — stock below key moving average")
            elif name == "BB Breakout":
                negative_reasons.append("Price broke below support — downward pressure present")
            elif name == "Volume Spike":
                negative_reasons.append("High sell volume detected — possible institutional exit")

    # Generate plain-English summary
    if action == "BUY":
        summary = (
            f"The stock shows strong positive signals. "
            f"{positive_reasons[0] if positive_reasons else 'Multiple technical indicators are positive.'}. "
            f"This appears to be a good entry opportunity with manageable risk."
        )
    elif action == "SELL":
        summary = (
            f"The stock is showing weakness. "
            f"{negative_reasons[0] if negative_reasons else 'Multiple technical indicators are negative.'}. "
            f"Consider reducing exposure or waiting for better prices."
        )
    else:
        summary = (
            "The stock is in a wait-and-watch zone. Signals are mixed — "
            "not a clear buy or sell right now. "
            "Patience is key; wait for a clearer trend before acting."
        )

    # Price targets
    entry_low  = round(price * 0.99, 2)
    entry_high = round(price * 1.01, 2)
    stop_loss  = sl if sl else round(price * (0.94 if action == "BUY" else 1.06), 2)
    target     = tp if tp else round(price * (1.10 if action == "BUY" else 0.92), 2)
    rr = "N/A"
    if sl and tp and price and abs(price - stop_loss) > 0:
        rr = f"1 : {abs(target - price) / abs(price - stop_loss):.1f}"

    return {
        "action":      action,
        "confidence":  confidence,
        "risk":        risk,
        "summary":     summary,
        "entry_zone":  f"₹{entry_low:,.2f} – ₹{entry_high:,.2f}",
        "stop_loss":   f"₹{stop_loss:,.2f}",
        "target":      f"₹{target:,.2f}",
        "risk_reward": rr,
        "positive_reasons": positive_reasons,
        "negative_reasons": negative_reasons,
        "blended_score": round(blended, 4),
    }

