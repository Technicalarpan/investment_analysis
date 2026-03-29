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

def investment_amount_advisor(investment_amount: float, risk: str) -> dict:
    if risk == "HIGH":
        invest_ratio = 0.3
        strategy = "Invest cautiously due to high risk"
        advice = "Market is volatile. Invest only a small portion now and wait for confirmation."
    elif risk == "MEDIUM":
        invest_ratio = 0.6
        strategy = "Invest partially due to moderate risk"
        advice = "Market is moderately bullish. Invest a portion now and the rest gradually."
    else:
        invest_ratio = 1.0
        strategy = "Safe to invest full amount"
        advice = "Market conditions are stable. You can invest your full amount."

    recommended = round(investment_amount * invest_ratio, 2)
    hold_back   = round(investment_amount - recommended, 2)

    return {
        "investment_amount":      investment_amount,
        "recommended_investment": recommended,
        "hold_back_amount":       hold_back,
        "invest_ratio":           invest_ratio,
        "investment_strategy":    strategy,
        "investment_advice":      advice,
    }


def allocation_advisor(amount: float, num_stocks: int, risk: str) -> dict:
    if num_stocks < 1:
        num_stocks = 1

    per_stock = round(amount / num_stocks, 2)

    if risk == "HIGH":
        invest_now = round(amount * 0.3, 2)
        invest_ratio = 0.3
        diversification_msg = "High risk — spread your investment across all stocks gradually."
    elif risk == "MEDIUM":
        invest_now = round(amount * 0.6, 2)
        invest_ratio = 0.6
        diversification_msg = "Good diversification — invest a portion now, rest over time."
    else:
        invest_now = amount
        invest_ratio = 1.0
        diversification_msg = "Low risk — diversification looks good, safe to invest now."

    invest_later  = round(amount - invest_now, 2)
    per_stock_now = round(invest_now / num_stocks, 2)
    good_diversification = num_stocks >= 3

    return {
        "total_amount":        amount,
        "num_stocks":          num_stocks,
        "per_stock":           per_stock,
        "invest_now":          invest_now,
        "invest_later":        invest_later,
        "per_stock_now":       per_stock_now,
        "invest_ratio":        invest_ratio,
        "diversification_msg": diversification_msg,
        "good_diversification": good_diversification,
    }


def smart_recommendation(decision: dict, investment_amount: float = None, num_stocks: int = None) -> dict:
    risk   = decision.get("risk", "MEDIUM")
    action = decision.get("action", "HOLD")

    result = {
        "action":  action,
        "risk":    risk,
        "summary": decision.get("summary", ""),
    }

    if investment_amount and investment_amount > 0:
        inv = investment_amount_advisor(investment_amount, risk)
        result["investment_advice"] = inv

        if num_stocks and num_stocks > 0:
            alloc = allocation_advisor(investment_amount, num_stocks, risk)
            result["allocation_advice"] = alloc

            if action == "BUY":
                mood = "bullish"
            elif action == "SELL":
                mood = "bearish"
            else:
                mood = "mixed/neutral"

            risk_desc = {"HIGH": "high risk — be cautious", "MEDIUM": "moderate risk", "LOW": "low risk"}.get(risk, "moderate risk")
            hold_back_str = f"Invest the remaining ₹{inv['hold_back_amount']:,.0f} gradually." if inv['hold_back_amount'] > 0 else "You can invest the full amount now."

            result["final_explanation"] = (
                f"Market is {mood} with {risk_desc}. "
                f"You can invest ₹{inv['recommended_investment']:,.0f} now out of ₹{investment_amount:,.0f}. "
                f"Allocate ~₹{alloc['per_stock_now']:,.0f} per stock across {num_stocks} stocks. "
                f"{hold_back_str}"
            )
        else:
            result["final_explanation"] = (
                f"{'Market looks ' + ('positive — consider buying.' if action == 'BUY' else 'weak — consider holding off.' if action == 'SELL' else 'neutral — wait for a clearer signal.')} "
                f"Recommended to invest ₹{inv['recommended_investment']:,.0f} now. {inv['investment_advice']}"
            )

    return result