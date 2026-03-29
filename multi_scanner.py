from data_engine import run_data_agent
from decision_making import compute_decision

def run_scanner(stocks: dict, period: str = "6mo", use_stub: bool = False, max_stocks: int = 10) -> list:
    """
    Run signal engine on multiple stocks and rank by confidence score.
    Returns ranked list with BUY/SELL/HOLD decision for each.
    """
    results = []
    stock_items = list(stocks.items())[:max_stocks]

    for i, (name, ticker) in enumerate(stock_items):
        seed = hash(ticker) % 1000 + i
        try:
            data = run_data_agent(ticker, period, use_stub=use_stub, seed=seed)
            ta   = data["ta_signals"]
            dec  = compute_decision(ta)
            price = ta.get("current_price", 0.0) or 0.0
            results.append({
                "rank":       0,
                "name":       name,
                "ticker":     ticker,
                "action":     dec["action"],
                "confidence": dec["confidence"],
                "risk":       dec["risk"],
                "price":      price,
                "score":      ta.get("composite_score", 0.0),
                "summary":    dec["summary"][:100] + "…" if len(dec["summary"]) > 100 else dec["summary"],
            })
        except Exception as e:
            results.append({
                "rank": 0, "name": name, "ticker": ticker,
                "action": "HOLD", "confidence": 0, "risk": "HIGH",
                "price": 0.0, "score": 0.0, "summary": f"Data unavailable: {str(e)[:50]}",
            })

    # Rank: BUY first (by confidence), then HOLD, then SELL
    def sort_key(x):
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        return (order.get(x["action"], 1), -x["confidence"])

    results.sort(key=sort_key)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
