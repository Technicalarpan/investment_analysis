import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SECTOR_MAP = {
    "RELIANCE.NS": "Energy / Conglomerate",
    "TCS.NS":      "Technology",
    "INFY.NS":     "Technology",
    "WIPRO.NS":    "Technology",
    "HCLTECH.NS":  "Technology",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS":"Banking",
    "AXISBANK.NS": "Banking",
    "KOTAKBANK.NS":"Banking",
    "SBIN.NS":     "Banking",
    "BAJFINANCE.NS":"Finance",
    "LT.NS":       "Infrastructure",
    "HINDUNILVR.NS":"FMCG",
    "ITC.NS":      "FMCG",
    "MARUTI.NS":   "Automobile",
    "TATAMOTORS.NS":"Automobile",
    "SUNPHARMA.NS":"Pharma",
    "ASIANPAINT.NS":"Paints",
    "TITAN.NS":    "Consumer Goods",
    "ADANIENT.NS": "Conglomerate",
    "BHARTIARTL.NS":"Telecom",
    "POWERGRID.NS":"Utilities",
    "NTPC.NS":     "Utilities",
    "ULTRACEMCO.NS":"Cement",
    "M&M.NS":      "Automobile",
}


def analyze_portfolio(holdings: list) -> dict:
    """
    Analyze portfolio: sector distribution, risk level, suggestions.
    holdings: list of {"ticker": "RELIANCE.NS", "name": "...", "allocation": 25.0}
    """
    if not holdings:
        return {"error": "No holdings provided"}

    # Filter out zero-allocation entries
    holdings = [h for h in holdings if h.get("allocation", 0) > 0]
    if not holdings:
        return {"error": "All allocations are zero. Please enter valid allocation percentages."}

    # Validate total allocation
    total_alloc = sum(h.get("allocation", 0) for h in holdings)
    if total_alloc <= 0:
        return {"error": "Total allocation must be greater than 0%"}

    # Sector distribution (normalize to 100% regardless of input total)
    sectors: dict[str, float] = {}
    for h in holdings:
        ticker = h.get("ticker", "")
        alloc  = float(h.get("allocation", 0))
        sector = SECTOR_MAP.get(ticker, "Other")
        sectors[sector] = sectors.get(sector, 0.0) + alloc

    sector_pct = {k: round(v / total_alloc * 100, 1) for k, v in sectors.items()}

    # Warn if total != 100 (show it, don't crash)
    alloc_note = ""
    if abs(total_alloc - 100) > 1:
        alloc_note = f"Note: Total allocation entered is {total_alloc:.1f}% (not 100%). Results shown as proportional %."

    max_sector_pct = max(sector_pct.values()) if sector_pct else 0.0
    num_sectors    = len(sector_pct)

    # Concentration risk
    if max_sector_pct > 50 or num_sectors < 3:
        concentration = "HIGH"
    elif max_sector_pct > 35 or num_sectors < 5:
        concentration = "MEDIUM"
    else:
        concentration = "LOW"

    dominant_sector = max(sector_pct, key=sector_pct.get) if sector_pct else "Unknown"

    # Personalised suggestions
    suggestions = []
    if alloc_note:
        suggestions.append(alloc_note)
    if max_sector_pct > 50:
        suggestions.append(f"⚠️ Your portfolio is heavily concentrated in {dominant_sector} ({max_sector_pct:.0f}%). Consider diversifying into other sectors to reduce risk.")
    if num_sectors < 4:
        suggestions.append("📊 You are in fewer than 4 sectors — add stocks from different industries to reduce concentration risk.")
    if "Technology" in sector_pct and sector_pct.get("Technology", 0) > 40:
        suggestions.append("💻 High tech exposure ({:.0f}%) — tech stocks can be volatile. Consider balancing with defensive sectors like FMCG or Utilities.".format(sector_pct.get("Technology", 0)))
    if "Banking" not in sector_pct and "Finance" not in sector_pct:
        suggestions.append("🏦 No banking or finance exposure — this sector often outperforms in rising-rate environments and adds stability.")
    if "Energy / Conglomerate" not in sector_pct and "Utilities" not in sector_pct:
        suggestions.append("⚡ Consider adding energy or utilities exposure for inflation protection and income stability.")
    if num_sectors >= 5 and max_sector_pct <= 35:
        suggestions.append("✅ Good diversification across multiple sectors. Keep monitoring quarterly and rebalance if any sector exceeds 35%.")
    if not suggestions:
        suggestions.append("✅ Your portfolio looks reasonably well-diversified. Review allocations periodically.")

    # Overall risk
    if concentration == "HIGH":
        overall_risk = "HIGH"
        risk_reason  = f"Heavily concentrated in {dominant_sector} ({max_sector_pct:.0f}%)"
    elif concentration == "MEDIUM":
        overall_risk = "MEDIUM"
        risk_reason  = f"Moderate diversification — {num_sectors} sectors"
    else:
        overall_risk = "LOW"
        risk_reason  = "Well-diversified across multiple sectors"

    summary = (
        f"Your portfolio is {overall_risk.lower()}-risk. "
        f"Dominant sector: {dominant_sector} ({max_sector_pct:.0f}%). "
        f"You hold stocks across {num_sectors} sector{'s' if num_sectors > 1 else ''}."
    )

    return {
        "sector_distribution": sector_pct,
        "dominant_sector":     dominant_sector,
        "num_sectors":         num_sectors,
        "concentration_risk":  concentration,
        "overall_risk":        overall_risk,
        "risk_reason":         risk_reason,
        "summary":             summary,
        "suggestions":         suggestions,
        "total_alloc_entered": round(total_alloc, 1),
    }

def project_portfolio_future(holdings: list, ta_results: dict = None, investment_amount: float = 100000) -> dict:
    """
    Project future portfolio value based on:
    1. Historical avg returns for each sector
    2. Current TA signal blended with sector expectations
    3. Monte Carlo-style range (conservative / base / optimistic)

    holdings: list of {ticker, name, allocation}
    ta_results: optional dict of {ticker: ta_signals} for signal-adjusted returns
    investment_amount: total investment in INR
    """
    if not holdings:
        return {"error": "No holdings provided"}

    holdings = [h for h in holdings if h.get("allocation", 0) > 0]
    if not holdings:
        return {"error": "No valid holdings"}

    total_alloc = sum(h.get("allocation", 0) for h in holdings)
    if total_alloc <= 0:
        return {"error": "Zero total allocation"}

    # Historical avg annual returns by sector (approximate NSE historical data)
    SECTOR_RETURNS = {
        "Technology":           {"base": 0.16, "std": 0.22},
        "Banking":              {"base": 0.13, "std": 0.18},
        "Finance":              {"base": 0.14, "std": 0.20},
        "Energy / Conglomerate":{"base": 0.12, "std": 0.17},
        "FMCG":                 {"base": 0.11, "std": 0.12},
        "Automobile":           {"base": 0.13, "std": 0.22},
        "Pharma":               {"base": 0.12, "std": 0.19},
        "Infrastructure":       {"base": 0.11, "std": 0.16},
        "Paints":               {"base": 0.13, "std": 0.15},
        "Consumer Goods":       {"base": 0.12, "std": 0.16},
        "Telecom":              {"base": 0.10, "std": 0.18},
        "Utilities":            {"base": 0.09, "std": 0.12},
        "Cement":               {"base": 0.11, "std": 0.17},
        "Conglomerate":         {"base": 0.12, "std": 0.20},
        "Other":                {"base": 0.11, "std": 0.18},
    }

    projections_by_stock = []
    portfolio_base_return = 0.0
    portfolio_bear_return = 0.0
    portfolio_bull_return = 0.0

    for h in holdings:
        ticker = h.get("ticker", "")
        name   = h.get("name", ticker)
        alloc  = float(h.get("allocation", 0)) / total_alloc  # normalize
        sector = SECTOR_MAP.get(ticker, "Other")
        sec_data = SECTOR_RETURNS.get(sector, SECTOR_RETURNS["Other"])

        base_annual = sec_data["base"]
        std_annual  = sec_data["std"]

        # Adjust return based on TA signal if available
        signal_adj = 0.0
        signal_label = "NEUTRAL"
        if ta_results and ticker in ta_results:
            sig = ta_results[ticker]
            comp = sig.get("composite_score", 0.0)
            signal_label = sig.get("signal_label", "NEUTRAL")
            # TA signal can nudge expected return by ±3%
            signal_adj = comp * 0.03

        adj_base = base_annual + signal_adj

        # Bear / Bull scenarios using ±1 std dev
        bear_annual = adj_base - std_annual * 0.8
        bull_annual = adj_base + std_annual * 0.8

        stock_amount = investment_amount * alloc

        projections_by_stock.append({
            "name":          name,
            "ticker":        ticker,
            "sector":        sector,
            "allocation_pct": round(alloc * 100, 1),
            "invested":      round(stock_amount, 0),
            "signal":        signal_label,
            "bear_1y":  round(stock_amount * (1 + bear_annual), 0),
            "base_1y":  round(stock_amount * (1 + adj_base), 0),
            "bull_1y":  round(stock_amount * (1 + bull_annual), 0),
            "bear_3y":  round(stock_amount * ((1 + bear_annual) ** 3), 0),
            "base_3y":  round(stock_amount * ((1 + adj_base) ** 3), 0),
            "bull_3y":  round(stock_amount * ((1 + bull_annual) ** 3), 0),
            "bear_5y":  round(stock_amount * ((1 + bear_annual) ** 5), 0),
            "base_5y":  round(stock_amount * ((1 + adj_base) ** 5), 0),
            "bull_5y":  round(stock_amount * ((1 + bull_annual) ** 5), 0),
            "base_annual_pct": round(adj_base * 100, 1),
        })

        portfolio_base_return += alloc * adj_base
        portfolio_bear_return += alloc * bear_annual
        portfolio_bull_return += alloc * bull_annual

    # Portfolio-level totals
    def _pv(r, n): return round(investment_amount * ((1 + r) ** n), 0)

    def _fmt(v):
        if v >= 1e7:   return f"₹{v/1e7:.2f} Cr"
        if v >= 1e5:   return f"₹{v/1e5:.2f} L"
        return f"₹{v:,.0f}"

    years = [1, 3, 5, 10]
    timeline = []
    for y in years:
        timeline.append({
            "years":       y,
            "bear_value":  _pv(portfolio_bear_return, y),
            "base_value":  _pv(portfolio_base_return, y),
            "bull_value":  _pv(portfolio_bull_return, y),
            "bear_fmt":    _fmt(_pv(portfolio_bear_return, y)),
            "base_fmt":    _fmt(_pv(portfolio_base_return, y)),
            "bull_fmt":    _fmt(_pv(portfolio_bull_return, y)),
            "bear_gain":   round((_pv(portfolio_bear_return, y) - investment_amount) / investment_amount * 100, 1),
            "base_gain":   round((_pv(portfolio_base_return, y) - investment_amount) / investment_amount * 100, 1),
            "bull_gain":   round((_pv(portfolio_bull_return, y) - investment_amount) / investment_amount * 100, 1),
        })

    return {
        "investment_amount":   investment_amount,
        "portfolio_base_return": round(portfolio_base_return * 100, 2),
        "portfolio_bear_return": round(portfolio_bear_return * 100, 2),
        "portfolio_bull_return": round(portfolio_bull_return * 100, 2),
        "timeline":            timeline,
        "by_stock":            projections_by_stock,
        "disclaimer":          "Projections are based on historical sector returns and current TA signals. Past performance does not guarantee future results. Not SEBI-registered advice.",
    }




def _build_projection_chart(timeline: list, investment_amount: float) -> "go.Figure":
    """Build a fan chart showing bear/base/bull portfolio projections."""
    years      = [0] + [t["years"]       for t in timeline]
    bear_vals  = [investment_amount] + [t["bear_value"]  for t in timeline]
    base_vals  = [investment_amount] + [t["base_value"]  for t in timeline]
    bull_vals  = [investment_amount] + [t["bull_value"]  for t in timeline]

    fig = go.Figure()

    # Fill between bear and bull
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=bull_vals + bear_vals[::-1],
        fill="toself",
        fillcolor="rgba(245,200,66,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        name="Range",
    ))

    # Bear line
    fig.add_trace(go.Scatter(
        x=years, y=bear_vals,
        mode="lines+markers",
        line=dict(color="#ff3f5b", width=1.8, dash="dash"),
        marker=dict(size=6),
        name="🔴 Bear Case",
    ))
    # Base line
    fig.add_trace(go.Scatter(
        x=years, y=base_vals,
        mode="lines+markers",
        line=dict(color="#f5c842", width=2.5),
        marker=dict(size=8),
        name="🟡 Base Case",
    ))
    # Bull line
    fig.add_trace(go.Scatter(
        x=years, y=bull_vals,
        mode="lines+markers",
        line=dict(color="#00e89a", width=1.8, dash="dash"),
        marker=dict(size=6),
        name="🟢 Bull Case",
    ))

    fig.add_hline(y=investment_amount,
                  line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1),
                  annotation_text="Your Investment",
                  annotation_font_color="#7a8ba0")

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#06090f", paper_bgcolor="#06090f",
        margin=dict(l=60, r=20, t=30, b=40),
        height=320,
        hovermode="x unified",
        xaxis=dict(
            title="Years", tickvals=[0, 1, 3, 5, 10],
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(family="JetBrains Mono"),
        ),
        yaxis=dict(
            title="Portfolio Value (₹)",
            gridcolor="rgba(255,255,255,0.04)",
            tickformat=",.0f",
            tickfont=dict(family="JetBrains Mono"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(color="#7a8ba0", size=10),
        ),
        font=dict(family="JetBrains Mono", color="#7a8ba0", size=10),
    )
    return fig


def _build_main_chart(df: pd.DataFrame, ticker: str) -> "go.Figure":
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.60, 0.18, 0.22],
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLCV",
        increasing=dict(line=dict(color="#00e89a"), fillcolor="#00e89a"),
        decreasing=dict(line=dict(color="#ff3f5b"), fillcolor="#ff3f5b"),
    ), row=1, col=1)

    for col, dash, alpha in [("BB_upper","dash","0.35"),("BB_middle","dot","0.20"),("BB_lower","dash","0.35")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=col.replace("BB_","BB "),
                line=dict(color=f"rgba(79,163,255,{alpha})", width=1, dash=dash),
            ), row=1, col=1)

    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=list(df.index) + list(df.index[::-1]),
            y=list(df["BB_upper"]) + list(df["BB_lower"][::-1]),
            fill="toself", fillcolor="rgba(79,163,255,0.04)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="BB Band",
        ), row=1, col=1)

    for col, color, lw in [("SMA_50","#f5c842",1.5), ("SMA_200","#b580ff",1.5)]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=col.replace("_","-"),
                line=dict(color=color, width=lw),
            ), row=1, col=1)

    bar_colors = [
        "#00e89a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff3f5b"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)

    if "Volume_MA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_MA"], name="Vol MA20",
            line=dict(color="#f5c842", width=1.5, dash="dot"),
        ), row=2, col=1)

    if all(c in df.columns for c in ["MACD", "MACD_signal", "MACD_hist"]):
        hist = df["MACD_hist"].fillna(0)
        hist_colors = [
            "rgba(0,232,154,0.65)" if v >= 0 else "rgba(255,63,91,0.65)"
            for v in hist
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=hist, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#4fa3ff", width=1.8),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#f5c842", width=1.8),
        ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#06090f", paper_bgcolor="#06090f",
        margin=dict(l=44, r=16, t=38, b=10),
        height=680,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            bgcolor="rgba(0,0,0,0.45)",
            font=dict(color="#7a8ba0", size=10),
        ),
        font=dict(family="JetBrains Mono, monospace", color="#7a8ba0"),
        title=dict(
            text=f"<b style='color:#f5c842'>{ticker}</b>  ·  Technical Chart",
            font=dict(size=14, family="Sora, sans-serif"), x=0.01,
        ),
    )
    for i in [1, 2, 3]:
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.03)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.03)", row=i, col=1)

    return fig


def _build_rsi_chart(df: pd.DataFrame) -> "go.Figure | None":
    if "RSI" not in df.columns:
        return None
    fig = go.Figure()
    fig.add_hrect(y0=65, y1=100, fillcolor="rgba(255,63,91,0.06)", line_width=0)
    fig.add_hrect(y0=0,  y1=35,  fillcolor="rgba(0,232,154,0.06)", line_width=0)
    fig.add_hline(y=65, line=dict(color="rgba(255,63,91,0.4)", dash="dot", width=1))
    fig.add_hline(y=35, line=dict(color="rgba(0,232,154,0.4)", dash="dot", width=1))
    fig.add_hline(y=50, line=dict(color="rgba(255,255,255,0.07)", width=1))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        line=dict(color="#4fa3ff", width=2),
        fill="tozeroy", fillcolor="rgba(79,163,255,0.07)",
        name="RSI (14)",
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#06090f", paper_bgcolor="#06090f",
        margin=dict(l=44, r=16, t=8, b=8),
        height=160, showlegend=False, hovermode="x",
        yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.03)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        font=dict(family="JetBrains Mono", color="#7a8ba0", size=10),
    )
    return fig


def _build_sector_pie(sector_pct: dict) -> "go.Figure":
    colors = ["#f5c842","#00e89a","#4fa3ff","#b580ff","#ff8c42","#ff3f5b","#00d4ff","#a0ff42","#ffb347","#c084fc"]
    fig = go.Figure(go.Pie(
        labels=list(sector_pct.keys()),
        values=list(sector_pct.values()),
        hole=0.55,
        marker=dict(colors=colors[:len(sector_pct)], line=dict(color="#06090f", width=2)),
        textfont=dict(family="JetBrains Mono", size=11),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#06090f",
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
        legend=dict(font=dict(size=10, color="#7a8ba0"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#e2eafc"),
    )
    return fig


def _build_backtest_chart(trades: list) -> "go.Figure":
    if not trades:
        return None
    cumulative = []
    cum = 0.0
    for t in trades:
        cum += t["return_pct"]
        cumulative.append(cum)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=cumulative,
        mode="lines+markers",
        line=dict(color="#f5c842", width=2),
        fill="tozeroy",
        fillcolor="rgba(245,200,66,0.07)",
        name="Cumulative Return %",
        marker=dict(size=5, color=["#00e89a" if t["success"] else "#ff3f5b" for t in trades]),
    ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#06090f", paper_bgcolor="#06090f",
        margin=dict(l=44, r=16, t=20, b=20),
        height=200,
        hovermode="x",
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Trade #"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.03)", title="Cumulative Return %"),
        font=dict(family="JetBrains Mono", color="#7a8ba0", size=10),
        showlegend=False,
    )
    return fig