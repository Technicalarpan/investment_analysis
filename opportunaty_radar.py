import os
import streamlit as st

try:
    import requests
    REQUESTS_OK = True
except:
    REQUESTS_OK = False


# Mock news data (used when News API not configured)
_MOCK_NEWS = [
    {
        "stock": "RELIANCE.NS", "name": "Reliance Industries",
        "headline": "Reliance Industries announces massive ₹75,000 Cr capex for renewable energy over next 3 years",
        "source": "Economic Times", "published": "2 hours ago",
        "keywords": ["capex", "renewable", "growth", "expansion"],
        "sentiment": "positive", "weight": 0.85,
        "alert": "New Contract / Expansion Alert: Major capital commitment signals strong growth outlook",
    },
    {
        "stock": "TCS.NS", "name": "TCS",
        "headline": "TCS wins $500M AI transformation deal with major US bank",
        "source": "Business Standard", "published": "5 hours ago",
        "keywords": ["deal", "win", "contract", "AI"],
        "sentiment": "positive", "weight": 0.80,
        "alert": "New Contract Alert: Large deal win boosts revenue visibility",
    },
    {
        "stock": "HDFCBANK.NS", "name": "HDFC Bank",
        "headline": "HDFC Bank promoters increase stake by 2% — strong vote of confidence",
        "source": "Mint", "published": "1 day ago",
        "keywords": ["promoter buying", "stake increase"],
        "sentiment": "positive", "weight": 0.90,
        "alert": "Promoter Buying Alert: Insiders buying their own stock — typically a very bullish sign",
    },
    {
        "stock": "INFY.NS", "name": "Infosys",
        "headline": "Infosys Q3 profit rises 12% YoY, upgrades revenue guidance for FY2026",
        "source": "Reuters", "published": "3 hours ago",
        "keywords": ["profit increase", "earnings beat", "guidance upgrade"],
        "sentiment": "positive", "weight": 0.88,
        "alert": "Earnings Beat Alert: Profit growth + guidance upgrade is a strong positive signal",
    },
    {
        "stock": "WIPRO.NS", "name": "Wipro",
        "headline": "Wipro faces headwinds as key client reduces IT spending — Q4 outlook cautious",
        "source": "Financial Express", "published": "6 hours ago",
        "keywords": ["headwinds", "client loss", "cautious"],
        "sentiment": "negative", "weight": -0.65,
        "alert": "Risk Alert: Client spending cuts could impact near-term revenue",
    },
    {
        "stock": "ADANIENT.NS", "name": "Adani Enterprises",
        "headline": "Adani Group announces strategic partnership with US firm for green hydrogen",
        "source": "Hindu Business Line", "published": "4 hours ago",
        "keywords": ["partnership", "green energy", "strategic"],
        "sentiment": "positive", "weight": 0.72,
        "alert": "Partnership Alert: New strategic alliance supports long-term growth narrative",
    },
    {
        "stock": "SBIN.NS", "name": "SBI",
        "headline": "SBI reports record quarterly profit of ₹16,891 Cr, NPA ratios improve significantly",
        "source": "Times of India", "published": "1 day ago",
        "keywords": ["record profit", "NPA improvement", "strong results"],
        "sentiment": "positive", "weight": 0.82,
        "alert": "Earnings Excellence Alert: Record profits + improving asset quality — very positive",
    },
    {
        "stock": "ICICIBANK.NS", "name": "ICICI Bank",
        "headline": "ICICI Bank launches AI-powered personal finance assistant — 50M users targeted",
        "source": "Economic Times", "published": "2 days ago",
        "keywords": ["AI", "innovation", "growth", "digital"],
        "sentiment": "positive", "weight": 0.68,
        "alert": "Innovation Alert: AI adoption at scale could drive fee income growth",
    },
]


def fetch_news_signals(stocks: list = None, ticker: str = None) -> list:
    """
    Fetch live news signals from NewsAPI for the selected stock.
    Falls back to curated mock data only if API key not set or API fails.
    """
    # Use environment variable if set, else fall back to the configured key
    news_api_key = os.environ.get("NEWS_API_KEY", "25e8dc764dea4960bd404dd972efb3d0").strip()

    # ── Positive / Negative keyword sets ──────────────────────────────────────
    POS_KEYWORDS = [
        "profit", "growth", "contract", "expansion", "upgrade", "record",
        "win", "deal", "buyback", "dividend", "strong", "beat", "surge",
        "increase", "positive", "promoter buying", "bullish", "outperform",
    ]
    NEG_KEYWORDS = [
        "loss", "decline", "fraud", "sell-off", "downgrade", "miss", "weak",
        "cut", "concern", "risk", "slowdown", "outflow", "disappointing",
        "penalty", "probe", "lawsuit", "crash", "fall", "bearish",
    ]

    def _score_headline(headline: str) -> tuple[str, float, str]:
        """Return (sentiment, weight, alert_text) for a headline."""
        hl = headline.lower()
        pos = sum(1 for w in POS_KEYWORDS if w in hl)
        neg = sum(1 for w in NEG_KEYWORDS if w in hl)

        if pos > neg:
            sent   = "positive"
            weight = min(0.95, 0.40 + pos * 0.12)
            # Generate plain-English alert
            if "profit" in hl or "earnings" in hl or "beat" in hl:
                alert = "Positive Earnings Alert: Strong profitability detected — bullish signal"
            elif "contract" in hl or "deal" in hl or "win" in hl:
                alert = "New Contract / Deal Alert: Revenue visibility improving — positive"
            elif "promoter" in hl or "buyback" in hl or "dividend" in hl:
                alert = "Insider Activity Alert: Company or promoters buying — strong confidence signal"
            elif "upgrade" in hl or "outperform" in hl:
                alert = "Analyst Upgrade Alert: Institutional sentiment turning positive"
            else:
                alert = "Positive Signal Detected: News sentiment is favourable"
        elif neg > pos:
            sent   = "negative"
            weight = -min(0.95, 0.40 + neg * 0.12)
            if "fraud" in hl or "probe" in hl or "lawsuit" in hl:
                alert = "⚠️ Regulatory / Legal Risk: Potential legal headwind — exercise caution"
            elif "loss" in hl or "miss" in hl or "disappointing" in hl:
                alert = "⚠️ Earnings Risk: Weak results may pressure the stock"
            elif "downgrade" in hl or "bearish" in hl:
                alert = "⚠️ Analyst Downgrade: Institutional sentiment turning negative"
            else:
                alert = "⚠️ Risk Alert: Negative news detected — proceed with caution"
        else:
            sent   = "neutral"
            weight = 0.0
            alert  = "Neutral: No strong directional signal in this news"

        return sent, weight, alert

    results = []

    if news_api_key and REQUESTS_OK:
        # Build query terms from ticker and stock name
        query_terms = []
        base_name = ""
        if ticker:
            # Convert ticker like "RELIANCE.NS" → "Reliance"
            base = ticker.replace(".NS", "").replace(".BO", "")
            # Map common tickers to full names for better results
            _TICKER_NAMES = {
                "RELIANCE": "Reliance Industries",
                "TCS": "TCS Tata Consultancy",
                "HDFCBANK": "HDFC Bank",
                "INFY": "Infosys",
                "ICICIBANK": "ICICI Bank",
                "KOTAKBANK": "Kotak Mahindra Bank",
                "AXISBANK": "Axis Bank",
                "SBIN": "SBI State Bank India",
                "BAJFINANCE": "Bajaj Finance",
                "LT": "Larsen Toubro",
                "HINDUNILVR": "Hindustan Unilever HUL",
                "WIPRO": "Wipro",
                "HCLTECH": "HCL Technologies",
                "TATAMOTORS": "Tata Motors",
                "MARUTI": "Maruti Suzuki",
                "ASIANPAINT": "Asian Paints",
                "SUNPHARMA": "Sun Pharma",
                "ADANIENT": "Adani Enterprises",
                "ITC": "ITC India",
                "BHARTIARTL": "Bharti Airtel",
                "M&M": "Mahindra",
                "POWERGRID": "Power Grid India",
                "NTPC": "NTPC India",
                "ULTRACEMCO": "UltraTech Cement",
                "TITAN": "Titan Company",
            }
            full_name = _TICKER_NAMES.get(base, base)
            base_name = full_name
            query_terms.append(full_name)

        if stocks:
            query_terms.extend(stocks[:2])

        if not query_terms:
            query_terms = ["Nifty50 India stock market"]

        # Try multiple endpoints and query strategies for richer results
        endpoints_to_try = []
        for qterm in query_terms[:2]:
            # 1. Everything endpoint - most articles
            endpoints_to_try.append((
                f"https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(qterm + ' India stock NSE')}"
                f"&language=en&sortBy=publishedAt&pageSize=10&apiKey={news_api_key}",
                qterm
            ))
            # 2. Top headlines - breaking news
            endpoints_to_try.append((
                f"https://newsapi.org/v2/top-headlines"
                f"?q={requests.utils.quote(qterm)}"
                f"&language=en&pageSize=5&apiKey={news_api_key}",
                qterm
            ))

        seen_headlines = set()
        for url, qterm in endpoints_to_try[:4]:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    articles = resp.json().get("articles", [])
                    for art in articles:
                        headline = (art.get("title") or "").strip()
                        if not headline or headline == "[Removed]":
                            continue
                        # De-duplicate
                        key = headline[:60].lower()
                        if key in seen_headlines:
                            continue
                        seen_headlines.add(key)

                        source    = art.get("source", {}).get("name", "Unknown")
                        published = art.get("publishedAt", "Recent")[:10]  # date only
                        description = (art.get("description") or "").strip()
                        # Use description to enrich scoring if available
                        full_text = headline + " " + description
                        sent, weight, alert = _score_headline(full_text)
                        results.append({
                            "stock":     ticker or qterm,
                            "name":      base_name or qterm.split()[0],
                            "headline":  headline,
                            "description": description[:200] if description else "",
                            "source":    source,
                            "published": published,
                            "keywords":  [w for w in POS_KEYWORDS + NEG_KEYWORDS if w in full_text.lower()][:5],
                            "sentiment": sent,
                            "weight":    weight,
                            "alert":     alert,
                            "url":       art.get("url", ""),
                        })
                elif resp.status_code == 401:
                    st.sidebar.warning("⚠️ NEWS_API_KEY invalid — using demo data")
                    break
                elif resp.status_code == 429:
                    st.sidebar.warning("⚠️ NewsAPI rate limit reached — using demo data")
                    break
            except Exception:
                pass

        if results:
            # Sort by weight magnitude (most impactful first)
            return sorted(results, key=lambda x: abs(x["weight"]), reverse=True)

    # ── Fallback: curated mock data (clearly labelled) ─────────────────────
    _MOCK_NEWS_DYNAMIC = [
        {
            "stock": ticker or "MARKET", "name": "Market",
            "headline": "NIFTY 50 companies report strong Q3 earnings growth amid robust domestic demand",
            "source": "Economic Times [Demo]", "published": "Today",
            "keywords": ["earnings", "growth", "strong"],
            "sentiment": "positive", "weight": 0.75,
            "alert": "Sector Earnings Alert: Broad-based earnings growth detected across large-caps",
        },
        {
            "stock": ticker or "MARKET", "name": "Market",
            "headline": "FII net buyers for 5th consecutive session — foreign inflows support Nifty rally",
            "source": "Business Standard [Demo]", "published": "Today",
            "keywords": ["FII", "buying", "inflow", "positive"],
            "sentiment": "positive", "weight": 0.70,
            "alert": "Institutional Buying Alert: Sustained FII inflows are a bullish market signal",
        },
        {
            "stock": ticker or "MARKET", "name": "Market",
            "headline": "RBI holds rates steady; governor signals accommodative stance in next MPC meet",
            "source": "Mint [Demo]", "published": "Yesterday",
            "keywords": ["RBI", "rates", "positive"],
            "sentiment": "positive", "weight": 0.60,
            "alert": "Macro Positive: Stable rates support equity valuations — favourable for markets",
        },
        {
            "stock": ticker or "MARKET", "name": "Market",
            "headline": "Rising US bond yields weigh on emerging market sentiment; India not immune",
            "source": "Reuters [Demo]", "published": "Yesterday",
            "keywords": ["yields", "risk", "sell-off"],
            "sentiment": "negative", "weight": -0.55,
            "alert": "⚠️ Global Risk Alert: US yield spike can trigger FII outflows from India",
        },
        {
            "stock": ticker or "MARKET", "name": "Market",
            "headline": "Crude oil prices rise 3% on supply cuts — input cost pressure for Indian companies",
            "source": "Financial Express [Demo]", "published": "2 days ago",
            "keywords": ["crude", "cost", "risk", "decline"],
            "sentiment": "negative", "weight": -0.50,
            "alert": "⚠️ Cost Risk Alert: Higher crude oil increases input costs and can compress margins",
        },
    ]
    return sorted(_MOCK_NEWS_DYNAMIC, key=lambda x: abs(x["weight"]), reverse=True)


def get_news_score_for_stock(ticker: str, news_signals: list) -> float:
    """Get aggregated news sentiment score for a specific stock."""
    relevant = [n for n in news_signals if n.get("stock") == ticker]
    if not relevant:
        return 0.0
    return sum(n["weight"] for n in relevant) / len(relevant)