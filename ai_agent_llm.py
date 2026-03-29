import os
import json
import re
import time
# Lazy import to avoid crash if not installed
def _lazy_import(dotted_path: str):
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None

ChatAnthropic = _lazy_import("langchain_anthropic.ChatAnthropic")
ChatOpenAI    = _lazy_import("langchain_openai.ChatOpenAI")
SystemMessage = _lazy_import("langchain_core.messages.SystemMessage")
HumanMessage  = _lazy_import("langchain_core.messages.HumanMessage")

_MOCK_FILINGS: dict[str, str] = {
    "RELIANCE.NS": """
Reliance Industries Q3 FY2026 Earnings (Mock Data)
Revenue: ₹2,35,000 Cr (+8.3% YoY) | EBITDA: ₹46,500 Cr (+11.2%) | PAT: ₹19,800 Cr (+9.5%)
• Jio Platforms: 14.2M net subscriber additions; ARPU rose to ₹197.6/month (+6% YoY).
• Retail segment crossed 1,000-store milestone; same-store-sales growth of 12%.
• New Energy capex ₹8,400 Cr this quarter. Net debt reduced by ₹12,000 Cr QoQ.
Bulk Deals: LIC of India BOUGHT 2.1 Cr shares @ ₹2,890 — accumulate signal.
Key Risks: crude oil price volatility; 5G competition intensifying.""",

    "TCS.NS": """
TCS Q3 FY2026 Results (Mock Data)
Revenue: $7.8 Bn (+5.4% YoY USD) | EBIT margin: 23.1% | Net cash: ₹1,02,000 Cr
• BFSI vertical (+6.2% YoY) — strongest growth in 6 quarters.
• GenAI TCV deal pipeline at $1.5 Bn; Board approved ₹17,000 Cr buyback.
Bulk Deals: Vanguard BOUGHT 30 lakh shares @ ₹3,920.
Key Risks: US client caution; INR appreciation headwind.""",

    "INFY.NS": """
Infosys Q3 FY2026 Results (Mock Data)
Revenue: $4.92 Bn (+4.8% CC) | Operating margin: 21.3% (+80 bps QoQ)
FY2026 revenue guidance upgraded to 5-6%. Cloud & AI services growing at 25% YoY.
Bulk Deals: Aberdeen Standard SOLD 15 lakh shares @ ₹1,810.
Key Risks: US visa tightening; European BFSI budget freezes.""",

    "HDFCBANK.NS": """
HDFC Bank Q3 FY2026 Results (Mock Data)
NII: ₹30,100 Cr (+12.4% YoY) | NIM: 3.52% | GNPA: 1.24% | PCR: 74%
• Credit-to-Deposit ratio improved to 82%; Branch network at 9,100+.
Bulk Deals: SBI Mutual Fund BOUGHT 45 lakh shares @ ₹1,680.
Key Risks: Deposit growth lagging; RBI scrutiny on unsecured lending.""",

    "ICICIBANK.NS": """
ICICI Bank Q3 FY2026 Results (Mock Data)
PAT: ₹11,800 Cr (+14.8% YoY) | ROE: 18.1% | GNPA: 2.15% | PCR: 78%
• Digital transactions: 88% of total. iMobile Pay: 110 million+ users.
Bulk Deals: HDFC AMC ADDED 18 lakh shares. FII holding rose to 47.2%.
Key Risks: Global macro uncertainty; credit card NPA uptick.""",

    "DEFAULT": """
Corporate Filing Summary — Indian Large-Cap (Mock Data)
Revenue growth: +7% YoY | Operating margin: 15-18% | Debt-to-Equity: 0.4
Promoter holding: ~52% (stable). No significant bulk deal activity.
Key Macro Risks: Nifty 50 P/E at ~22×; FII outflow risk; INR/USD at ₹84.""",
}

_SYSTEM_PROMPT = """You are Alpha Radar, an elite quantitative analyst specialising in Indian equity markets (NSE/BSE). 

Your job: given TA signals and corporate filings, synthesise ONE definitive trading signal for RETAIL INVESTORS.

CRITICAL OUTPUT RULE: Respond with ONLY valid JSON. No markdown, no preamble.

Required JSON schema:
{
  "ticker": "string",
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <integer 0-100>,
  "time_horizon": "SHORT_TERM (1-4 weeks)" | "MEDIUM_TERM (1-3 months)" | "LONG_TERM (3-12 months)",
  "entry_price_zone": "string e.g. ₹2,820–₹2,870",
  "stop_loss": "string e.g. ₹2,740",
  "take_profit": "string e.g. ₹3,020",
  "risk_reward": "string e.g. 1 : 2.4",
  "plain_english_summary": "3-4 sentences a beginner investor can understand. No jargon.",
  "bullish_factors": ["array of concise strings in plain language"],
  "bearish_factors": ["array of concise strings in plain language"],
  "key_catalysts": ["upcoming events that could move the stock"],
  "disclaimer": "This is algorithmic research output, NOT SEBI-registered investment advice. Consult your financial advisor."
}

Rules:
- Always use plain English in summaries — no RSI/MACD jargon for non-experts
- If TA and FA signals diverge, lower confidence and consider NEUTRAL
- Reference Indian market context: FII flows, RBI MPC, Nifty valuations
- Never recommend leveraged, F&O, or intraday positions"""


def _build_llm():
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    oai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if ant_key and ChatAnthropic is not None:
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.2, api_key=ant_key)
    if oai_key and ChatOpenAI is not None:
        return ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=oai_key)
    return None


def _rule_based_signal(ticker: str, ta: dict, filing: str) -> dict:
    comp  = ta.get("composite_score", 0.0)
    price = ta.get("current_price", 0.0) or 0.0
    sl    = ta.get("suggested_stop_loss")
    tp    = ta.get("suggested_take_profit")
    atr   = ta.get("atr", 0.0) or 0.0

    fl = filing.lower()
    bull_kw = ["growth","upgrade","buyback","record","strong","profit","accumulate","partnership","increase","dividend","improvement","positive","rise"]
    bear_kw = ["loss","debt","downgrade","miss","weak","sell","cut","decline","concern","risk","slowdown","outflow","disappointing"]
    fa_bull = sum(fl.count(k) for k in bull_kw)
    fa_bear = sum(fl.count(k) for k in bear_kw)
    fa_score = (fa_bull - fa_bear) / max(fa_bull + fa_bear, 1)

    blended = comp * 0.7 + fa_score * 0.3

    if blended >= 0.25:
        sig, conf = "BULLISH", min(90, int(abs(blended) * 85 + 25))
    elif blended <= -0.25:
        sig, conf = "BEARISH", min(90, int(abs(blended) * 85 + 25))
    else:
        sig, conf = "NEUTRAL", max(20, 50 - int(abs(blended) * 25))

    summaries = {
        "BULLISH": f"{ticker} is showing positive signals. Price trend is rising and trading activity is above normal. This could be a good time to consider entering, but always set a stop-loss to protect your investment.",
        "BEARISH": f"{ticker} is showing signs of weakness. The price trend has turned negative with selling pressure. It may be wise to stay on the sidelines until the situation stabilizes.",
        "NEUTRAL": f"{ticker} is in a wait-and-watch phase. Signals are mixed — not a clear buy or sell. Patient investors should wait for a clearer direction before taking action.",
    }

    ez = (f"₹{price - 0.5*atr:.1f}–₹{price + 0.5*atr:.1f}" if atr > 0 else "Near current market price")
    if sl and tp and price and abs(price - sl) > 0:
        rr = f"1 : {abs(tp - price) / abs(price - sl):.1f}"
    else:
        rr = "N/A"

    return {
        "ticker":   ticker,
        "signal":   sig,
        "confidence": conf,
        "time_horizon": "SHORT_TERM (1-4 weeks)",
        "entry_price_zone": ez,
        "stop_loss":   f"₹{sl}"  if sl else "N/A",
        "take_profit": f"₹{tp}"  if tp else "N/A",
        "risk_reward": rr,
        "plain_english_summary": summaries[sig],
        "bullish_factors": [p["detail"][:80] for p in ta.get("patterns", []) if p["score"] > 0.1] or ["No strong positive signals detected."],
        "bearish_factors": [p["detail"][:80] for p in ta.get("patterns", []) if p["score"] < -0.1] or ["No major risks detected currently."],
        "key_catalysts": ["Upcoming quarterly earnings release", "RBI Monetary Policy Committee decision", "Monthly FII/DII flow data", "Global crude oil price movement"],
        "disclaimer": "This is algorithmic research output, NOT SEBI-registered investment advice. Consult your financial advisor.",
        "_source": "rule_based",
    }


def run_ai_agent(ticker: str, ta_signals: dict, filing_text: str = None) -> dict:
    filing = filing_text or _MOCK_FILINGS.get(ticker, _MOCK_FILINGS["DEFAULT"])
    llm    = _build_llm()

    if llm is None:
        return _rule_based_signal(ticker, ta_signals, filing)

    pat_lines = "\n".join(
        f"  [{p['name']:22s}] score={p['score']:+.2f}  {p['detail'][:80]}"
        for p in ta_signals.get("patterns", [])
    )
    human_msg = f"""=== TECHNICAL ANALYSIS SIGNALS ===
Ticker          : {ticker}
Composite Score : {ta_signals.get('composite_score', 0):+.4f}
TA Label        : {ta_signals.get('signal_label', 'NEUTRAL')}
Current Price   : ₹{ta_signals.get('current_price', 'N/A')}
ATR (14-day)    : ₹{ta_signals.get('atr', 'N/A')}

Individual Patterns:
{pat_lines or "No patterns detected."}

=== CORPORATE FILING + FUNDAMENTAL SUMMARY ===
{filing.strip()}

Generate JSON trading signal."""

    for attempt in range(1, 4):
        try:
            messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)]
            raw   = llm.invoke(messages).content
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found")
            parsed = json.loads(clean[start:end])
            parsed.setdefault("disclaimer", "This is algorithmic research output, NOT SEBI-registered investment advice. Consult your financial advisor.")
            parsed["_source"] = "llm"
            return parsed
        except Exception as exc:
            is_rate_limit = any(k in str(exc).lower() for k in ["rate","429","quota","overload"])
            if attempt < 3:
                time.sleep((5 ** attempt) if is_rate_limit else 2)
            else:
                result = _rule_based_signal(ticker, ta_signals, filing)
                result["_source"] = "rule_based_after_llm_failure"
                return result

    return _rule_based_signal(ticker, ta_signals, filing)
