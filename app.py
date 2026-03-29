import os
import time
import logging
import warnings
from data_engine import *
from decision_making import *
from ai_agent_llm import *
from opportunaty_radar import *
from multi_scanner import *
from portfolio_ana import *
from backtesting import *

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pandas_ta as ta
    TA_OK = True
except Exception:
    TA_OK = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

st.set_page_config(
    page_title="Alpha Radar v2 | AI for Indian Investor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
_STOCKS = NIFTY_50

def _metric_card(label: str, value: str, sub: str = None, sub_color: str = None) -> None:
    sub_html = (f'<div class="mc-sub" style="color:{sub_color or "var(--muted)"};">{sub}</div>' if sub else "")
    st.markdown(
        f'<div class="mc"><div class="mc-label">{label}</div>'
        f'<div class="mc-val">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _render_decision_card(decision: dict, ticker: str, price: float) -> None:
    action   = decision["action"].lower()
    conf     = decision["confidence"]
    risk     = decision["risk"]
    risk_cls = {"LOW": "rp-low", "MEDIUM": "rp-mid", "HIGH": "rp-high"}.get(risk, "rp-mid")

    emoji = {"buy": "🟢", "sell": "🔴", "hold": "⏸️"}.get(action, "⏸️")

    pos_reasons = decision.get("positive_reasons", [])
    neg_reasons = decision.get("negative_reasons", [])

    def reason_li(items, dot_cls):
        return "".join(f'<li class="ri"><span class="ri-dot {dot_cls}"></span>{item}</li>' for item in items[:4])

    st.markdown(f"""
<div class="dc {action}">
  <!-- Header row -->
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
    <div>
      <div class="action-pill {action}">{emoji}&nbsp; {decision["action"]}</div>
      <div style="font-size:.82rem; color:var(--muted); margin-top:8px; font-family:'JetBrains Mono',monospace;">
        {ticker} &nbsp;·&nbsp; ₹{price:,.2f}
      </div>
    </div>
    <div style="text-align:right;">
      <div class="mc-label">Confidence</div>
      <div style="font-size:1.6rem; font-weight:800; font-family:'Sora',sans-serif; color:{'var(--green)' if action=='buy' else 'var(--red)' if action=='sell' else 'var(--blue)'};">{conf}%</div>
      <div class="conf-track" style="width:120px; margin-left:auto;">
        <div class="conf-fill {action}" style="width:{conf}%;"></div>
      </div>
      <div style="margin-top:8px;"><span class="risk-pill {risk_cls}">⚠ {risk} RISK</span></div>
    </div>
  </div>

  <!-- Plain-English explanation -->
  <div class="plain-reason">{decision["summary"]}</div>

  <!-- Price targets -->
  <div class="sh">Suggested Action Levels</div>
  <div class="price-grid">
    <div class="pg-item">
      <div class="pg-lbl">Entry Zone</div>
      <div class="pg-val" style="color:var(--txt);">{decision["entry_zone"]}</div>
    </div>
    <div class="pg-item">
      <div class="pg-lbl" style="color:var(--red);">Stop Loss</div>
      <div class="pg-val" style="color:var(--red);">{decision["stop_loss"]}</div>
      <div style="font-size:.65rem;color:var(--muted);margin-top:2px;">Exit if price falls here</div>
    </div>
    <div class="pg-item">
      <div class="pg-lbl" style="color:var(--green);">Target</div>
      <div class="pg-val" style="color:var(--green);">{decision["target"]}</div>
      <div style="font-size:.65rem;color:var(--muted);margin-top:2px;">Expected upside</div>
    </div>
  </div>

  <div style="font-size:.78rem;color:var(--muted);margin-top:10px;">
    📐 Risk : Reward &nbsp;{decision["risk_reward"]}
  </div>

  <!-- Why section -->
  <div class="sh" style="margin-top:18px;">Why This Recommendation?</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div>
      <div style="font-size:.72rem;color:var(--green);font-weight:700;margin-bottom:6px;">✅ Positive Signals</div>
      <ul class="reason-list">
        {reason_li(pos_reasons or ['No strong positive signals currently'], 'g')}
      </ul>
    </div>
    <div>
      <div style="font-size:.72rem;color:var(--red);font-weight:700;margin-bottom:6px;">⚠ Risk Factors</div>
      <ul class="reason-list">
        {reason_li(neg_reasons or ['No major risk signals detected'], 'r')}
      </ul>
    </div>
  </div>

  <div class="disc">⚖️ NOT SEBI-registered investment advice. This is AI-generated analysis for educational purposes only. Consult a SEBI-registered financial advisor before investing.</div>
</div>""", unsafe_allow_html=True)


def _render_ai_card(sd: dict) -> None:
    """Render the full AI signal card (supplementary to decision card)."""
    sig  = sd.get("signal", "NEUTRAL").upper()
    conf = sd.get("confidence", 50)
    src  = "🤖 AI-Powered" if sd.get("_source") == "llm" else "⚙️ Rule-Based Engine"
    color = {"BULLISH":"#00e89a","BEARISH":"#ff3f5b","NEUTRAL":"#4fa3ff"}.get(sig,"#4fa3ff")

    bull = "".join(f'<li class="ri"><span class="ri-dot g"></span>{item}</li>' for item in (sd.get("bullish_factors",[]) or [])[:4])
    bear = "".join(f'<li class="ri"><span class="ri-dot r"></span>{item}</li>' for item in (sd.get("bearish_factors",[]) or [])[:4])
    cat  = "".join(f'<li class="ri"><span class="ri-dot y"></span>{item}</li>' for item in (sd.get("key_catalysts",[]) or [])[:4])

    st.markdown(f"""
<div style="background:var(--card2);border:1px solid var(--border);border-radius:14px;padding:22px 26px;margin-top:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);">AI Deep Analysis</div>
    <div style="font-size:.68rem;color:var(--muted);">{src}</div>
  </div>
  <div style="font-size:.88rem;line-height:1.75;color:var(--muted);border-left:3px solid rgba(245,200,66,.4);padding-left:14px;margin-bottom:16px;">
    {sd.get("plain_english_summary","No summary available.")}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
    <div><div style="font-size:.7rem;color:var(--green);font-weight:700;margin-bottom:6px;">POSITIVE FACTORS</div><ul class="reason-list">{bull}</ul></div>
    <div><div style="font-size:.7rem;color:var(--red);font-weight:700;margin-bottom:6px;">RISK FACTORS</div><ul class="reason-list">{bear}</ul></div>
    <div><div style="font-size:.7rem;color:var(--gold);font-weight:700;margin-bottom:6px;">KEY CATALYSTS</div><ul class="reason-list">{cat}</ul></div>
  </div>
  <div class="disc" style="margin-top:12px;">⚖️ {sd.get("disclaimer","")}</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# ═══════════════════════  MAIN APP  ═══════════════════════════════════════
# =============================================================================

def main() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="banner">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div>
          <h1>📡 Alpha Radar <span class="badge">v2.0</span></h1>
          <p class="sub">AI Decision-Support System for Indian Retail Investors &nbsp;·&nbsp; ET AI Hackathon 2026</p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:8px 0 20px;">
          <div style="font-size:2.2rem;">📡</div>
          <div style="font-size:1rem;font-weight:800;color:#f5c842;font-family:'Sora',sans-serif;">ALPHA RADAR</div>
          <div style="font-size:.66rem;color:#7a8ba0;margin-top:4px;letter-spacing:1px;">ET AI HACKATHON 2026</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### Stock Selector")
        mode = st.radio("Input mode", ["Quick Select", "Manual Entry"], horizontal=True, label_visibility="collapsed")

        if mode == "Quick Select":
            stock_name = st.selectbox("Choose stock", list(_STOCKS.keys()), index=2)
            ticker     = _STOCKS[stock_name]
        else:
            ticker = st.text_input("NSE Ticker", "RELIANCE.NS", placeholder="e.g. WIPRO.NS").strip().upper()

        st.markdown(f"""
        <div style="background:rgba(245,200,66,.08);border:1px solid rgba(245,200,66,.2);
                    border-radius:9px;padding:10px 14px;margin:10px 0 16px;">
          <div style="font-size:.65rem;color:#7a8ba0;">Selected</div>
          <div style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#f5c842;font-size:1.1rem;">{ticker}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### Settings")
        period   = st.select_slider("Data Period", options=["3mo","6mo","1y","2y"], value="1y")
        use_stub = st.checkbox("🔌 Offline Demo Mode", help="Uses synthetic price data — no internet needed.")

        has_ant = bool(os.environ.get("ANTHROPIC_API_KEY","").strip())
        has_oai = bool(os.environ.get("OPENAI_API_KEY","").strip())
        if has_ant:   llm_label, llm_color = "🟢 Anthropic Claude", "#00e89a"
        elif has_oai: llm_label, llm_color = "🟢 OpenAI GPT-4o",    "#00e89a"
        else:         llm_label, llm_color = "⚙️ Rule-Based Engine", "#f5c842"

        st.markdown(
            f'<div class="mc-label" style="margin-top:8px;">Signal Engine</div>'
            f'<div style="font-size:.82rem;color:{llm_color};margin-bottom:14px;">{llm_label}</div>',
            unsafe_allow_html=True,
        )

        pkg_status = [
            f"{'✅' if YF_OK    else '❌'} yfinance",
            f"{'✅' if TA_OK    else '⚠️'} pandas_ta",
            f"{'✅' if PLOTLY_OK else '❌'} plotly",
        ]
        st.markdown(
            '<div class="mc-label">Package Status</div>'
            + "".join(f'<div style="font-size:.72rem;color:#7a8ba0;">{s}</div>' for s in pkg_status),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        run_btn = st.button("🚀  Run Alpha Radar", type="primary")
        st.markdown("""
        <div style="font-size:.67rem;color:#3a4455;line-height:1.9;margin-top:12px;">
          <b style="color:#7a8ba0;">Modules</b><br>
          Decision Engine · Opportunity Radar<br>
          Backtesting · Portfolio Analyzer<br>
          Technical Details · Explainable AI<br><br>
          <b style="color:#7a8ba0;">⚠️ Not SEBI investment advice</b>
        </div>""", unsafe_allow_html=True)

    # ── Landing page ──────────────────────────────────────────────────────────
    if not run_btn:
        st.markdown('<div class="sh">What Alpha Radar Does For You</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        features = [
            ("🎯", "Clear BUY / SELL / HOLD", "No more confusing RSI or MACD numbers. Get a plain-English decision with confidence score and risk level."),
            ("📰", "Opportunity Radar", "AI scans real-time news headlines for profit alerts, promoter buying, and new contracts — so you never miss an opportunity."),
            ("🕰️", "Backtesting", "See how this signal has performed on real historical data. Know the win rate before you act on it."),
            ("🗣️", "Plain-Language Explanations", "Every signal comes with a simple explanation — no finance degree needed."),
            ("💼", "Portfolio Analyzer", "Enter your holdings and get instant risk analysis, sector breakdown, and diversification advice."),
            ("📊", "Technical Indicators", "MACD, RSI, Bollinger Bands, Volume Spike — all computed live from NSE data."),
        ]
        for col, (ic, ti, bo) in zip([c1, c2, c3], features[:3]):
            with col:
                st.markdown(
                    f'<div class="mc" style="text-align:center;padding:24px 16px;">'
                    f'<div style="font-size:2rem;margin-bottom:8px;">{ic}</div>'
                    f'<div style="font-weight:700;font-family:Sora,sans-serif;margin-bottom:8px;">{ti}</div>'
                    f'<div style="font-size:.82rem;color:#7a8ba0;line-height:1.65;">{bo}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        for col, (ic, ti, bo) in zip([c4, c5, c6], features[3:]):
            with col:
                st.markdown(
                    f'<div class="mc" style="text-align:center;padding:24px 16px;">'
                    f'<div style="font-size:2rem;margin-bottom:8px;">{ic}</div>'
                    f'<div style="font-weight:700;font-family:Sora,sans-serif;margin-bottom:8px;">{ti}</div>'
                    f'<div style="font-size:.82rem;color:#7a8ba0;line-height:1.65;">{bo}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
        st.markdown(
            '<div style="text-align:center;margin:48px 0;font-size:.84rem;color:#7a8ba0;">'
            '← Select a stock in the sidebar and click '
            '<b style="color:#f5c842;">🚀 Run Alpha Radar</b> to begin'
            '</div>', unsafe_allow_html=True,
        )
        return

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    progress = st.progress(0)
    status   = st.empty()

    status.markdown('<div style="color:#7a8ba0;font-size:.82rem;">⏳ Fetching NSE market data…</div>', unsafe_allow_html=True)
    progress.progress(10)

    with st.spinner(f"Downloading {ticker}…"):
        data_result = run_data_agent(ticker, period, use_stub=use_stub)

    df         = data_result["raw_df"]
    ta_signals = data_result["ta_signals"]
    meta       = data_result["meta"]

    progress.progress(35)
    status.markdown('<div style="color:#7a8ba0;font-size:.82rem;">⏳ Scanning news signals…</div>', unsafe_allow_html=True)

    news_signals = fetch_news_signals(ticker=ticker)
    news_score   = get_news_score_for_stock(ticker, news_signals)

    progress.progress(55)
    status.markdown('<div style="color:#7a8ba0;font-size:.82rem;">⏳ Computing decision…</div>', unsafe_allow_html=True)

    decision    = compute_decision(ta_signals, news_score)

    progress.progress(70)
    status.markdown('<div style="color:#7a8ba0;font-size:.82rem;">⏳ Running AI agent…</div>', unsafe_allow_html=True)

    with st.spinner("AI analysis in progress…"):
        signal_data = run_ai_agent(ticker, ta_signals)

    progress.progress(95)
    time.sleep(0.1)
    progress.progress(100)
    time.sleep(0.1)
    progress.empty()
    status.empty()

    # ── Top Metrics ───────────────────────────────────────────────────────────
    def sg(row, col, default=0.0):
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    last      = df.iloc[-1]
    prev      = df.iloc[-2]
    price     = sg(last, "Close")
    prev_p    = sg(prev, "Close")
    pct_chg   = ((price - prev_p) / prev_p * 100) if prev_p else 0.0
    rsi_val   = sg(last, "RSI", None)
    atr_val   = sg(last, "ATR", None)
    vol       = sg(last, "Volume", 0.0)
    vol_ma    = sg(last, "Volume_MA", 0.0)
    vol_ratio = (vol / vol_ma) if vol_ma > 0 else 0.0
    ta_conf   = ta_signals.get("confidence_pct", 0.0)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        _metric_card("Last Price", f"₹{price:,.2f}",
                     sub=f"{'+' if pct_chg >= 0 else ''}{pct_chg:.2f}% today",
                     sub_color="#00e89a" if pct_chg >= 0 else "#ff3f5b")
    with m2:
        action_color = {"BUY":"#00e89a","SELL":"#ff3f5b","HOLD":"#4fa3ff"}.get(decision["action"],"#4fa3ff")
        st.markdown(
            f'<div class="mc"><div class="mc-label">Decision</div>'
            f'<div class="mc-val" style="color:{action_color};">{decision["action"]}</div>'
            f'<div class="mc-sub">{decision["confidence"]}% confidence</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "N/A"
        rsi_clr = ("#ff3f5b" if (rsi_val and rsi_val > 65) else "#00e89a" if (rsi_val and rsi_val < 35) else "#e2eafc")
        rsi_lbl = ("Overbought ⚠" if (rsi_val and rsi_val > 65) else "Oversold ✅" if (rsi_val and rsi_val < 35) else "Neutral zone")
        st.markdown(
            f'<div class="mc"><div class="mc-label">RSI (14)</div>'
            f'<div class="mc-val" style="color:{rsi_clr};">{rsi_str}</div>'
            f'<div class="mc-sub">{rsi_lbl}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        risk_color = {"LOW":"#00e89a","MEDIUM":"#f5c842","HIGH":"#ff3f5b"}.get(decision["risk"],"#f5c842")
        st.markdown(
            f'<div class="mc"><div class="mc-label">Risk Level</div>'
            f'<div class="mc-val" style="color:{risk_color};">{decision["risk"]}</div>'
            f'<div class="mc-sub">ATR: ₹{atr_val:.2f}' if atr_val else f'<div class="mc-sub">Daily volatility',
            unsafe_allow_html=True,
        )
    with m5:
        vr_clr = "#f5c842" if vol_ratio >= 1.5 else "#e2eafc"
        st.markdown(
            f'<div class="mc"><div class="mc-label">Volume Ratio</div>'
            f'<div class="mc-val" style="color:{vr_clr};">{vol_ratio:.2f}×</div>'
            f'<div class="mc-sub">{"⚡ Spike — High Activity!" if vol_ratio >= 1.5 else "vs 20-day avg"}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab4, tab5 = st.tabs([
        "📊 Analysis & Decision",
        "📰 Opportunity Radar",
        "🕰️ Backtesting",
        "💼 Portfolio",
    ])

    # ─────────── TAB 1: MAIN ANALYSIS ────────────────────────────────────────
    with tab1:
        col_chart, col_decision = st.columns([3, 2], gap="medium")

        with col_chart:
            st.markdown('<div class="sh">📈 Price Chart</div>', unsafe_allow_html=True)
            if PLOTLY_OK:
                st.plotly_chart(_build_main_chart(df, ticker), use_container_width=True,
                                config={"displayModeBar": True, "scrollZoom": True})
                if "RSI" in df.columns:
                    st.markdown('<div class="sh">RSI (14)</div>', unsafe_allow_html=True)
                    rsi_fig = _build_rsi_chart(df)
                    if rsi_fig:
                        st.plotly_chart(rsi_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.line_chart(df["Close"].rename(f"{ticker} Close ₹"))

        with col_decision:
            st.markdown('<div class="sh">🎯 AI Investment Decision</div>', unsafe_allow_html=True)
            _render_decision_card(decision, ticker, price)
            _render_ai_card(signal_data)

    # ─────────── TAB 2: OPPORTUNITY RADAR ────────────────────────────────────
    with tab2:
        st.markdown('<div class="sh">📰 Live Opportunity Radar — News-Based Signals</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.82rem;color:#7a8ba0;margin-bottom:16px;">'
            'AI scans news headlines for profit alerts, promoter buying, new contracts and more. '
            'These are early signals — always verify before acting.'
            '</div>', unsafe_allow_html=True,
        )

        pos_news = [n for n in news_signals if n["sentiment"] == "positive"]
        neg_news = [n for n in news_signals if n["sentiment"] == "negative"]
        neu_news = [n for n in news_signals if n["sentiment"] == "neutral"]

        col_pos, col_neg = st.columns([3, 2])
        with col_pos:
            st.markdown(f'<div class="sh">🟢 Positive Alerts ({len(pos_news)})</div>', unsafe_allow_html=True)
            for n in pos_news[:6]:
                weight_bar = int(abs(n["weight"]) * 100)
                desc_html = f'<div style="font-size:.75rem;color:#7a8ba0;margin-top:3px;line-height:1.5;">{n.get("description","")[:180]}</div>' if n.get("description") else ""
                url_html  = f'<a href="{n["url"]}" target="_blank" style="font-size:.68rem;color:#4fa3ff;text-decoration:none;">Read full article →</a>' if n.get("url") else ""
                st.markdown(f"""
<div class="opp-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;">
    <div style="flex:1;">
      <div class="opp-tag pos">OPPORTUNITY</div>
      <div class="opp-title">{n['name']}</div>
      <div class="opp-body">{n['headline']}</div>
      {desc_html}
      <div style="margin-top:8px;font-size:.72rem;color:#7a8ba0;">{n['source']} &nbsp;·&nbsp; {n['published']} &nbsp; {url_html}</div>
      <div style="margin-top:6px;background:rgba(0,232,154,.08);border-left:3px solid var(--green);
                  padding:6px 10px;border-radius:0 6px 6px 0;font-size:.78rem;color:var(--green);">
        ✅ {n['alert']}
      </div>
    </div>
    <div style="text-align:center;flex-shrink:0;">
      <div class="opp-score">{int(abs(n['weight'])*100)}%</div>
      <div style="font-size:.6rem;color:#7a8ba0;">signal</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with col_neg:
            st.markdown(f'<div class="sh">🔴 Risk Alerts ({len(neg_news)})</div>', unsafe_allow_html=True)
            for n in neg_news[:4]:
                st.markdown(f"""
<div class="opp-card" style="border-color:rgba(255,63,91,.2);">
  <div class="opp-tag neg">RISK ALERT</div>
  <div class="opp-title">{n['name']}</div>
  <div class="opp-body">{n['headline']}</div>
  <div style="margin-top:6px;font-size:.72rem;color:#7a8ba0;">{n['source']} &nbsp;·&nbsp; {n['published']}</div>
  <div style="margin-top:6px;background:rgba(255,63,91,.08);border-left:3px solid var(--red);
              padding:6px 10px;border-radius:0 6px 6px 0;font-size:.78rem;color:var(--red);">
    ⚠️ {n['alert']}
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="sh" style="margin-top:20px;">🔑 Opportunity Keywords Detected</div>', unsafe_allow_html=True)
            all_kws = []
            for n in news_signals:
                all_kws.extend(n.get("keywords", []))
            kw_counts = {}
            for kw in all_kws:
                kw_counts[kw] = kw_counts.get(kw, 0) + 1
            top_kws = sorted(kw_counts.items(), key=lambda x: -x[1])[:12]
            kw_html = " ".join(
                f'<span style="display:inline-block;background:rgba(245,200,66,.10);border:1px solid rgba(245,200,66,.2);'
                f'border-radius:20px;padding:3px 12px;font-size:.74rem;color:#f5c842;margin:3px 2px;">'
                f'{kw} ({cnt})</span>'
                for kw, cnt in top_kws
            )
            st.markdown(f'<div style="line-height:2;">{kw_html}</div>', unsafe_allow_html=True)

    # ─────────── TAB 4: BACKTESTING ──────────────────────────────────────────
    with tab4:
        st.markdown('<div class="sh">🕰️ Backtesting — How Has This Signal Performed?</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.82rem;color:#7a8ba0;margin-bottom:16px;">'
            'Applies the same signal rules on real historical price data (yfinance) to show you how reliable '
            'this strategy has been. Uses a <b style="color:#f5c842;">lower signal threshold</b> to capture '
            'more trades for a meaningful backtest.'
            '</div>', unsafe_allow_html=True,
        )

        if len(df) >= 100:
            with st.spinner("Running backtest on historical data…"):
                bt = run_backtest(df)

            if "error" not in bt:
                period_lbl = bt.get("period_label", "analysed period")
                hold_lbl   = bt.get("hold_days", 20)

                b1, b2, b3, b4 = st.columns(4)
                win_rate = bt["win_rate"]
                win_color = "#00e89a" if win_rate >= 60 else "#f5c842" if win_rate >= 45 else "#ff3f5b"

                with b1: st.markdown(
                    f'<div class="mc"><div class="mc-label">Win Rate</div>'
                    f'<div class="mc-val" style="color:{win_color};">{win_rate}%</div>'
                    f'<div class="mc-sub">Signals that were correct</div></div>', unsafe_allow_html=True)
                with b2: _metric_card("Total Signals", str(bt["total_signals"]), sub=f"{bt['winning_signals']} successful")
                with b3: st.markdown(
                    f'<div class="mc"><div class="mc-label">Avg Return</div>'
                    f'<div class="mc-val" style="color:{"#00e89a" if bt["avg_return_pct"] >= 0 else "#ff3f5b"};">'
                    f'{bt["avg_return_pct"]:+.2f}%</div>'
                    f'<div class="mc-sub">per signal ({hold_lbl}-bar hold)</div></div>', unsafe_allow_html=True)
                with b4: _metric_card("BUY Win Rate", f"{bt['buy_win_rate']}%", sub=f"SELL: {bt['sell_win_rate']}%")

                # Headline message
                msg_color = "#00e89a" if win_rate >= 60 else "#f5c842" if win_rate >= 45 else "#ff3f5b"
                st.markdown(f"""
<div style="background:rgba(245,200,66,.06);border:1px solid rgba(245,200,66,.15);
            border-radius:10px;padding:16px 20px;margin:16px 0;font-size:.9rem;color:{msg_color};">
  📊 {bt['message']}
</div>""", unsafe_allow_html=True)

                # Win rate bar
                st.markdown(f"""
<div style="margin:12px 0;">
  <div style="display:flex;justify-content:space-between;font-size:.72rem;color:#7a8ba0;margin-bottom:4px;">
    <span>Win Rate</span><span>{win_rate}%</span>
  </div>
  <div class="bt-bar-bg"><div class="bt-bar-fill" style="width:{win_rate}%;background:{win_color};"></div></div>
</div>""", unsafe_allow_html=True)

                # Sample trades chart
                if bt.get("sample_trades") and PLOTLY_OK:
                    st.markdown('<div class="sh">Recent Signal Outcomes</div>', unsafe_allow_html=True)
                    bt_fig = _build_backtest_chart(bt.get("sample_trades", []))
                    if bt_fig:
                        st.plotly_chart(bt_fig, use_container_width=True, config={"displayModeBar": False})

                # Sample trades table
                with st.expander("📋 View Recent Signal Trades"):
                    trades_df = pd.DataFrame(bt.get("sample_trades", []))
                    if not trades_df.empty:
                        trades_df["Result"] = trades_df["success"].map({True: "✅ Win", False: "❌ Loss"})
                        st.dataframe(
                            trades_df[["date_in","direction","return_pct","Result"]].rename(
                                columns={"date_in":"Date","direction":"Action","return_pct":"Return %","Result":"Outcome"}
                            ),
                            use_container_width=True, hide_index=True,
                        )
            else:
                st.warning(f"⚠️ {bt['error']}")
        else:
            st.warning("⚠️ Need at least 100 bars of data for backtesting. Use period '1y' or '2y'.")

    # ─────────── TAB 5: PORTFOLIO ANALYZER ───────────────────────────────────
    with tab5:
        st.markdown('<div class="sh">💼 Portfolio Analyzer & Future Profit Projector</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.82rem;color:#7a8ba0;margin-bottom:16px;">'
            'Enter your holdings and investment amount to get risk assessment, sector breakdown, '
            'and AI-powered future profit projections based on historical returns and current market signals.'
            '</div>', unsafe_allow_html=True,
        )

        st.markdown("#### Enter Your Portfolio")

        inv_col, _ = st.columns([2, 3])
        with inv_col:
            investment_amount = st.number_input(
                "💰 Total Investment Amount (₹)",
                min_value=10000, max_value=100000000,
                value=100000, step=10000,
                help="Enter the total amount you have invested or plan to invest across all stocks."
            )

        num_holdings = st.number_input("Number of stocks in portfolio", min_value=1, max_value=15, value=5)

        holdings = []
        cols_per_row = 3
        stock_list = list(_STOCKS.keys())

        for i in range(0, num_holdings, cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(row_cols):
                idx = i + j
                if idx < num_holdings:
                    with col:
                        sel   = st.selectbox(f"Stock {idx+1}", stock_list, index=min(idx, len(stock_list)-1), key=f"port_stock_{idx}")
                        alloc = st.number_input(f"Allocation %", min_value=0.0, max_value=100.0, value=round(100/num_holdings,1), key=f"port_alloc_{idx}")
                        holdings.append({"ticker": _STOCKS[sel], "name": sel, "allocation": alloc})

        analyze_btn = st.button("💼  Analyze Portfolio & Project Future Returns")
        if analyze_btn:
            total_entered = sum(h["allocation"] for h in holdings)
            if abs(total_entered - 100) > 1:
                st.warning(f"⚠️ Total allocation entered: **{total_entered:.1f}%** (ideally should be 100%). Results will be shown as proportional percentages.")

            result = analyze_portfolio(holdings)

            # Run per-stock TA for signal-adjusted projections
            with st.spinner("Running signal analysis for projection accuracy…"):
                ta_for_proj = {}
                for h in holdings:
                    try:
                        d = run_data_agent(h["ticker"], "1y", use_stub=use_stub)
                        ta_for_proj[h["ticker"]] = d["ta_signals"]
                    except Exception:
                        pass

            proj = project_portfolio_future(holdings, ta_results=ta_for_proj, investment_amount=investment_amount)

            if "error" not in result:
                # ── Risk Summary Banner ──
                risk_cls = {"LOW":"rp-low","MEDIUM":"rp-mid","HIGH":"rp-high"}.get(result["overall_risk"],"rp-mid")
                st.markdown(f"""
<div class="port-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div class="mc-label">Portfolio Summary</div>
      <div style="font-size:1.05rem;font-weight:700;margin:6px 0;">{result['summary']}</div>
      <div style="font-size:.82rem;color:#7a8ba0;">{result['num_sectors']} sectors &nbsp;·&nbsp; dominant: {result['dominant_sector']}</div>
    </div>
    <div><span class="risk-pill {risk_cls}">{result['overall_risk']} RISK</span></div>
  </div>
</div>""", unsafe_allow_html=True)

                # ── Future Profit Projections ──────────────────────────────────
                st.markdown('<div class="sh" style="margin-top:20px;">📈 Future Profit Projections</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:.78rem;color:#7a8ba0;margin-bottom:12px;">'
                    f'Based on sector historical returns + current TA signals. '
                    f'Investment: ₹{investment_amount:,.0f} · '
                    f'Expected annual return: <b style="color:#f5c842;">{proj["portfolio_base_return"]:.1f}%</b>'
                    f' (Bear: {proj["portfolio_bear_return"]:.1f}% | Bull: {proj["portfolio_bull_return"]:.1f}%)</div>',
                    unsafe_allow_html=True,
                )

                # Timeline cards
                t_cols = st.columns(len(proj["timeline"]))
                for col, t in zip(t_cols, proj["timeline"]):
                    gain_color = "#00e89a" if t["base_gain"] >= 0 else "#ff3f5b"
                    with col:
                        st.markdown(f"""
<div class="mc" style="text-align:center;padding:18px 12px;">
  <div class="mc-label">{t['years']} Year{'s' if t['years']>1 else ''}</div>
  <div style="font-size:1.15rem;font-weight:800;color:{gain_color};font-family:'JetBrains Mono',monospace;margin:6px 0;">
    {t['base_fmt']}
  </div>
  <div style="font-size:.75rem;color:#00e89a;margin-bottom:2px;">🟢 Bull: {t['bull_fmt']}</div>
  <div style="font-size:.75rem;color:#ff3f5b;">🔴 Bear: {t['bear_fmt']}</div>
  <div style="font-size:.65rem;color:#7a8ba0;margin-top:6px;">{t['base_gain']:+.1f}% gain (base)</div>
</div>""", unsafe_allow_html=True)

                # Projection Chart
                if PLOTLY_OK:
                    st.markdown('<div class="sh" style="margin-top:18px;">Portfolio Value Over Time</div>', unsafe_allow_html=True)
                    proj_fig = _build_projection_chart(proj["timeline"], investment_amount)
                    st.plotly_chart(proj_fig, use_container_width=True, config={"displayModeBar": False})

                # Per-stock projection table
                st.markdown('<div class="sh" style="margin-top:18px;">Per-Stock Projections (Base Case)</div>', unsafe_allow_html=True)
                if proj.get("by_stock"):
                    rows = []
                    for s in proj["by_stock"]:
                        rows.append({
                            "Stock":        s["name"],
                            "Sector":       s["sector"],
                            "Signal":       s["signal"],
                            "Allocation":   f"{s['allocation_pct']:.1f}%",
                            "Invested (₹)": f"₹{s['invested']:,.0f}",
                            "1Y Value":     f"₹{s['base_1y']:,.0f}",
                            "3Y Value":     f"₹{s['base_3y']:,.0f}",
                            "5Y Value":     f"₹{s['base_5y']:,.0f}",
                            "Ann. Return":  f"{s['base_annual_pct']:.1f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Disclaimer
                st.markdown(f"""
<div class="disc" style="margin-top:16px;">
  ⚖️ {proj.get('disclaimer', '')}
</div>""", unsafe_allow_html=True)

                # ── Sector Analysis ────────────────────────────────────────────
                st.markdown('<div class="sh" style="margin-top:24px;">📊 Sector Risk Analysis</div>', unsafe_allow_html=True)
                p_col1, p_col2 = st.columns([2, 3])
                with p_col1:
                    if PLOTLY_OK:
                        st.plotly_chart(_build_sector_pie(result["sector_distribution"]),
                                        use_container_width=True, config={"displayModeBar": False})
                    else:
                        for s, pct in result["sector_distribution"].items():
                            st.markdown(f"**{s}**: {pct}%")

                with p_col2:
                    st.markdown('<div class="sh">Diversification Suggestions</div>', unsafe_allow_html=True)
                    for sug in result["suggestions"]:
                        st.markdown(f"""
<div style="background:rgba(245,200,66,.06);border:1px solid rgba(245,200,66,.15);
            border-radius:8px;padding:12px 16px;margin-bottom:8px;font-size:.85rem;color:var(--txt);">
  💡 {sug}
</div>""", unsafe_allow_html=True)

                    st.markdown('<div class="sh" style="margin-top:16px;">Concentration by Sector</div>', unsafe_allow_html=True)
                    for sector, pct in sorted(result["sector_distribution"].items(), key=lambda x: -x[1]):
                        bar_color = "#ff3f5b" if pct > 40 else "#f5c842" if pct > 25 else "#00e89a"
                        st.markdown(f"""
<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;font-size:.78rem;color:#7a8ba0;margin-bottom:3px;">
    <span>{sector}</span><span style="font-family:'JetBrains Mono',monospace;font-weight:600;">{pct}%</span>
  </div>
  <div class="bt-bar-bg"><div class="bt-bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>
</div>""", unsafe_allow_html=True)

            else:
                st.error(f"❌ {result['error']}")

    # ── Footer ────────────────────────────────────────────────────────────────

    dr = meta.get("date_range", ("N/A","N/A"))
    st.markdown(f"""
<div style="margin-top:28px;padding:14px 20px;background:rgba(255,255,255,.015);
            border-radius:10px;border:1px solid var(--border);
            font-size:.68rem;color:#3a4455;display:flex;gap:20px;flex-wrap:wrap;align-items:center;">
  <span>📅 {dr[0]} → {dr[1]}</span>
  <span>📊 {meta.get('row_count','?')} bars · {period}</span>
  <span>{'🔌 Offline' if use_stub else '🌐 Live NSE'}</span>
  <span>🐍 Python | pandas_ta: {'✅' if TA_OK else '⚠️ fallback'}</span>
  <span style="margin-left:auto;color:#f5c842;">Alpha Radar v2.0 · ET AI Hackathon 2026</span>
</div>""", unsafe_allow_html=True)


# =============================================================================
if __name__ == "__main__":
    main()
