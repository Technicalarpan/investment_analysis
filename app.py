import os
import re
import json
import time
import random
import logging
import warnings
import datetime
from typing import Optional
from data_engine import *
from decision_making import *

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