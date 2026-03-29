import os
import re
import json
import time
import random
import logging
import warnings
import datetime
from typing import Optional

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