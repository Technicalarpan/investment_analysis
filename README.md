# AI for Indian Investors (Alpha Radar)

A practical, end-to-end project that tries to solve a real problem in the Indian stock market — most retail investors have data, but not clarity.

This project builds an **intelligence layer** on top of market data to help answer a simple question:

> “What should I actually do with my money right now?”

---

## 🧩 Problem

India has 14+ crore demat accounts, but a large number of investors:

- Follow tips instead of data
- Miss important signals (filings, insider activity, etc.)
- Don’t fully understand charts or technical indicators
- Manage portfolios based on gut feeling

Even with platforms like NSE or ET Markets, the *interpretation layer* is missing.

---

## 💡 What this project does

Instead of just showing data, this system tries to:

- Convert data → signals  
- Signals → decisions  
- Decisions → explanations  

So the output is not just numbers, but **actionable insight**.

---

## 🚀 Features

### 1. Opportunity Radar
Tracks multiple sources like:
- Corporate filings
- Quarterly results
- Bulk/block deals
- Insider trades
- News & sentiment shifts

👉 Output: Daily “opportunity signals”

---

### 2. Chart Pattern Intelligence
Detects:
- Breakouts
- Reversals
- RSI-based signals
- Trend changes

👉 Also explains in simple language *why* a pattern matters.

---

### 3. Decision Engine (Buy / Hold / Sell)
Combines:
- Technical indicators
- Trend strength
- Signal confidence

👉 Outputs a final decision with reasoning.

---

### 4. Portfolio Analysis
- Allocation insights
- Sector distribution
- Performance charts

---

### 5. Backtesting Engine
- Test strategies on past data
- Understand if a signal actually works

---

### 6. AI Explanation Layer
- Converts logic into human-readable reasoning
- Example:
  “Stock shows breakout with strong volume and RSI support — probability of upward move is high.”

---

## 🏗️ Project Structure

```
.
├── app.py                  # main Flask app
├── data_engine.py          # data fetching + preprocessing
├── decision_making.py      # buy/sell/hold logic
├── portfolio_ana.py        # charts and portfolio insights
├── backtesting.py          # strategy testing
├── multi_scanner.py        # scan multiple stocks
├── opportunaty_radar.py    # signal detection
├── ai_agent_llm.py         # explanation layer (LLM)
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/ai-investor.git
cd ai-investor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```



### 3. Setup environment variables

Create a `.env` file and add required keys (if any):

```
API_KEY=your_key_here
```

---

## ▶️ Run the project

```bash
python -m streamlit run app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 📌 Example Usage

You can:

- Analyze a stock
- See Buy/Hold/Sell signal
- Check charts and indicators
- Run backtesting
- Ask AI for explanation

Example query:
> “Is it a good time to invest ₹10,000 in this stock?”

---

## 🛠️ Tech Stack

- Python
- Flask
- Pandas / NumPy
- Plotly / Matplotlib
- Basic LLM integration

---

## ⚠️ Disclaimer

This is not financial advice.

The project is built for learning and experimentation. Always do your own research before investing real money.

---

## 👨‍💻 Author

Arpan Mukherjee  
NIT Durgapur (Mechanical + Minor in CS)  
IIT Madras (BS in Data Science & AI)

Soumyadeep Dey  
NIT Durgapur (Mechanical Engineering)

Sangram Bask  
NIT Durgapur (Computer Science and Engineering)

Anjali Chaudhury  
NIT Durgapur (Computer Science and Engineering)


---

## 🙌 Notes

This project was built with a focus on:
- clarity over complexity  
- practical usefulness  
- explainable decisions  

If you’re improving it, feel free to extend:
- better signals
- real-time alerts
- portfolio optimization

---

