import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ---------------------------------------------
# APP INFO
# ---------------------------------------------
st.set_page_config(page_title="Crypto Short Scanner v2", layout="wide")
st.title("📉 Crypto Short Scanner v2")
st.caption("Selecteert potentiële short-kansen op basis van RSI, Bollinger Bands en volume-analyse (Yahoo Finance).")

# ---------------------------------------------
# INSTELLINGEN
# ---------------------------------------------
top_n = st.sidebar.slider("Aantal top coins (CoinGecko)", 10, 100, 50)
period = st.sidebar.selectbox("Periode", ["7d", "14d", "30d"])
interval = st.sidebar.selectbox("Interval", ["1h", "2h", "4h"])
rsi_thresh = st.sidebar.slider("RSI overbought-grens", 50, 90, 70)
vol_mult = st.sidebar.slider("Volume multiplier t.o.v. gemiddelde", 1.0, 5.0, 2.0)
st.sidebar.markdown("---")
st.sidebar.caption("Laatste update: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---------------------------------------------
# FUNCTIES
# ---------------------------------------------
def get_top_coins_coingecko(n=50):
    """Haalt top n coins op uit CoinGecko en vertaalt ze naar Yahoo-tickers"""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": False,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = []
        for c in data:
            name = c.get("name", "")
            symbol = c.get("symbol", "").upper()
            yahoo_ticker = f"{symbol}-USD"
            results.append((name, yahoo_ticker))
        return results
    except Exception as e:
        st.error(f"❌ Fout bij ophalen CoinGecko: {e}")
        return []

def fetch_ohlcv_yahoo(ticker, period="7d", interval="1h"):
    """Haalt OHLCV-data op via Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df["datetime"] = df.index
        return df
    except Exception:
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_ohlcv(df):
    """Bereken indicatoren"""
    df["rsi"] = calc_rsi(df["Close"])
    df["ma20"] = df["Close"].rolling(20).mean()
    df["std"] = df["Close"].rolling(20).std()
    df["bb_high"] = df["ma20"] + 2 * df["std"]
    df["vol_avg"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    return {
        "close": last["Close"],
        "rsi": last["rsi"],
        "bb_high": last["bb_high"],
        "vol": last["Volume"],
        "avg_vol": last["vol_avg"],
        "datetime": last.name
    }

def score_signal(metrics, rsi_thresh, vol_mult):
    """Bereken short-score op basis van condities"""
    score = 0
    reasons = []
    if metrics["rsi"] > rsi_thresh:
        score += 1
        reasons.append("RSI hoog")
    if metrics["close"] > metrics["bb_high"]:
        score += 1
        reasons.append("Bollinger bovenkant")
    if metrics["vol"] > metrics["avg_vol"] * vol_mult:
        score += 1
        reasons.append("Hoog volume")
    return score, reasons

# ---------------------------------------------
# OPHALEN COINS
# ---------------------------------------------
coins = get_top_coins_coingecko(top_n)

if not coins:
    st.warning("⚠️ Geen coins gevonden van CoinGecko, probeer later opnieuw.")
    st.stop()

# ---------------------------------------------
# ANALYSE LOOP
# ---------------------------------------------
results = []
progress = st.progress(0)
status_text = st.empty()

for i, (name, ticker) in enumerate(coins):
    progress.progress((i + 1) / len(coins))
    status_text.text(f"Analyseren: {name} ({ticker})...")

    df = fetch_ohlcv_yahoo(ticker, period=period, interval=interval)
    if df is None or len(df) < 25:
        results.append({
            "name": name,
            "ticker": ticker,
            "score": 0,
            "reasons": "no_data",
            "close": np.nan,
            "rsi": np.nan,
            "bb_high": np.nan,
            "vol": np.nan,
            "avg_vol": np.nan,
            "datetime": np.nan,
        })
        continue

    metrics = analyze_ohlcv(df)
    score, reasons = score_signal(metrics, rsi_thresh, vol_mult)

    results.append({
        "name": name,
        "ticker": ticker,
        "score": score,
        "reasons": ", ".join(reasons) if reasons else "geen_signaal",
        "close": metrics["close"],
        "rsi": metrics["rsi"],
        "bb_high": metrics["bb_high"],
        "vol": metrics["vol"],
        "avg_vol": metrics["avg_vol"],
        "datetime": metrics["datetime"],
    })

progress.empty()
status_text.empty()

# ---------------------------------------------
# RESULTATEN
# ---------------------------------------------
df_out = pd.DataFrame(results)
df_out = df_out.sort_values(by=["score", "rsi"], ascending=[False, False])

st.subheader("📊 Analyse Resultaten")
st.dataframe(df_out, use_container_width=True)

top_signals = df_out[df_out["score"] >= 2]
if not top_signals.empty:
    st.success(f"✅ {len(top_signals)} potentiële short-signalen gevonden!")
    st.dataframe(top_signals[["name", "ticker", "score", "reasons", "rsi", "close"]])
else:
    st.info("Geen sterke short-signalen gevonden met de huidige instellingen.")

st.caption("© 2025 Crypto Short Scanner — Gebouwd met Streamlit & Yahoo Finance")
