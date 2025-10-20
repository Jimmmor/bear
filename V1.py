import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ---------------------------------------------
# APP INFO
# ---------------------------------------------
st.set_page_config(page_title="Crypto Short Scanner v2.2", layout="wide")
st.title("📉 Crypto Short Scanner v2.2")
st.caption("Analyseert top coins voor mogelijke short-signalen via RSI, Bollinger Bands & Volume (Yahoo Finance).")

# ---------------------------------------------
# INSTELLINGEN
# ---------------------------------------------
top_n = st.sidebar.slider("Aantal top coins", 10, 100, 50)
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
    """Statische lijst van top coins om API-fouten te vermijden."""
    base_list = [
        ("Bitcoin", "BTC-USD"), ("Ethereum", "ETH-USD"), ("BNB", "BNB-USD"),
        ("Solana", "SOL-USD"), ("XRP", "XRP-USD"), ("Dogecoin", "DOGE-USD"),
        ("Cardano", "ADA-USD"), ("Avalanche", "AVAX-USD"), ("Polkadot", "DOT-USD"),
        ("Chainlink", "LINK-USD"), ("Uniswap", "UNI-USD"), ("Litecoin", "LTC-USD"),
        ("Polygon", "MATIC-USD"), ("Internet Computer", "ICP-USD"),
        ("Aptos", "APT-USD"), ("Near Protocol", "NEAR-USD"),
        ("Arbitrum", "ARB-USD"), ("Stellar", "XLM-USD"), ("Filecoin", "FIL-USD"),
        ("Cosmos", "ATOM-USD"), ("VeChain", "VET-USD"), ("Optimism", "OP-USD"),
        ("Immutable", "IMX-USD"), ("SUI", "SUI-USD"), ("Maker", "MKR-USD"),
        ("Render", "RNDR-USD"), ("The Graph", "GRT-USD"), ("Algorand", "ALGO-USD"),
        ("Hedera", "HBAR-USD"), ("Aave", "AAVE-USD"), ("Quant", "QNT-USD"),
        ("Tezos", "XTZ-USD"), ("Theta", "THETA-USD"), ("EOS", "EOS-USD"),
        ("Kava", "KAVA-USD"), ("Flow", "FLOW-USD"), ("Gala", "GALA-USD"),
        ("Chiliz", "CHZ-USD"), ("Zcash", "ZEC-USD"), ("Curve DAO", "CRV-USD"),
        ("Synthetix", "SNX-USD"), ("Fantom", "FTM-USD"), ("1inch", "1INCH-USD"),
        ("Balancer", "BAL-USD"), ("Dash", "DASH-USD"), ("Enjin Coin", "ENJ-USD"),
        ("Lido DAO", "LDO-USD"), ("Bonk", "BONK-USD"), ("JasmyCoin", "JASMY-USD"),
        ("Pepe", "PEPE-USD"), ("Shiba Inu", "SHIB-USD"),
    ]
    return base_list[:n]

def fetch_ohlcv_yahoo(ticker, period="7d", interval="1h"):
    """Haalt OHLCV-data op via Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        df["datetime"] = df.index
        return df
    except Exception:
        return None

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_ohlcv(df):
    """Bereken indicatoren en geef laatste waarden terug als floats"""
    if df is None or df.empty or len(df) < 20:
        return {k: np.nan for k in ["close", "rsi", "bb_high", "vol", "avg_vol", "datetime"]}

    df["rsi"] = calc_rsi(df["Close"])
    df["ma20"] = df["Close"].rolling(20).mean()
    df["std"] = df["Close"].rolling(20).std()
    df["bb_high"] = df["ma20"] + 2 * df["std"]
    df["vol_avg"] = df["Volume"].rolling(20).mean()

    last = df.tail(1).iloc[0]

    return {
        "close": float(last.get("Close", np.nan)),
        "rsi": float(last.get("rsi", np.nan)),
        "bb_high": float(last.get("bb_high", np.nan)),
        "vol": float(last.get("Volume", np.nan)),
        "avg_vol": float(last.get("vol_avg", np.nan)),
        "datetime": last.name
    }

def score_signal(metrics, rsi_thresh, vol_mult):
    """Bereken short-score op basis van condities"""
    score = 0
    reasons = []

    rsi = metrics.get("rsi", np.nan)
    close = metrics.get("close", np.nan)
    bb_high = metrics.get("bb_high", np.nan)
    vol = metrics.get("vol", np.nan)
    avg_vol = metrics.get("avg_vol", np.nan)

    if pd.notna(rsi) and rsi > rsi_thresh:
        score += 1
        reasons.append("RSI hoog")
    if pd.notna(close) and pd.notna(bb_high) and close > bb_high:
        score += 1
        reasons.append("Bollinger bovenkant")
    if pd.notna(vol) and pd.notna(avg_vol) and vol > avg_vol * vol_mult:
        score += 1
        reasons.append("Hoog volume")

    return score, reasons

# ---------------------------------------------
# OPHALEN COINS
# ---------------------------------------------
coins = get_top_coins_coingecko(top_n)

if not coins:
    st.warning("⚠️ Geen coins beschikbaar.")
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
            "name": name, "ticker": ticker, "score": 0, "reasons": "no_data",
            "close": np.nan, "rsi": np.nan, "bb_high": np.nan,
            "vol": np.nan, "avg_vol": np.nan, "datetime": np.nan,
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
if "score" in df_out.columns and "rsi" in df_out.columns:
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
