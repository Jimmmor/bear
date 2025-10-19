
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import ta
import time
from datetime import datetime

st.set_page_config(page_title="Crypto Short Screener v2 (Yahoo+CoinGecko)", layout="wide")

st.title("📉 Crypto Short Screener v2")
st.markdown("Detecteert potentieel oververhitte crypto-activa (short-candidates) met data van CoinGecko + Yahoo Finance.")

# --------------------
# Sidebar / Controls
# --------------------
with st.sidebar:
    st.header("Instellingen")
    top_n = st.number_input("Top N coins (market cap)", min_value=10, max_value=100, value=50, step=10)
    period = st.selectbox("Data periode (yfinance)", options=["1mo","3mo","6mo"], index=1)
    interval = st.selectbox("Interval", options=["1h","4h","1d"], index=0)
    rsi_thresh = st.slider("RSI threshold (overbought)", min_value=60, max_value=95, value=80)
    vol_mult = st.slider("Volume spike multiplier", min_value=1.1, max_value=3.0, value=1.5, step=0.1)
    run_button = st.button("Run scan")
    st.markdown("---")
    st.write("Tips:")
    st.write("• Gebruik een lagere top_n als Streamlit Cloud timeouts optreden.")
    st.write("• Interval 1h geeft meer signalen maar kost meer API-calls.")

# --------------------
# Helper functions
# --------------------

@st.cache_data(ttl=300)
def get_top_coins_coingecko(n=50):
    """Return list of tuples (name, symbol) for top-n coins by market cap using CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": False,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    for name, ticker in coins:
        metrics = None
        score, reasons = 0, []
        try:
            df = fetch_ohlcv_yahoo(ticker, period=period, interval=interval)
            if df is not None and len(df) >= 50:
                metrics = analyze_ohlcv(df)
                score, reasons = score_signal(metrics, rsi_thresh, vol_mult)
        except Exception:
            pass  # bij fout blijven score=0 en reasons=[]
    
        # Voeg altijd een consistente dict toe
        results.append({
            "name": name,
            "ticker": ticker,
            "score": score,
            "reasons": ", ".join(reasons) if reasons else "no_signal",
            "close": metrics.get("close") if metrics else np.nan,
            "rsi": metrics.get("rsi") if metrics else np.nan,
            "bb_high": metrics.get("bb_high") if metrics else np.nan,
            "vol": metrics.get("vol") if metrics else np.nan,
            "avg_vol": metrics.get("avg_vol") if metrics else np.nan,
            "datetime": metrics.get("datetime") if metrics else np.nan,
        })
    
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(by=["score","rsi"], ascending=[False, False])
    

@st.cache_data(ttl=120)
def fetch_ohlcv_yahoo(ticker, period="3mo", interval="1h"):
    """Fetch OHLCV for a ticker via yfinance. Return DataFrame or None on failure."""
    try:
        # yfinance accepts period & interval
        df = yf.download(ticker, period=period, interval=interval, progress=False, rounding=True)
        if df is None or df.empty:
            return None
        # Ensure necessary columns
        if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
            return None
        df = df.dropna()
        return df
    except Exception:
        return None


def analyze_ohlcv(df, rsi_window=14, bb_window=20, atr_window=14, vol_window=20):
    """Compute indicators and return last-row metrics and a short-bias score."""
    out = {}
    try:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        vol = df["Volume"].astype(float)

        rsi = ta.momentum.RSIIndicator(close, window=rsi_window).rsi()
        bb = ta.volatility.BollingerBands(close, window=bb_window, window_dev=2)
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        atr = ta.volatility.AverageTrueRange(high, low, close, window=atr_window).average_true_range()
        avg_vol = vol.rolling(vol_window).mean()

        last = df.iloc[-1]
        last_idx = df.index[-1]

        last_rsi = float(rsi.iloc[-1]) if not rsi.isna().all() else np.nan
        last_bb_high = float(bb_high.iloc[-1]) if not bb_high.isna().all() else np.nan
        last_atr = float(atr.iloc[-1]) if not atr.isna().all() else np.nan
        last_avg_vol = float(avg_vol.iloc[-1]) if not np.isnan(avg_vol.iloc[-1]) else np.nan

        price = float(last["Close"])
        vol_now = float(last["Volume"]) if not np.isnan(last["Volume"]) else np.nan

        out = {
            "datetime": last_idx,
            "close": price,
            "rsi": last_rsi,
            "bb_high": last_bb_high,
            "atr": last_atr,
            "vol": vol_now,
            "avg_vol": last_avg_vol,
        }
    except Exception:
        return None
    return out


def score_signal(metrics, rsi_thresh=80, vol_mult=1.5):
    """Simple scoring: +1 per condition met. Returns score and explanation list."""
    score = 0
    reasons = []
    if metrics is None:
        return 0, ["no_data"]
    if not np.isnan(metrics.get("rsi", np.nan)) and metrics["rsi"] > rsi_thresh:
        score += 1
        reasons.append(f"RSI>{rsi_thresh}")
    if not np.isnan(metrics.get("bb_high", np.nan)) and metrics["close"] > metrics["bb_high"]:
        score += 1
        reasons.append("Price>BB_high")
    if not np.isnan(metrics.get("avg_vol", np.nan)) and metrics["avg_vol"] > 0:
        if metrics.get("vol", 0) > vol_mult * metrics["avg_vol"]:
            score += 1
            reasons.append(f"Vol>{vol_mult}xAvg")
    return score, reasons

# --------------------
# Main scanning logic
# --------------------

if run_button:
    with st.spinner("Haalt top coins op van CoinGecko..."):
        try:
            coins = get_top_coins_coingecko(top_n)
        except Exception as e:
            st.error(f"Fout bij ophalen CoinGecko: {e}")
            st.stop()

    st.info(f"Scanning {len(coins)} coins — elke coin haalt data via Yahoo (kan enkele minuten duren)")

    results = []
    progress_bar = st.progress(0)
    total = len(coins)
    i = 0
    for name, ticker in coins:
        i += 1
        progress_bar.progress(int(i/total*100))
        # polite sleep to avoid rate limits
        time.sleep(0.5)
        metrics = None
        try:
            df = fetch_ohlcv_yahoo(ticker, period=period, interval=interval)
            if df is None or len(df) < 50:
                # try a few common symbol fixes (WBTC -> WBTC-USD, or use coin id?)
                results.append({"name": name, "ticker": ticker, "status": "no_data"})
                continue
            metrics = analyze_ohlcv(df)
            score, reasons = score_signal(metrics, rsi_thresh, vol_mult)
            results.append({
                "name": name,
                "ticker": ticker,
                "score": score,
                "reasons": ", ".join(reasons),
                "close": metrics.get("close"),
                "rsi": metrics.get("rsi"),
                "bb_high": metrics.get("bb_high"),
                "vol": metrics.get("vol"),
                "avg_vol": metrics.get("avg_vol"),
                "datetime": metrics.get("datetime")
            })
        except Exception as e:
            results.append({"name": name, "ticker": ticker, "status": f"error: {e}"})

    df_out = pd.DataFrame(results)
    if df_out.empty:
        st.warning("Geen resultaten — mogelijk timeouts of geen data voor gekozen interval")
    else:
        df_out = df_out.sort_values(by=["score","rsi"], ascending=[False, False])
        st.subheader("Resultaten")
        st.dataframe(df_out.reset_index(drop=True))

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"short_scan_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv")

        st.markdown("---")
        st.write("Legenda: score = aantal triggers (RSI, Price>BB_high, Volume spike). Hoger = sterker short bias.")

else:
    st.info("Pas instellingen in de sidebar en klik op 'Run scan' om de top coins te analyseren.")


# --------------------
# Footer notes
# --------------------
st.markdown("---")
st.caption("Opmerking: dit is een **research tool** — geen handelsadvies. Gebruik kleine posities en altijd stop-loss.")


# requirements.txt (copy-paste contents into a separate file when deploying)
# ---------------------------------
# streamlit
# pandas
# numpy
# requests
# yfinance
# ta
# ---------------------------------
