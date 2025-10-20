import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="🐻 Crypto Bearish Probability Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0d1117;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3748 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .coin-row {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3748 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        border: 1px solid #30363d;
        cursor: pointer;
        transition: all 0.3s;
    }
    .coin-row:hover {
        border: 1px solid #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }
    .prob-high {color: #dc2626; font-weight: bold; font-size: 24px;}
    .prob-medium {color: #ea580c; font-weight: bold; font-size: 24px;}
    .prob-low {color: #10b981; font-weight: bold; font-size: 24px;}
    h1 {color: #ef4444 !important; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_coin' not in st.session_state:
    st.session_state.selected_coin = None
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# Header
st.markdown("# 🐻 CRYPTO BEARISH PROBABILITY SCANNER")
st.markdown("<p style='text-align: center; color: #9ca3af;'>AI-Powered Probability Analysis for Short Positions</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Scanner Settings")
top_n = st.sidebar.slider("📊 Number of Coins", 10, 100, 50, 5)
timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1h", "4h", "1d"], index=1)
period_map = {"1h": "7d", "4h": "30d", "1d": "90d"}
period = period_map[timeframe]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Probability Weights")
weight_rsi = st.sidebar.slider("RSI Weight", 0, 30, 15)
weight_macd = st.sidebar.slider("MACD Weight", 0, 30, 15)
weight_ema = st.sidebar.slider("EMA Trend Weight", 0, 30, 15)
weight_bb = st.sidebar.slider("Bollinger Weight", 0, 30, 10)
weight_volume = st.sidebar.slider("Volume Weight", 0, 20, 10)
weight_momentum = st.sidebar.slider("Momentum Weight", 0, 20, 10)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Scan", type="primary"):
    st.session_state.scan_results = None
    st.session_state.selected_coin = None

# Top coins
def get_top_coins(n=50):
    coins = [
        ("Bitcoin", "BTC-USD"), ("Ethereum", "ETH-USD"), ("BNB", "BNB-USD"),
        ("Solana", "SOL-USD"), ("XRP", "XRP-USD"), ("Cardano", "ADA-USD"),
        ("Avalanche", "AVAX-USD"), ("Dogecoin", "DOGE-USD"), ("Polkadot", "DOT-USD"),
        ("Polygon", "MATIC-USD"), ("Chainlink", "LINK-USD"), ("Litecoin", "LTC-USD"),
        ("Uniswap", "UNI-USD"), ("Near Protocol", "NEAR-USD"), ("Arbitrum", "ARB-USD"),
        ("Stellar", "XLM-USD"), ("Cosmos", "ATOM-USD"), ("Filecoin", "FIL-USD"),
        ("Aptos", "APT-USD"), ("Optimism", "OP-USD"), ("VeChain", "VET-USD"),
        ("Algorand", "ALGO-USD"), ("Hedera", "HBAR-USD"), ("Internet Computer", "ICP-USD"),
        ("Render", "RNDR-USD"), ("The Graph", "GRT-USD"), ("Aave", "AAVE-USD"),
        ("Immutable", "IMX-USD"), ("SUI", "SUI-USD"), ("Maker", "MKR-USD"),
        ("Tezos", "XTZ-USD"), ("EOS", "EOS-USD"), ("Flow", "FLOW-USD"),
        ("Theta", "THETA-USD"), ("Gala", "GALA-USD"), ("Zcash", "ZEC-USD"),
        ("Synthetix", "SNX-USD"), ("Curve DAO", "CRV-USD"), ("Fantom", "FTM-USD"),
        ("1inch", "1INCH-USD"), ("Enjin Coin", "ENJ-USD"), ("Chiliz", "CHZ-USD"),
        ("Lido DAO", "LDO-USD"), ("Pepe", "PEPE-USD"), ("Shiba Inu", "SHIB-USD"),
        ("Bonk", "BONK-USD"), ("Floki", "FLOKI-USD"), ("JasmyCoin", "JASMY-USD"),
        ("Quant", "QNT-USD"), ("Kava", "KAVA-USD"), ("Dash", "DASH-USD")
    ]
    return coins[:n]

# Technical indicators
def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None
    
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # EMAs
    df['ema_9'] = close.ewm(span=9).mean()
    df['ema_20'] = close.ewm(span=20).mean()
    df['ema_50'] = close.ewm(span=50).mean()
    df['ema_200'] = close.ewm(span=200).mean()
    
    # Bollinger Bands
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Volume
    df['vol_sma'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_sma']
    
    # Price momentum
    df['roc'] = close.pct_change(10) * 100
    
    # Support/Resistance
    df['recent_high'] = high.rolling(20).max()
    df['recent_low'] = low.rolling(20).min()
    
    return df

def calculate_bearish_probability(df, weights):
    if df is None or len(df) < 50:
        return None
    
    last = df.iloc[-1]
    prev_5 = df.iloc[-5]
    
    prob_components = {}
    total_prob = 0
    
    # 1. RSI Analysis (0-weight_rsi%)
    if pd.notna(last['rsi']):
        if last['rsi'] > 70:
            rsi_prob = weights['rsi'] * (min(last['rsi'] - 70, 30) / 30)
            prob_components['RSI Overbought'] = rsi_prob
            total_prob += rsi_prob
        elif last['rsi'] < 50:
            prob_components['RSI Neutral'] = 0
    
    # 2. MACD Analysis (0-weight_macd%)
    if pd.notna(last['macd']) and pd.notna(last['macd_signal']):
        if last['macd'] < last['macd_signal']:
            macd_strength = abs(last['macd_hist']) / abs(last['macd']) if last['macd'] != 0 else 0
            macd_prob = weights['macd'] * min(macd_strength * 2, 1)
            prob_components['MACD Bearish'] = macd_prob
            total_prob += macd_prob
    
    # 3. EMA Trend Analysis (0-weight_ema%)
    ema_score = 0
    if pd.notna(last['ema_20']) and last['Close'] < last['ema_20']:
        ema_score += 0.25
    if pd.notna(last['ema_50']) and last['Close'] < last['ema_50']:
        ema_score += 0.25
    if pd.notna(last['ema_200']) and last['Close'] < last['ema_200']:
        ema_score += 0.25
    if pd.notna(last['ema_20']) and pd.notna(last['ema_50']) and last['ema_20'] < last['ema_50']:
        ema_score += 0.25
    
    if ema_score > 0:
        ema_prob = weights['ema'] * ema_score
        prob_components['Downtrend'] = ema_prob
        total_prob += ema_prob
    
    # 4. Bollinger Bands (0-weight_bb%)
    if pd.notna(last['bb_upper']) and last['Close'] > last['bb_upper']:
        bb_distance = (last['Close'] - last['bb_upper']) / last['bb_upper']
        bb_prob = weights['bb'] * min(bb_distance * 10, 1)
        prob_components['Above BB'] = bb_prob
        total_prob += bb_prob
    
    # 5. Volume Analysis (0-weight_volume%)
    if pd.notna(last['vol_ratio']):
        # High volume on down days = bearish
        price_change = (last['Close'] - prev_5['Close']) / prev_5['Close']
        if price_change < 0 and last['vol_ratio'] > 1.5:
            vol_prob = weights['volume'] * min(last['vol_ratio'] / 3, 1)
            prob_components['Distribution'] = vol_prob
            total_prob += vol_prob
    
    # 6. Momentum (0-weight_momentum%)
    if pd.notna(last['roc']):
        if last['roc'] < -2:
            mom_prob = weights['momentum'] * min(abs(last['roc']) / 10, 1)
            prob_components['Negative Momentum'] = mom_prob
            total_prob += mom_prob
    
    # Normalize to 0-100%
    max_possible = sum(weights.values())
    probability = min(100, (total_prob / max_possible) * 100) if max_possible > 0 else 0
    
    return {
        'probability': probability,
        'components': prob_components,
        'indicators': {
            'rsi': last['rsi'],
            'macd': last['macd'],
            'macd_signal': last['macd_signal'],
            'price': last['Close'],
            'ema_20': last['ema_20'],
            'ema_50': last['ema_50'],
            'bb_upper': last['bb_upper'],
            'vol_ratio': last['vol_ratio'],
            'roc': last['roc']
        }
    }

# Scan coins
if st.session_state.scan_results is None:
    st.markdown("### 🔍 Scanning Market...")
    coins = get_top_coins(top_n)
    results = []
    
    weights = {
        'rsi': weight_rsi,
        'macd': weight_macd,
        'ema': weight_ema,
        'bb': weight_bb,
        'volume': weight_volume,
        'momentum': weight_momentum
    }
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    for idx, (name, ticker) in enumerate(coins):
        progress_bar.progress((idx + 1) / len(coins))
        status.text(f"Analyzing {name}... ({idx + 1}/{len(coins)})")
        
        try:
            df = yf.download(ticker, period=period, interval=timeframe, progress=False)
            if df.empty or len(df) < 50:
                continue
            
            df = calculate_indicators(df)
            analysis = calculate_bearish_probability(df, weights)
            
            if analysis:
                results.append({
                    'name': name,
                    'ticker': ticker,
                    'probability': analysis['probability'],
                    'components': analysis['components'],
                    'indicators': analysis['indicators'],
                    'df': df
                })
        except:
            continue
    
    progress_bar.empty()
    status.empty()
    
    st.session_state.scan_results = sorted(results, key=lambda x: x['probability'], reverse=True)

# Display results
results = st.session_state.scan_results

if not results:
    st.warning("⚠️ No data available. Try adjusting settings.")
    st.stop()

# Summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Coins Scanned", len(results))
with col2:
    high_prob = len([r for r in results if r['probability'] >= 70])
    st.metric("🔴 High Risk (≥70%)", high_prob)
with col3:
    medium_prob = len([r for r in results if 50 <= r['probability'] < 70])
    st.metric("🟠 Medium (50-70%)", medium_prob)
with col4:
    avg_prob = np.mean([r['probability'] for r in results])
    st.metric("📈 Avg Probability", f"{avg_prob:.1f}%")

st.markdown("---")

# Coin list
if st.session_state.selected_coin is None:
    st.markdown("### 🎯 Bearish Probability Ranking")
    st.caption("Click on any coin for detailed analysis")
    
    for result in results:
        prob = result['probability']
        prob_class = 'prob-high' if prob >= 70 else 'prob-medium' if prob >= 50 else 'prob-low'
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.markdown(f"**{result['name']}** ({result['ticker']})")
        with col2:
            st.markdown(f"<span class='{prob_class}'>{prob:.1f}%</span>", unsafe_allow_html=True)
        with col3:
            components = list(result['components'].keys())[:3]
            st.caption(" • ".join(components) if components else "Low bearish signals")
        with col4:
            if st.button("📊", key=result['ticker']):
                st.session_state.selected_coin = result
                st.rerun()
        
        st.markdown("---")

# Detailed view
else:
    coin = st.session_state.selected_coin
    
    if st.button("← Back to List"):
        st.session_state.selected_coin = None
        st.rerun()
    
    st.markdown(f"# {coin['name']} ({coin['ticker']})")
    st.markdown("---")
    
    # Probability breakdown
    col1, col2 = st.columns([1, 2])
    
    with col1:
        prob = coin['probability']
        prob_color = '#dc2626' if prob >= 70 else '#ea580c' if prob >= 50 else '#10b981'
        st.markdown(f"<div style='text-align: center; padding: 30px; background: {prob_color}; border-radius: 15px;'>"
                   f"<h1 style='color: white; margin: 0;'>{prob:.1f}%</h1>"
                   f"<p style='color: white; margin: 5px 0 0 0;'>BEARISH PROBABILITY</p></div>", 
                   unsafe_allow_html=True)
        
        st.markdown("### 📊 Contributing Factors")
        for component, value in coin['components'].items():
            st.metric(component, f"{value:.1f}%")
    
    with col2:
        st.markdown("### 📈 Key Indicators")
        ind = coin['indicators']
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Price", f"${ind['price']:.4f}")
            st.metric("RSI", f"{ind['rsi']:.1f}")
        with col_b:
            st.metric("EMA 20", f"${ind['ema_20']:.4f}")
            st.metric("EMA 50", f"${ind['ema_50']:.4f}")
        with col_c:
            st.metric("MACD", f"{ind['macd']:.4f}")
            st.metric("Volume Ratio", f"{ind['vol_ratio']:.2f}x")
    
    st.markdown("---")
    
    # Charts
    df = coin['df'].tail(100)
    
    # Main price chart
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        shared_xaxes=True,
        subplot_titles=('Price & EMAs', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_20'], name='EMA 20',
                            line=dict(color='orange', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50',
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_200'], name='EMA 200',
                            line=dict(color='purple', width=2)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                            line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                            line=dict(color='green', width=1, dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                            line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD',
                            line=dict(color='blue', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                            line=dict(color='orange', width=2)), row=3, col=1)
    
    colors = ['red' if x < 0 else 'green' for x in df['macd_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histogram',
                        marker_color=colors), row=3, col=1)
    
    fig.update_layout(
        height=900,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.caption("⚠️ This probability is based on technical analysis only. Always do your own research.")
