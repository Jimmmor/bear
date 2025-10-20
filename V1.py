import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="🐻 Crypto Short Scanner Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark crypto theme
st.markdown("""
<style>
    .main {background-color: #0d1117;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3748 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .short-signal-high {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    .short-signal-medium {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #f97316;
    }
    .short-signal-low {
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #6b7280;
    }
    h1 {color: #ef4444 !important;}
    h2 {color: #f87171 !important;}
    h3 {color: #fca5a5 !important;}
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🐻 CRYPTO SHORT SCANNER PRO")
    st.caption("Advanced Technical Analysis for Short Opportunities")

# Sidebar settings
st.sidebar.header("⚙️ Scanner Settings")
top_n = st.sidebar.slider("📊 Coins to scan", 10, 100, 50, 5)
timeframe = st.sidebar.selectbox("⏱️ Timeframe", ["1h", "2h", "4h", "1d"], index=2)
period_map = {"1h": "7d", "2h": "14d", "4h": "30d", "1d": "60d"}
period = period_map[timeframe]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Signal Filters")
min_score = st.sidebar.slider("Minimum Short Score", 0, 100, 60)
rsi_thresh = st.sidebar.slider("RSI Overbought", 60, 90, 70)
vol_increase = st.sidebar.slider("Volume Spike (%)", 100, 500, 200)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5min)")
if auto_refresh:
    st.sidebar.info("Scanner refreshes every 5 minutes")

st.sidebar.markdown("---")
st.sidebar.caption(f"🕐 Last update: {datetime.now().strftime('%H:%M:%S')}")

# Top coins list
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
    """Calculate advanced technical indicators"""
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
    df['ema_20'] = close.ewm(span=20).mean()
    df['ema_50'] = close.ewm(span=50).mean()
    
    # Bollinger Bands
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_std'] = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Volume
    df['vol_sma'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_sma']
    
    # ATR for volatility
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Price momentum
    df['momentum'] = close.pct_change(10) * 100
    
    # Support/Resistance (recent highs/lows)
    df['recent_high'] = high.rolling(20).max()
    df['recent_low'] = low.rolling(20).min()
    
    return df

def analyze_short_opportunity(df, rsi_thresh, vol_increase):
    """Analyze if asset is a good short candidate"""
    if df is None or len(df) < 50:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    signals = []
    risk_level = "LOW"
    
    # 1. RSI Overbought (weight: 15)
    if last['rsi'] > rsi_thresh:
        points = min(15, int((last['rsi'] - rsi_thresh) / 2))
        score += points
        signals.append(f"🔴 RSI Overbought ({last['rsi']:.1f})")
    
    # 2. Price above Bollinger Upper (weight: 15)
    if last['Close'] > last['bb_upper']:
        score += 15
        signals.append("🔴 Above BB Upper")
    
    # 3. Bearish MACD (weight: 15)
    if last['macd'] < last['macd_signal'] and prev['macd'] > prev['macd_signal']:
        score += 15
        signals.append("🔴 MACD Bearish Cross")
    elif last['macd_hist'] < 0:
        score += 8
        signals.append("⚠️ MACD Bearish")
    
    # 4. Price below EMA (trend confirmation) (weight: 15)
    if last['Close'] < last['ema_20']:
        score += 8
        signals.append("⚠️ Below EMA20")
    if last['Close'] < last['ema_50']:
        score += 7
        signals.append("⚠️ Below EMA50")
    
    # 5. Volume spike (weight: 10)
    if last['vol_ratio'] > (vol_increase / 100):
        score += 10
        signals.append(f"🔴 Volume Spike ({last['vol_ratio']:.1f}x)")
    
    # 6. Near resistance (weight: 10)
    if last['Close'] >= last['recent_high'] * 0.98:
        score += 10
        signals.append("🔴 At Resistance")
    
    # 7. Bearish momentum (weight: 10)
    if last['momentum'] < -3:
        score += 10
        signals.append(f"🔴 Bearish Momentum ({last['momentum']:.1f}%)")
    
    # 8. Bearish divergence detection (weight: 10)
    if len(df) >= 20:
        price_trend = (last['Close'] - df.iloc[-20]['Close']) / df.iloc[-20]['Close']
        rsi_trend = last['rsi'] - df.iloc[-20]['rsi']
        if price_trend > 0 and rsi_trend < 0:
            score += 10
            signals.append("🔴 Bearish Divergence")
    
    # Risk level
    if score >= 75:
        risk_level = "HIGH"
    elif score >= 50:
        risk_level = "MEDIUM"
    
    # Calculate stop loss and take profit
    stop_loss = last['recent_high'] * 1.02
    take_profit_1 = last['Close'] * 0.95
    take_profit_2 = last['Close'] * 0.90
    
    return {
        'score': min(100, score),
        'signals': signals,
        'risk_level': risk_level,
        'rsi': last['rsi'],
        'macd': last['macd'],
        'macd_signal': last['macd_signal'],
        'price': last['Close'],
        'ema_20': last['ema_20'],
        'ema_50': last['ema_50'],
        'bb_upper': last['bb_upper'],
        'volume': last['Volume'],
        'vol_ratio': last['vol_ratio'],
        'atr': last['atr'],
        'stop_loss': stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'potential_gain': ((last['Close'] - take_profit_2) / last['Close']) * 100,
        'risk_reward': ((last['Close'] - take_profit_2) / (stop_loss - last['Close'])) if stop_loss > last['Close'] else 0
    }

# Main scanning loop
st.markdown("### 🔍 Scanning Market...")
coins = get_top_coins(top_n)
results = []

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
        analysis = analyze_short_opportunity(df, rsi_thresh, vol_increase)
        
        if analysis and analysis['score'] >= min_score:
            results.append({
                'name': name,
                'ticker': ticker,
                'score': analysis['score'],
                'risk': analysis['risk_level'],
                'price': analysis['price'],
                'rsi': analysis['rsi'],
                'signals': analysis['signals'],
                'stop_loss': analysis['stop_loss'],
                'tp1': analysis['take_profit_1'],
                'tp2': analysis['take_profit_2'],
                'potential': analysis['potential_gain'],
                'rr_ratio': analysis['risk_reward'],
                'df': df
            })
    except:
        continue

progress_bar.empty()
status.empty()

# Display results
if not results:
    st.warning("⚠️ No short opportunities found with current settings. Try adjusting filters.")
    st.stop()

# Sort by score
results = sorted(results, key=lambda x: x['score'], reverse=True)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Opportunities Found", len(results))
with col2:
    high_risk = len([r for r in results if r['risk'] == 'HIGH'])
    st.metric("🔥 High Confidence", high_risk)
with col3:
    avg_score = np.mean([r['score'] for r in results])
    st.metric("📊 Avg Score", f"{avg_score:.0f}")
with col4:
    avg_potential = np.mean([r['potential'] for r in results])
    st.metric("💰 Avg Potential", f"{avg_potential:.1f}%")

st.markdown("---")

# Display top opportunities
st.markdown("### 🎯 Top Short Opportunities")

for result in results[:10]:
    risk_class = f"short-signal-{result['risk'].lower()}"
    
    with st.container():
        st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.markdown(f"### {result['name']}")
            st.markdown(f"**{result['ticker']}**")
            st.metric("Score", f"{result['score']}/100")
        
        with col2:
            st.metric("Current Price", f"${result['price']:.4f}")
            st.metric("RSI", f"{result['rsi']:.1f}")
            if result['rr_ratio'] > 0:
                st.metric("Risk/Reward", f"{result['rr_ratio']:.2f}")
        
        with col3:
            st.markdown("**🎯 Targets:**")
            st.write(f"🛑 Stop Loss: ${result['stop_loss']:.4f}")
            st.write(f"✅ TP1 (-5%): ${result['tp1']:.4f}")
            st.write(f"✅ TP2 (-10%): ${result['tp2']:.4f}")
        
        with col4:
            risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}
            st.markdown(f"### {risk_emoji[result['risk']]}")
            st.markdown(f"**{result['risk']}**")
            st.markdown(f"**{result['potential']:.1f}%** gain")
        
        # Signals
        st.markdown("**📡 Signals:**")
        st.write(" • ".join(result['signals'][:5]))
        
        # Mini chart
        if 'df' in result:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                              vertical_spacing=0.05, shared_xaxes=True)
            
            df = result['df'].tail(50)
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'
            ), row=1, col=1)
            
            # EMAs
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_20'], name='EMA20',
                                    line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA50',
                                    line=dict(color='blue', width=1)), row=1, col=1)
            
            # BB
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                                    line=dict(color='red', width=1, dash='dash')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                                    line=dict(color='purple', width=2)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=400, showlegend=False, 
                            template='plotly_dark',
                            margin=dict(l=0, r=0, t=0, b=0))
            fig.update_xaxes(rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

# Full table view
st.markdown("### 📋 All Results")
df_display = pd.DataFrame([{
    'Rank': idx + 1,
    'Coin': r['name'],
    'Ticker': r['ticker'],
    'Score': f"{r['score']}/100",
    'Risk': r['risk'],
    'Price': f"${r['price']:.4f}",
    'RSI': f"{r['rsi']:.1f}",
    'Potential': f"{r['potential']:.1f}%",
    'R/R': f"{r['rr_ratio']:.2f}" if r['rr_ratio'] > 0 else 'N/A'
} for idx, r in enumerate(results)])

st.dataframe(df_display, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("⚠️ Disclaimer: This tool is for educational purposes only. Always do your own research and never invest more than you can afford to lose.")

if auto_refresh:
    import time
    time.sleep(300)
    st.rerun()
