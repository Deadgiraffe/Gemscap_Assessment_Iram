import streamlit as st
import redis
import json
import pandas as pd
import duckdb
import vectorbt as vbt
import os
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DB_PATH = "market_data.duckdb"

st.set_page_config(layout="wide", page_title="RT-QIS Dashboard")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
<style>
    /* Global Reset */
    .stApp { background-color: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif; }
    
    /* Headers */
    h1, h2, h3, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { 
        color: #ffffff !important; 
        font-family: 'Inter', sans-serif; 
        letter-spacing: -0.5px; 
    }
    
    /* Metrics */
    .stMetric { background-color: #1e222d; border: 1px solid #2a2e39; padding: 15px; border-radius: 4px; }
    .stMetric label { color: #787b86 !important; font-size: 14px; }
    div[data-testid="stMetricValue"] { color: #f0f3fa !important; font-size: 24px; font-weight: 600; }
    
    /* Sidebar Specifics */
    section[data-testid="stSidebar"] { background-color: #1e222d; border-right: 1px solid #2a2e39; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #d1d4dc !important; }
    
    /* Input Fields (Selectbox, NumberInput, etc.) */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2a2e39 !important;
        color: #d1d4dc !important;
        border-color: #434651 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #2962ff !important;
    }
    /* Dropdown Menu Items */
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background-color: #1e222d !important;
        color: #d1d4dc !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #2a2e39 !important;
    }
    
    /* Buttons */
    /* Buttons (Normal and Download) */
    .stButton button, .stDownloadButton button {
        background-color: #2962ff !important; 
        color: white !important; 
        border: none;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    .stButton button:hover, .stDownloadButton button:hover { 
        background-color: #1e53e5 !important; 
        color: white !important;
        border-color: #1e53e5 !important;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] { background-color: #1e222d; border: 1px solid #2a2e39; }
    
    /* File Uploader */
    [data-testid='stFileUploader'] {
        background-color: #1e222d;
        border: 1px dashed #434651;
        border-radius: 5px;
        padding: 10px;
    }
    [data-testid='stFileUploader'] section {
        background-color: #1e222d !important;
    }
    [data-testid='stFileUploader'] button {
        background-color: #2962ff !important;
        color: white !important;
        border: none;
    }
    /* Small text in uploader */
    [data-testid='stFileUploader'] small {
        color: #787b86 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

st.title("RT-QIS: Alpha Generation Console")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Strategy Configuration")
    strategy_mode = st.radio("Hedge Ratio Model", ["Kalman Filter", "Rolling OLS (River)", "Robust (Huber)"])
    
    st.header("Parameters")
    primary_ticker = st.selectbox("Analysis Target", ["BTCUSDT", "ETHUSDT"])
    secondary_ticker = "ETHUSDT" if primary_ticker == "BTCUSDT" else "BTCUSDT"
    
    chart_mode = st.selectbox("Chart Layout", ["Overlay", "Stacked", "Single"])
    
    z_score_thresh = st.slider("Z-Score Threshold", 1.0, 5.0, 2.0, 0.1)
    timeframe = st.selectbox("Chart Timeframe", ["1m", "5m"])
    
    st.divider()
    # Fix: User requested pause for zooming
    pause_updates = st.checkbox("Freeze/Pause Live View", value=False, help="Stop updates to zoom/inspect the chart", key="pause_chk")
    
    st.divider()
    st.header("Analytics")
    if st.button("Run ADF Test (Stationarity)"):
        run_adf = True
    else:
        run_adf = False
    
    st.divider()
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload OHLC CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_df'] = up_df
            st.success(f"Loaded {len(up_df)} rows")
        except Exception as e:
             st.error(f"Upload failed: {e}")
             
    st.divider()
    st.header("Backtesting")
    if st.button("Run Instant Backtest"):
        st.session_state['run_backtest'] = True
    
    st.divider()


# --- DATA FETCHING ---
@st.cache_data(ttl=5)
def fetch_historical_candles(symbol='btcusdt', timeframe='1m'):
    try:
        table = "candles_5m" if timeframe == "5m" else "candles_1m"
        # Adjust limit based on timeframe for better visibility
        # 1m: Last 6 hours (360 mins)
        # 5m: Last 24 hours (288 * 5 mins)
        limit = 288 if timeframe == "5m" else 360
        
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute(f"SELECT time, open, high, low, close, volume FROM {table} WHERE symbol='{symbol}' ORDER BY time DESC LIMIT {limit}").df()
        con.close()
        if not df.empty:
            df = df.sort_values('time')
            # Fix: Ensure datetime64[ns] before converting to Unix Seconds
            df['time'] = df['time'].astype('datetime64[ns]').astype('int64') // 10**9
            return df
    except Exception:
        pass
    return pd.DataFrame()

def fetch_live_data():
    r = get_redis_client()
    
    # Fetch Data Streams
    # Reduced to 10k to balance performance (prevent UI freeze) and data continuity
    ticks_data = r.xrevrange("stream:ticks", count=10000)
    # Analytics: Fetch more to try filling gaps
    analytics_data = r.xrevrange("stream:analytics", count=2000)
    alerts_data = r.xrevrange("stream:alerts", count=5)
    
    # 1. Parse Ticks for Live Candle
    tick_rows_btc = []
    tick_rows_eth = []
    liq_rows = []
    latest_price = 0.0
    latency = 0.0
    
    for mid, data in ticks_data:
        d = json.loads(data['json'])
        ts_sec = d['timestamp'] / 1000
        sym = d['symbol'].lower()
        
        if d['type'] == 'trade':
            p = float(d['data']['p'])
            q = float(d['data']['q'])
            if sym == 'btcusdt':
                tick_rows_btc.append({"time": ts_sec, "price": p, "volume": q})
                if latest_price == 0: 
                    latest_price = p
                    latency = d.get('latency', 0)
            elif sym == 'ethusdt':
                tick_rows_eth.append({"time": ts_sec, "price": p, "volume": q})
                
        elif d['type'] == 'bookTicker' and sym == 'btcusdt':
            liq_rows.append({
                "time": ts_sec,
                "bid_sz": float(d['data']['B']), "ask_sz": float(d['data']['A']),
                "bid_px": float(d['data']['b']), "ask_px": float(d['data']['a'])
            })
            
    # --- HELPER TO MERGE LIVE + HIST ---
    def get_full_candle_df(tick_rows, symbol):
        df_ticks = pd.DataFrame(tick_rows)
        df_live = pd.DataFrame()
        
        # Determine Resample Rule
        # timeframe var is in global scope from sidebar
        tf_map = {'1m': '1min', '5m': '5min'}
        rule = tf_map.get(timeframe, '1min')
        
        if not df_ticks.empty:
            df_ticks['datetime'] = pd.to_datetime(df_ticks['time'], unit='s')
            live_agg = df_ticks.set_index('datetime').resample(rule).agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            })
            live_agg.columns = ['open', 'high', 'low', 'close', 'volume']
            df_live = live_agg.reset_index()
            df_live['time'] = df_live['datetime'].astype('int64') // 10**9
            
        df_hist = fetch_historical_candles(symbol, timeframe=timeframe)
        
        if not df_hist.empty and not df_live.empty:
            last_hist_time = df_hist['time'].iloc[-1]
            df_live = df_live[df_live['time'] >= last_hist_time]
            full = pd.concat([df_hist[df_hist['time'] < df_live['time'].min()], df_live]).sort_values('time')
        elif not df_hist.empty: full = df_hist
        else: full = df_live
        
        if not full.empty:
            full = full[full['low'] > 0].copy()
            full['datetime'] = pd.to_datetime(full['time'], unit='s')
        return full, df_ticks

    full_btc, df_ticks_btc = get_full_candle_df(tick_rows_btc, 'btcusdt')
    full_eth, df_ticks_eth = get_full_candle_df(tick_rows_eth, 'ethusdt')

    # 4. Analytics
    analytics_rows = []
    for mid, data in analytics_data:
        try:
            state = json.loads(data['json'])
            if strategy_mode == "Kalman Filter": 
                fv = state['kalman']['fv']
                alpha = state['kalman']['alpha']
                z_val = state['kalman']['z']
            elif strategy_mode == "Rolling OLS (River)": 
                fv = state['river']['fv']
                alpha = state['river']['alpha']
                # River/Huber might not compute Z in worker, calculate here if missing
                z_val = (latest_price - fv) if latest_price else 0 
            else: 
                fv = state['huber']['fv']
                alpha = state['huber']['alpha']
                z_val = (latest_price - fv) if latest_price else 0
            
            # Filter bad analytics too
            if fv > 0: 
                analytics_rows.append({
                    "time": int(mid.split('-')[0])/1000, 
                    "fair_value": fv, 
                    "beta_active": state['kalman']['beta'] if strategy_mode=="Kalman Filter" else (state['river']['beta'] if strategy_mode=="Rolling OLS (River)" else state['huber']['beta']),
                    "alpha": alpha,
                    "z_score": z_val
                })
        except: pass
        
    df_ana = pd.DataFrame(analytics_rows)
    
    # 5. Liquidity
    df_liq = pd.DataFrame(liq_rows)
    if not df_liq.empty: df_liq = df_liq.sort_values('time')

    # 6. Parse Alerts (Worker Side)
    alerts = []
    for mid, data in alerts_data:
        try:
             if 'msg' in data: 
                 alerts.append({"id": mid, "msg": data['msg']})
             elif 'json' in data: 
                 a = json.loads(data['json'])
                 alerts.append({"id": mid, "msg": a['msg'], "ts": a.get('ts')})
        except: pass

    return full_btc, full_eth, df_ticks_btc, df_ticks_eth, df_ana, df_liq, alerts, latest_price, latency

@st.fragment(run_every=1.0)
def live_view():
    
    # Logic: Pause Handling
    # If paused, we skip fetch, but we MUST load from Session State
    is_paused = st.session_state.get('pause_updates', False) # Default to false if not set manually
    # Actually, sidebar sets the widget key? No, st.checkbox returns value.
    # We can't access widget value inside fragment easily if it's outside?
    # Actually, st.session_state is global.
    # But for sidebar widget, we need to know its key if we want to access it here.
    # I didn't set a key above. I assigned to `pause_updates` variable.
    # But `live_view` is a fragment, it re-runs.
    # Streamlit fragments capture closure scope?
    # No, `live_view` is called at the end of script.
    # But `run_every` implies it runs independently.
    # It does NOT capture `pause_updates` from the main script run unless passed or in session state.
    # I WILL USE A SESSION KEY FOR THE CHECKBOX.
    
    # Re-read: I need to assign key='pause_chk' to the checkbox in the main script area.
    # I will modify the previous block (lines 92-96) to add key.
    # WAIT, I am replacing lines 93-216 in one go.
    # Let's ensure the checkbox has a key.
    
    # ... inside live_view ...
    should_pause = st.session_state.get('pause_chk', False)

    if not should_pause:
        df_btc, df_eth, df_ticks_btc, df_ticks_eth, df_ana_new, df_liq, alerts, price, latency = fetch_live_data()
        
        # Select active ticks
        df_ticks = df_ticks_btc if primary_ticker == "BTCUSDT" else df_ticks_eth
        
        # Accumulate Analytics (Fair Value) History
        # Because Redis cleans up old keys, but we want a nice line chart.
        if not df_ana_new.empty:
            df_ana_new['datetime'] = pd.to_datetime(df_ana_new['time'], unit='s')
            
            existing_ana = st.session_state.get('accumulated_ana', pd.DataFrame())
            if not existing_ana.empty:
                # Merge and Dedupe
                combined = pd.concat([existing_ana, df_ana_new]).drop_duplicates(subset=['time']).sort_values('time')
                # Keep last 10000 points to avoid memory leak
                combined = combined.tail(10000)
                st.session_state['accumulated_ana'] = combined
                df_ana = combined
            else:
                st.session_state['accumulated_ana'] = df_ana_new
                df_ana = df_ana_new
        else:
            df_ana = st.session_state.get('accumulated_ana', pd.DataFrame())
            
        # Update Cache
        st.session_state['last_candles'] = df_btc
        st.session_state['last_eth'] = df_eth
        st.session_state['last_raw_ticks'] = df_ticks
        st.session_state['last_ana'] = df_ana
        st.session_state['last_liq'] = df_liq
        st.session_state['last_price'] = price
        st.session_state['last_latency'] = latency
        # Alerts
        if alerts:
            st.session_state['last_alerts_list'] = alerts

    else:
        # PAUSED: Load from Cache
        df_btc = st.session_state.get('last_candles', pd.DataFrame())
        df_eth = st.session_state.get('last_eth', pd.DataFrame())
        df_ticks = st.session_state.get('last_raw_ticks', pd.DataFrame())
        df_ana = st.session_state.get('last_ana', pd.DataFrame())
        df_liq = st.session_state.get('last_liq', pd.DataFrame())
        price = st.session_state.get('last_price', 0.0)
        latency = st.session_state.get('last_latency', 0.0)
        alerts = [] # Don't show new alerts when paused

    # Fallback if empty at start
    if df_btc.empty:
        st.info("Initializing Data Feed...")
        return

    # --- ALERT TOASTS ---
    # ... (rest of logic) ...
    if alerts:
        last_seen = st.session_state.get('last_alert_id', '0')
        newest_id = last_seen
        for a in reversed(alerts): # Oldest to Newest
             if a['id'] > last_seen:
                 st.toast(f"🚨 {a['msg']}", icon="⚠️")
                 newest_id = a['id']
        st.session_state['last_alert_id'] = newest_id

    # --- TECHNICAL INDICATORS ---
    # --- TECHNICAL INDICATORS ---
    if len(df_btc) > 0:
        # Calculate Indicators (min_periods=1 ensures visibility even with >1 data point)
        df_btc['sma'] = df_btc['close'].rolling(window=20, min_periods=1).mean()
        df_btc['std'] = df_btc['close'].rolling(window=20, min_periods=1).std()
        df_btc['upper'] = df_btc['sma'] + (z_score_thresh * df_btc['std'])
        df_btc['lower'] = df_btc['sma'] - (z_score_thresh * df_btc['std'])
        
        # Determine Volume Colors
        df_btc['vol_color'] = np.where(df_btc['close'] >= df_btc['open'], '#089981', '#f23645')

    if len(df_eth) > 0:
        df_eth['sma'] = df_eth['close'].rolling(window=20, min_periods=1).mean()
        df_eth['std'] = df_eth['close'].rolling(window=20, min_periods=1).std()
        df_eth['upper'] = df_eth['sma'] + (z_score_thresh * df_eth['std'])
        df_eth['lower'] = df_eth['sma'] - (z_score_thresh * df_eth['std'])
        df_eth['vol_color'] = np.where(df_eth['close'] >= df_eth['open'], '#089981', '#f23645')

    # --- PLOTTING ---
    fig = make_subplots(
        rows=3 if chart_mode == "Stacked" else 2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2] if chart_mode == "Stacked" else [0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]] if chart_mode == "Stacked" else 
              [[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Helper for Timezone
    def to_ist(series):
        # Assuming series is pd.DatetimeIndex or Series in UTC-naive (which implies UTC for unit='s')
        # We first localize to UTC, then convert to Asia/Kolkata
        return series.dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

    # Convert Times & formatting
    # Convert Times & formatting
    # Revert to standard DateTime for robustness
    if not df_btc.empty: df_btc['datetime_ist'] = to_ist(df_btc['datetime'])
    if not df_eth.empty: df_eth['datetime_ist'] = to_ist(df_eth['datetime'])
    
    # Analytics needs aggregation to match Candle frequency (prevent vertical noise)
    if not df_ana.empty: 
        df_ana['datetime_ist'] = to_ist(df_ana['datetime'])
        # Sort chronologically
        df_ana = df_ana.sort_values('datetime')

    # Dynamic Data Selection
    if primary_ticker == "BTCUSDT":
        df_prim, df_sec = df_btc, df_eth
        prim_name, sec_name = "BTC/USDT", "ETH/USDT"
        plot_fv = df_ana
    else:
        df_prim, df_sec = df_eth, df_btc
        prim_name, sec_name = "ETH/USDT", "BTC/USDT"
        plot_fv = pd.DataFrame() 

    # 1. Primary Candlestick
    fig.add_trace(go.Candlestick(
        x=df_prim.get('datetime_ist', []),
        open=df_prim['open'], high=df_prim['high'], low=df_prim['low'], close=df_prim['close'],
        name=prim_name,
        increasing_line_color='#089981', decreasing_line_color='#f23645'
    ), row=1, col=1)

    # 2. Indicators (Primary)
    # 2. Indicators (Primary)
    if 'sma' in df_prim:
        # Fix: Change SMA color to White to stand out from Bollinger Bands
        fig.add_trace(go.Scatter(x=df_prim['datetime_ist'], y=df_prim['sma'], line=dict(color='#FFFFFF', width=2), name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_prim['datetime_ist'], y=df_prim['upper'], line=dict(color='rgba(0, 229, 255, 0.3)', width=1), name='Upper BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_prim['datetime_ist'], y=df_prim['lower'], line=dict(color='rgba(0, 229, 255, 0.3)', width=1), name='Lower BB'), row=1, col=1)

    # 3. Fair Value (Overlay on Primary) - DISABLED per user request
    # if not df_ana.empty and primary_ticker == "BTCUSDT":
    #     fig.add_trace(go.Scatter(x=df_ana['datetime_ist'], y=df_ana['fair_value'], 
    #                             line=dict(color='#2979FF', width=3), name='Fair Value (BTC)'), row=1, col=1)
                                
    # 4. Secondary Asset Logic
    if chart_mode != "Single" and not df_sec.empty:
        sec_trace = go.Candlestick(
            x=df_sec['datetime_ist'],
            open=df_sec['open'], high=df_sec['high'], low=df_sec['low'], close=df_sec['close'],
            name=sec_name,
            increasing_line_color='#FFEE58', decreasing_line_color='#EF6C00'
        )
        
        if chart_mode == "Stacked":
            fig.add_trace(sec_trace, row=2, col=1)
        else: # Overlay
            sec_trace.yaxis = "y2"
            fig.add_trace(sec_trace, row=1, col=1, secondary_y=True)

    # 5. Volume
    vol_row = 3 if chart_mode == "Stacked" else 2
    if 'volume' in df_prim:
        fig.add_trace(go.Bar(
            x=df_prim['datetime_ist'], y=df_prim['volume'],
            marker_color=df_prim.get('vol_color', '#089981'), name='Volume'
        ), row=vol_row, col=1)

    # --- LAYOUT STYLING ---
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#E0E0E0', family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="white"), bgcolor="rgba(0,0,0,0.5)"),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        uirevision='no_reset' 
    )
    
    # Grid Styling
    grid_color = 'rgba(42, 46, 57, 0.5)'
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, gridwidth=1, zeroline=False, 
                     type='date')
    
    # Y-Axis Styling
    fig.update_yaxes(title=f'{prim_name} Price', showgrid=True, gridcolor=grid_color, gridwidth=1, zeroline=False, side='right', row=1, col=1)
    if chart_mode == "Overlay":
        fig.update_yaxes(title=f'{sec_name} Price', showgrid=False, zeroline=False, side='left', secondary_y=True, row=1, col=1)
    elif chart_mode == "Stacked":
        fig.update_yaxes(title=f'{sec_name} Price', showgrid=True, gridcolor=grid_color, side='right', row=2, col=1)

    # --- LAYOUT INSIDE FRAGMENT ---
    col_charts, col_tape = st.columns([3, 1])

    with col_charts:
        st.subheader(f"{prim_name} vs {strategy_mode} Fair Value")
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False, 'scrollZoom': True})
    
        # --- LIQUIDITY VIS ---
        st.markdown("### Market Depth Pressure (Bid vs Ask Size)")
        if not df_liq.empty:
            df_liq['datetime'] = pd.to_datetime(df_liq['time'], unit='s')
            df_liq['datetime_ist'] = df_liq['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            
            # 1. Rename logic
            liq_plot_df = df_liq.rename(columns={'bid_sz': 'Bid Size', 'ask_sz': 'Ask Size'})
            
            df_liq_melt = liq_plot_df.melt(id_vars=['datetime_ist'], value_vars=['Bid Size', 'Ask Size'], var_name='Type', value_name='Size')
            
            liq_fig = px.area(df_liq_melt, x='datetime_ist', y='Size', color='Type',
                          color_discrete_map={'Bid Size': '#00e676', 'Ask Size': '#ff1744'},
                          title=None)
            liq_fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font=dict(color='white'),
                              showlegend=True,
                              legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)"),
                              xaxis=dict(showgrid=False, tickformat='%H:%M:%S', type='date', title="Time (IST)"), 
                              yaxis=dict(showgrid=True, gridcolor='#333'))
            st.plotly_chart(liq_fig, width="stretch")

    # --- TAPE & METRICS ---
    with col_tape:
        st.markdown("### Market State")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric(f"{primary_ticker} Price", f"{df_prim['close'].iloc[-1]:.2f}" if not df_prim.empty else "Init...")
        col_m2.metric("Latency", f"{abs(latency):.1f} ms")
        
        if not df_ana.empty:
            last_row = df_ana.iloc[-1]
            beta_val = last_row.get('beta_active', 0.0)
            # Handle alpha/beta for inversion
            alpha_val = last_row.get('alpha', 0.0)
            fv_val_btc = last_row['fair_value']
            
            if primary_ticker == "ETHUSDT" and abs(beta_val) > 1e-9:
                # Inverted FV: ETH_fv = (BTC_price - Alpha) / Beta
                # Use current BTC price from sec
                btc_price = df_sec['close'].iloc[-1] if not df_sec.empty else 0
                fv_val = (btc_price - alpha_val) / beta_val
            else:
                fv_val = fv_val_btc

            st.metric(f"{primary_ticker} Fair Value", f"{fv_val:.2f}")
            st.metric("Hedge Beta (BTC/ETH)", f"{beta_val:.4f}")
        else:
            st.metric(f"{primary_ticker} Fair Value", "Init...")
            st.metric("Hedge Beta", "Init...")
        
        st.markdown("### Recent Trades")
        if not df_ticks.empty and 'time' in df_ticks:
            tape_df = df_ticks[['time', 'price', 'volume']].sort_values('time', ascending=False).head(15)
            tape_df = tape_df.copy()
            # Fix: Display IST Time
            tape_df['datetime'] = pd.to_datetime(tape_df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            tape_df['time'] = tape_df['datetime'].dt.strftime('%H:%M:%S')
            st.dataframe(tape_df[['time', 'price', 'volume']], hide_index=True)
        else:
            st.info("Waiting for tick data...")
        
        st.divider()
        st.markdown("### Export")
        c1, c2 = st.columns(2)
        csv_ticks = df_ticks.to_csv(index=False).encode('utf-8')
        c1.download_button("Download Tick CSV", csv_ticks, "ticks.csv", "text/csv", key='dl_btn_tick')
        
        if not df_ana.empty:
            csv_ana = df_ana.to_csv(index=False).encode('utf-8')
            c2.download_button("Download Analytics CSV", csv_ana, "analytics.csv", "text/csv", key='dl_btn_ana')

    # --- CLIENT SIDE ALERTS (From Slider) ---
    # We check the latest Z-Score against the User Threshold
    if not df_ana.empty and 'z_score' in df_ana.columns:
        last_z = df_ana.iloc[-1]['z_score']
        # Rate Limiting (60s)
        now = pd.Timestamp.now().timestamp()
        last_toast = st.session_state.get('last_toast_time', 0)
        
        if abs(last_z) > z_score_thresh and (now - last_toast > 60):
            st.toast(f"⚠️ High Z-Score Alert: {abs(last_z):.2f} (> {z_score_thresh})", icon="🔥")
            st.session_state['last_toast_time'] = now

    # --- UPLOADED DATA VIEW ---
    up_data = st.session_state.get('uploaded_df')
    if up_data is not None:
         st.divider()
         st.subheader("Uploaded Data Analysis")
         st.dataframe(up_data.head())
         # Simple plot
         if 'close' in up_data.columns:
             st.line_chart(up_data['close'])

live_view()


# --- ANALYTICS / ADF ---
if run_adf:
    st.divider()
    st.header("Stationarity Test (ADF)")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df_hist = con.execute("SELECT symbol, close, time FROM candles_1m ORDER BY time DESC LIMIT 2000").df()
        con.close()
        
        if not df_hist.empty:
            # Fix: Filter 0 prices (bad data)
            df_hist = df_hist[df_hist['close'] > 0]
            # Fix: Deduplicate before pivoting to avoid Reshape Error
            df_hist = df_hist.drop_duplicates(subset=['time', 'symbol'])
            
            pivoted = df_hist.pivot(index='time', columns='symbol', values='close').dropna()
            prim_sym = primary_ticker.lower()
            sec_sym = secondary_ticker.lower()
            
            if prim_sym in pivoted and sec_sym in pivoted:
                import statsmodels.api as sm
                model = sm.OLS(pivoted[prim_sym], sm.add_constant(pivoted[sec_sym])).fit()
                spread = model.resid
                
                result = adfuller(spread)
                st.write(f"**ADF Statistic:** {result[0]:.4f}")
                st.write(f"**p-value:** {result[1]:.4f}")
                
                if result[1] < 0.05:
                    st.success("Spread is STATIONARY (Mean Reverting). Safe to trade.")
                else:
                    st.error("Spread is NON-STATIONARY. Random Walk danger.")
                    
                # Fix: Use Plotly for ADF Chart to handle gaps (Categorical Axis)
                spread_df = spread.reset_index()
                spread_df.columns = ['datetime', 'spread']
                # Localize and Format
                spread_df['datetime'] = spread_df['datetime'].astype('datetime64[ns]').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
                spread_df['date_str'] = spread_df['datetime'].dt.strftime('%H:%M %d-%b')
                
                fig_adf = px.line(spread_df, x='date_str', y='spread', title="Spread / Residuals (Mean Reversion)")
                fig_adf.update_xaxes(type='category', nticks=10, showgrid=True, gridcolor='rgba(42, 46, 57, 0.5)')
                fig_adf.update_yaxes(showgrid=True, gridcolor='rgba(42, 46, 57, 0.5)')
                # Add Zero Line for reference
                fig_adf.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Equilibrium (0)")
                
                fig_adf.update_layout(
                    height=350,
                    plot_bgcolor='#131722',
                    paper_bgcolor='#131722',
                    font=dict(color='#E0E0E0'),
                    xaxis_title=None,
                    yaxis_title="Spread ($)"
                )
                st.plotly_chart(fig_adf, use_container_width=True)
            else:
                st.warning("Not enough overlapping Data.")
        else:
             st.warning("No historical data.")
             
    except Exception as e:
        st.error(f"ADF Error: {e}")

# --- BACKTEST SECTION ---
if st.session_state.get('run_backtest'):
    st.markdown("### Strategy Backtest (VectorBT)")
    st.write(f"Instant Mean Reversion Test on Loaded Data ({primary_ticker})")
    try:
        df = fetch_historical_candles(primary_ticker.lower())
        if not df.empty:
            # VBT Logic
            price = df.set_index('time')['close']
            # Strategy: RSI < 30 Buy, RSI > 70 Sell
            rsi = vbt.RSI.run(price, 14)
            entries = rsi.rsi_below(30)
            exits = rsi.rsi_above(70)
            
            pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10000)
            st.metric("Total Return [%]", f"{pf.total_return()*100:.2f}%")
            st.metric("Win Rate [%]", f"{pf.stats()['Win Rate [%]']:.2f}%")
            
            st.plotly_chart(pf.plot(), use_container_width=True)
            
            if st.button("Close Backtest"):
                st.session_state['run_backtest'] = False
                st.rerun()
        else:
            st.warning("No data to backtest.")
    except Exception as e:
        st.error(f"Backtest failed: {e}")
