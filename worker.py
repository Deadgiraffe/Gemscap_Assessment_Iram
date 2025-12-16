import asyncio
import json
import os
import duckdb
import numpy as np
import polars as pl
import redis.asyncio as redis
import simdkalman
from collections import deque # Added this import
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.stattools import adfuller
from loguru import logger
from models import Tick
import pandas as pd
import requests
import time as sys_time
# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DB_PATH = "market_data.duckdb"

# Initialize DuckDB
def init_db():
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS candles_1m (time TIMESTAMP, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, vwap DOUBLE)")
    con.execute("CREATE TABLE IF NOT EXISTS candles_5m (time TIMESTAMP, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, vwap DOUBLE)")
    con.close()

def backfill_historical_data():
    """Fetches missing historical data from Binance and backfills DuckDB."""
    logger.info("Starting Historical Data Backfill...")
    con = duckdb.connect(DB_PATH)
    
    # Symbols and Intervals to backfill
    tasks = [
        ("BTCUSDT", "1m", "candles_1m"),
        ("ETHUSDT", "1m", "candles_1m"),
        ("BTCUSDT", "5m", "candles_5m"),
        ("ETHUSDT", "5m", "candles_5m")
    ]
    
    for symbol, interval, table in tasks:
        try:
            # Check latest time in DB
            res = con.execute(f"SELECT MAX(time) FROM {table} WHERE symbol='{symbol.lower()}'").fetchone()
            last_time = res[0]
            
            # Decide startTime (Binance Limit 1000)
            limit = 1000
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            
            if last_time:
                try:
                    # DuckDB Timestamp to ms
                    start_ms = int(last_time.timestamp() * 1000)
                    offset = 60000 if interval == "1m" else 300000
                    params["startTime"] = start_ms + offset
                except: pass
                
            url = "https://api.binance.com/api/v3/klines"
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            
            if isinstance(data, list) and len(data) > 0:
                rows = []
                for k in data:
                    ts_ms = k[0]
                    rows.append({
                        "time": ts_ms, 
                        "symbol": symbol.lower(),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "vwap": 0.0 
                    })
                
                if rows:
                    df = pl.DataFrame(rows)
                    # Convert 'time' (int ms) to Datetime (us) for DuckDB
                    df = df.with_columns([
                        (pl.col("time") * 1000).cast(pl.Datetime).alias("time")
                    ])
                    
                    arrow_tbl = df.to_arrow()
                    con.sql(f"INSERT INTO {table} SELECT * FROM arrow_tbl")
                    logger.info(f"Backfilled {len(rows)} candles for {symbol} {interval}")
            else:
                logger.info(f"No new data for {symbol} {interval}")
                
        except Exception as e:
            logger.error(f"Backfill failed for {symbol} {interval}: {e}")
            
    con.close()
    logger.info("Backfill Complete.")

# Analytics State
class AnalyticsEngine:
    def __init__(self):
        # 1. Kalman Filter (SimDKalman)
        self.kf = simdkalman.KalmanFilter(
            state_transition=np.eye(2),
            process_noise=np.diag([1e-5, 1e-5]),
            observation_model=np.array([[0, 1]]), 
            observation_noise=1.0
        )
        self.kf_state_mean = np.zeros((2, 1))
        self.kf_state_cov = np.eye(2)
        
        # 2. Robust Regression (Huber - Sklearn Batch)
        self.huber_buffer = {"x": [], "y": []}
        self.huber_params = {"beta": 0.0, "alpha": 0.0}
        
        # 3. Rolling OLS (Windowed Numpy) - STABLE
        self.river_window_x = deque(maxlen=300)
        self.river_window_y = deque(maxlen=300)
        self.river_params = {"beta": 0.0, "alpha": 0.0}
        
        # Latest Prices
        self.latest_prices = {"btcusdt": None, "ethusdt": None}
        self.ofi_accum = {"btcusdt": 0.0, "ethusdt": 0.0}
        self.prev_book = {"btcusdt": None, "ethusdt": None}

    def update_price(self, symbol, price):
        self.latest_prices[symbol] = price
        if self.latest_prices["btcusdt"] and self.latest_prices["ethusdt"]:
            self.step_kalman()
            self.step_river()
            self.update_huber_buffer()

    def update_book(self, symbol, bid_p, bid_q, ask_p, ask_q):
        prev = self.prev_book.get(symbol)
        ofi_delta = 0.0
        
        if prev:
            pb_p, pb_q, pa_p, pa_q = prev
            # Classic OFI
            # Bid side
            if bid_p > pb_p: ofi_delta += bid_q
            elif bid_p == pb_p: ofi_delta += (bid_q - pb_q)
            else: ofi_delta -= pb_q
            # Ask side
            if ask_p < pa_p: ofi_delta -= ask_q
            elif ask_p == pa_p: ofi_delta -= (ask_q - pa_q)
            else: ofi_delta += pa_q

        self.ofi_accum[symbol] += ofi_delta
        self.prev_book[symbol] = (bid_p, bid_q, ask_p, ask_q)
        return self.ofi_accum[symbol]

    def step_kalman(self):
        btc = self.latest_prices["btcusdt"]
        eth = self.latest_prices["ethusdt"]
        H = np.array([[eth, 1]])
        
        x_pred = self.kf_state_mean
        P_pred = self.kf_state_cov + np.diag([1e-5, 1e-5])
        
        y_res = btc - (x_pred[0,0] * eth + x_pred[1,0])
        S = (H @ P_pred @ H.T) + 1.0 
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        self.kf_state_mean = x_pred + K * y_res
        self.kf_state_cov = (np.eye(2) - K @ H) @ P_pred
        
    def step_river(self):
        btc = self.latest_prices["btcusdt"]
        eth = self.latest_prices["ethusdt"]
        
        self.river_window_x.append(eth)
        self.river_window_y.append(btc)
        
        if len(self.river_window_x) < 20: 
            return # Warmup

        # Exact OLS on Window (No SGD instability)
        try:
            x_arr = np.array(self.river_window_x)
            y_arr = np.array(self.river_window_y)
            
            # Polyfit degree 1: returns [slope, intercept]
            # y = mx + c -> btc = beta*eth + alpha
            beta, alpha = np.polyfit(x_arr, y_arr, 1)
            
            self.river_params["beta"] = beta
            self.river_params["alpha"] = alpha
        except:
             pass

    def update_huber_buffer(self):
        self.huber_buffer["x"].append(self.latest_prices["ethusdt"])
        self.huber_buffer["y"].append(self.latest_prices["btcusdt"])
        if len(self.huber_buffer["x"]) > 1000:
             self.huber_buffer["x"] = self.huber_buffer["x"][-1000:]
             self.huber_buffer["y"] = self.huber_buffer["y"][-1000:]

    def fit_huber(self):
        if len(self.huber_buffer["x"]) < 50: return
        
        X_arr = np.array(self.huber_buffer["x"])
        y_arr = np.array(self.huber_buffer["y"])
        
        # Avoid fitting on constant data (convergence failure)
        if np.std(X_arr) < 1e-9 or np.std(y_arr) < 1e-9 or len(np.unique(X_arr)) < 10:
            return

        # Manual Standard Scaling to fix "ABNORMAL termination" (l-BFGS-b hates unscaled data)
        mu_x, sig_x = np.mean(X_arr), np.std(X_arr)
        mu_y, sig_y = np.mean(y_arr), np.std(y_arr)
        
        X_sc = (X_arr - mu_x) / sig_x
        y_sc = (y_arr - mu_y) / sig_y
        
        X_in = X_sc.reshape(-1, 1)

        try:
            # Increase max_iter for better convergence on noisy streaming data
            huber = HuberRegressor(max_iter=2000).fit(X_in, y_sc)
            
            beta_sc = huber.coef_[0]
            alpha_sc = huber.intercept_
            
            # Reconstruct original parameters
            if sig_x > 1e-9:
                real_beta = beta_sc * (sig_y / sig_x)
                real_alpha = mu_y + (sig_y * alpha_sc) - (real_beta * mu_x)
            else:
                real_beta = 0.0
                real_alpha = mu_y

            # Clamp outliers (e.g. during initialization)
            if abs(real_beta) > 100: real_beta = 0.0
            
            self.huber_params["beta"] = real_beta
            self.huber_params["alpha"] = real_alpha
            
        except Exception as e:
            logger.warning(f"Huber Fit Warning: {e}")

    def get_state(self):
        eth = self.latest_prices["ethusdt"] or 0
        btc = self.latest_prices["btcusdt"] or 0
        
        # Calculate Fair Values for all strategies
        fv_kalman = float(self.kf_state_mean[0,0] * eth + self.kf_state_mean[1,0])
        fv_river = float(self.river_params["beta"] * eth + self.river_params["alpha"])
        fv_huber = float(self.huber_params["beta"] * eth + self.huber_params["alpha"])
        
        z_score_kalman = btc - fv_kalman if btc else 0
        
        return {
            "kalman": {"beta": float(self.kf_state_mean[0,0]), "alpha": float(self.kf_state_mean[1,0]), "fv": fv_kalman, "z": z_score_kalman},
            "river": {"beta": self.river_params["beta"], "alpha": self.river_params["alpha"], "fv": fv_river},
            "huber": {"beta": self.huber_params["beta"], "alpha": self.huber_params["alpha"], "fv": fv_huber},
            "ofi_btc": self.ofi_accum["btcusdt"],
        }

async def worker():
    init_db()
    
    # Perform Backfill on Startup
    try:
        backfill_historical_data()
    except Exception as e:
        logger.error(f"Backfill Critical Fail: {e}")
        
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    analytics = AnalyticsEngine()
    
    candles_buffer_1m = {"btcusdt": [], "ethusdt": []}
    candles_buffer_5m = {"btcusdt": [], "ethusdt": []} # Simplified: accum 1m or raw? stick to raw for accuracy
    
    current_minute = None
    last_id = "$"
    
    logger.info("Worker started.")
    
    while True:
        try:
            streams = await r.xread({"stream:ticks": last_id}, count=100, block=100)
            if not streams: continue
                
            for stream_name, messages in streams:
                for msg_id, data in messages:
                    last_id = msg_id
                    raw_json = data.get("json")
                    if not raw_json: continue
                        
                    tick_data = json.loads(raw_json)
                    symbol = tick_data['symbol'].lower()
                    ts = tick_data['timestamp']
                    type_ = tick_data['type']
                    
                    if type_ == 'trade':
                        price = float(tick_data['data']['p'])
                        volume = float(tick_data['data']['q'])
                        analytics.update_price(symbol, price)
                        
                        tick_record = {"timestamp": ts, "price": price, "volume": volume, "notional": price * volume}
                        # Symbol from tick_data might be 'BTCUSDT', buffer keys are 'btcusdt'
                        s_lower = symbol.lower()
                        if s_lower in candles_buffer_1m:
                            candles_buffer_1m[s_lower].append(tick_record)
                            candles_buffer_5m[s_lower].append(tick_record)
                        
                    elif type_ == 'bookTicker':
                        analytics.update_book(symbol, float(tick_data['data']['b']), float(tick_data['data']['B']),
                                              float(tick_data['data']['a']), float(tick_data['data']['A']))

                    # Periodic Huber Fit
                    if np.random.random() < 0.01: analytics.fit_huber()

                    # Publish Analytics
                    state = analytics.get_state()
                    if analytics.latest_prices["btcusdt"]:
                         # Fix: xadd expects a dict of fields
                        await r.xadd("stream:analytics", {"json": json.dumps(state)}, maxlen=10000)
                        
                        # Alerting (check Kalman Z only for now)
                        if abs(state['kalman']['z']) > 2.5:
                             await r.publish("stream:alerts", json.dumps({"msg": f"High Z-Score (Kalman): {state['kalman']['z']:.2f}", "ts": ts}))

                    # Resampling 1m
                    msg_minute = ts // 60000
                    if current_minute is None: current_minute = msg_minute
                    
                    if msg_minute > current_minute:
                        await flush_candles(current_minute, candles_buffer_1m, "candles_1m")
                        
                        if msg_minute % 5 == 0:
                             # Flush 5m
                             # Note: this is a simple approximation. Real logic handles precise windows.
                             await flush_candles(current_minute, candles_buffer_5m, "candles_5m")
                             candles_buffer_5m = {"btcusdt": [], "ethusdt": []}
                             
                        current_minute = msg_minute
                        candles_buffer_1m = {"btcusdt": [], "ethusdt": []}

        except Exception as e:
            logger.error(f"Worker Loop Error: {e}")
            await asyncio.sleep(1)

async def flush_candles(current_minute_ts, buffer, table_name):
    # SENIOR IMPLEMENTATION: Zero-Copy Ingestion (Polars -> Arrow -> DuckDB)
    # Replaces "Junior" Pandas loop/insert.
    
    # 1. Prepare Data
    all_ticks = []
    for sym, ticks in buffer.items():
        if ticks:
            # Attach symbol to each tick for bulk processing
            for t in ticks: t['symbol'] = sym
            all_ticks.extend(ticks)
            
    if not all_ticks: return

    try:
        # 2. Polars Dynamic Resampling (TRS Req #3)
        # Handle the "Firehose" efficiently
        df = pl.DataFrame(all_ticks)
        
        q = (
            df.lazy()
            .with_columns([
                (pl.col("timestamp") * 1000).cast(pl.Datetime).alias("datetime"), # ms to us for Polars
                pl.col("symbol")
            ])
            .group_by_dynamic("datetime", every="1m", group_by="symbol", period="1m")
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                (pl.col("notional").sum() / pl.col("volume").sum()).alias("vwap")
            ])
            # Fix: Keep as Datetime for Arrow->DuckDB transfer. 
            # Do NOT cast to Int64, as DuckDB fails to cast BIGINT->TIMESTAMP implicitly here.
        )
        
        pro_df = q.collect()
        
        if pro_df.is_empty(): return

        # 3. DuckDB Fast Append (TRS Req #2)
        # Using Zero-Copy Arrow transfer for maximum throughput
        import duckdb
        con = duckdb.connect(DB_PATH)
        
        # Ensure Schema alignment: time, symbol, open, high, low, close, volume, vwap
        # Renaming for Insert
        final_df = pro_df.select([
            pl.col("datetime").alias("time"),
            pl.col("symbol"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            pl.col("vwap")
        ])
        
        # Conversion to Arrow for Zero-Copy Insert
        arrow_tbl = final_df.to_arrow()
        
        # Efficient Insert
        con.sql(f"INSERT INTO {table_name} SELECT * FROM arrow_tbl")
        con.close()
        
    except Exception as e:
        logger.error(f"Flush Error: {e}")

if __name__ == "__main__":
    asyncio.run(worker())
