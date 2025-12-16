# Real-Time Quantitative Intelligence System (RT-QIS)

## Executive Summary
RT-QIS is a high-frequency trading dashboard designed to ingest, process, and visualize cryptocurrency market data (BTC/USDT, ETH/USDT) in real-time. It enables traders to monitor microstructure events, visualize statistical arbitrage signals (Cointegration/Z-Score), and execute data-driven decisions with sub-second latency (`< 500ms` analytical processing).

This project was built to address the challenge of **"Modular Reactivity"**: decoupling the high-velocity "Producer" (Exchange Feeds) from the "Consumer" (UI/Charts) to prevent blocking and ensure smooth, 60 FPS visualization.

---

## 🚀 Setup & Execution

### Prerequisites
- **Docker & Docker Compose** (Required)
- Python 3.10+ (For local development only)

### Quick Start
The entire stack is containerized. To launch the system:

```bash
# 1. Clone the repository
git clone <repo_url>
cd <repo_name>

# 2. Start the Application (Backend + Frontend + DB + Redis)
docker-compose up --build
```

Access the Dashboard at: **[http://localhost:8501](http://localhost:8501)**

---

## 🏗️ Methodology & Architecture

The system follows a **Microservices Loop Architecture**:

1.  **Ingestion Layer (`ingest.py`)**:
    *   Connects to Binance WebSocket (`wss://stream.binance.com`).
    *   Subscribes to `@trade` and `@bookTicker`.
    *   Normalizes raw JSON into strict Pydantic models.
    *   Publishes ticks to **Redis Streams** (`stream:ticks`).

2.  **Processing & Storage Layer (`worker.py`)**:
    *   Consumes the Redis Stream.
    *   **Hot Storage**: Adds ticks to a Rolling Window (`deque`) for instantaneous OLS/Kalman calculations.
    *   **Cold Storage**: Aggregates ticks into 1-minute and 5-minute candles using **Polars** (High-Performance DataFrame) and persists them to **DuckDB** (`market_data.duckdb`) for historical analysis.
    *   **Analytics**: Runs **SimDKalman** (Dynamic Hedge Ratio), **Huber Regressor** (Robust "Fat-Tail" Fitting), and **ADF Tests** on the stream.
    *   Publishes computed signals (Z-Score, Fair Value) to `stream:analytics`.

3.  **Visualization Layer (`dashboard.py`)**:
    *   Built with **Streamlit** + `st.fragment` for partial re-rendering (solving the classic Streamlit "whole page reload" lag).
    *   Fetches "Live Snapshot" from Redis and "History" from DuckDB.
    *   Merges them into a seamless standard dataframe.
    *   Renders interactive **Plotly** charts (Candlesticks, Z-Score, Market Depth).

---

## 🧠 Analytics & Quantitative Models

### 1. Kalman Filter (Dynamic Hedge Ratio)
Instead of a static "Moving Average", we use a **Kalman Filter** to estimate the "True" Hedge Ratio ($\beta$) between BTC and ETH.
*   *Why?* Crypto correlations are non-stationary. A static model lags behind market shifts. The Kalman Filter adapts instantly to regime changes.
*   *Implementation*: `SimDKalman` library for vectorized state transitions.

### 2. Rolling OLS (Windowed)
A classic statistical arbitrage model.
*   *Logic*: Calculates linear regression ($y = \beta x + \alpha$) over a sliding window (e.g., last 300 ticks).
*   *Improvement*: Replaced unstable Online SGD (River) with a deterministic **Numpy Polyfit** on a `deque` buffer to prevent gradient explosion.

### 3. Robust Regression (Huber Loss)
*   *Problem*: Crypto data has massive outliers (flash crashes). Standard OLS squared error is sensitive to these.
*   *Solution*: **Huber Loss** penalizes outliers linearly rather than quadratically, creating a "Fair Value" line that ignores noise/wicks.

### 4. Stationarity Test (ADF)
*   *Feature*: On-demand **Augmented Dickey-Fuller (ADF)** test.
*   *Utility*: Tells the trader if the spread is "Mean Reverting" (Safe to Grid Trade) or "Random Walk" (Dangerous/Trending).

---

## 🛠️ Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Ingestion** | **FastAPI** | AsyncIO concurrency for handling 1000+ msgs/sec without blocking. |
| **Message Broker** | **Redis** | In-memory Pub/Sub for sub-millisecond decoupling of Backend/Frontend. |
| **Database** | **DuckDB** | OLAP database. Super-fast for aggregating millions of ticks into candles. |
| **Frontend** | **Streamlit** | Rapid prototyping. Augmented with `st.fragment` for high performance. |
| **DataFrames** | **Polars** | Rust-based processing. 10x-50x faster than Pandas for resampling. |

---

## 🐛 Challenges & Solutions (Technical Survey)

During development, we encountered and solved several critical HFT engineering problems:

1.  **Issue: "Sparse" Graphs (Visual Gaps)**
    *   *Problem*: When the system is restarted, the chart showed empty space for the offline time, making it look broken.
    *   *Solution*: Implemented **Automatic Historical Backfill**. On startup, the worker queries Binance API for missing candles and fills the DuckDB gaps instantly.

2.  **Issue: UI Freezing / "Translucent" Overlay**
    *   *Problem*: Processing 50,000 tick/sec on the frontend CPU thread caused Streamlit to hang.
    *   *Solution*: Optimized Redis fetch buffer to **10,000 items**. This proved to be the "Sweet Spot" between data density and UI responsiveness.

3.  **Issue: Gradient Explosion in Online Learning**
    *   *Problem*: The `river` library's SGD optimizer was unstable with unscaled crypto prices ($90,000 BTC), causing the Hedge Ratio to go to Infinity.
    *   *Solution*: Switched to a **Windowed OLS** approach using `collections.deque` and `numpy.polyfit`. It provides the same "streaming" benefit but with mathematical stability.

4.  **Issue: Time Alignment**
    *   *Problem*: Mixing "Live" Redis data (UTC) with "Historical" DB data (Local Time) caused candles to jump around.
    *   *Solution*: Standardized the entire pipeline to **Unix Timestamp (Seconds)** internally, and only converted to **IST (Indian Standard Time)** at the very last step in the Visualization layer.

---

## ✅ Deliverables Checklist

- [x] **Real-Time Dashboard**: Working `dashboard.py`.
- [x] **Analytics Engine**: `worker.py` (Kalman, Huber, OLS).
- [x] **Alerts**: UI-Toast Alerts based on Z-Score Thresholds.
- [x] **Data Management**: Upload OHLC CSV & Download Analytics CSV.
- [x] **Architecture**: Decoupled Producer-Consumer via Redis.

