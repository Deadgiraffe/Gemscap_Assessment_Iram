# AI Usage Transparency

## Overview
This project was developed with the assistance of Large Language Models (LLMs) to accelerate coding, debugging, and documentation. Below is a transparency report on how AI tools were utilized.

## Tools Used
*   **Antigravity (Google DeepMind)**: Used for agentic coding, file manipulation, and orchestrating the development environment.
*   **LLM (Gemini/GPT-4 class)**: Used for code generation, architectural reasoning, and error diagnosis.

## Usage Scenarios & Prompts

### 1. Architectural Design
AI was used to compare `SQLite` vs `DuckDB` and `Flask` vs `FastAPI`.
*   *Prompt logic*: "Compare SQLite and DuckDB for high-frequency time-series storage. Which handles window functions better?"
*   *Outcome*: Selected **DuckDB** for its column-oriented nature and `ASOF JOIN` capabilities.

### 2. Code Generation (Quant Models)
AI generated the boilerplate for the **Kalman Filter** and **Huber Regressor**.
*   *Prompt logic*: "Implement a SimDKalman filter in Python for dynamic hedge ratio estimation between two assets."
*   *Outcome*: Utilized the generated `SimDKalman` class, but manually tuned the `observation_noise` matrix based on testing findings.

### 3. Debugging & Error Resolution
AI was instrumental in solving the **Gradient Explosion** issue in the River library.
*   *Prompt logic*: "River LinearRegression is returning infinite weights for Bitcoin price data. How to fix?"
*   *Outcome*: AI suggested scaling the data. We decided to switch to a **Windowed Polyfit** approach for deterministic stability instead of scaling.

### 4. Frontend Optimization
AI suggested `st.fragment` (Streamlit Fragments) to solve the latency issue.
*   *Prompt logic*: "Streamlit reloads the whole page on every tick. How to update just the chart?"
*   *Outcome*: Implemented `@st.fragment(run_every=1.0)` to isolate the chart rendering loop.

## Disclaimer
While AI provided code snippets and logic, all critical financial logic (Z-Score calculation, Spread formula, Backfill logic) was verified and integrated manually by the developer. The final architecture logic is a result of human engineering decisions.
