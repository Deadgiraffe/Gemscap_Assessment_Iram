from pydantic import BaseModel
from typing import Optional

class Trade(BaseModel):
    s: str  # Symbol
    p: float  # Price
    q: float  # Quantity
    T: int  # Timestamp (Transaction Time)
    m: bool # Is buyer maker

class BookTicker(BaseModel):
    s: str  # Symbol
    b: float  # Best bid price
    B: float  # Best bid qty
    a: float  # Best ask price
    A: float  # Best ask qty
    T: int    # Timestamp
    u: int    # Update ID

class Tick(BaseModel):
    type: str # 'trade' or 'bookTicker'
    data: dict # The raw data or normalized data
    timestamp: int # Event time (T)
    receipt_timestamp: float # Local time
    latency: float # receipt - T (in ms)
    symbol: str
