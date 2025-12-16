import asyncio
import json
import os
import time
import redis.asyncio as redis
import websockets
from fastapi import FastAPI
from pydantic import ValidationError
from loguru import logger

from models import Trade, BookTicker, Tick

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
BINANCE_WS_URL = "wss://fstream.binance.com/ws/btcusdt@trade/ethusdt@trade/btcusdt@bookTicker/ethusdt@bookTicker"

# Tuning Parameters
BATCH_SIZE = 200     # Flush after N messages
FLUSH_INTERVAL = 0.05 # Flush every N seconds (50ms)

app = FastAPI()
redis_client: redis.Redis = None
ws_task = None

# Global Buffer state
write_buffer = []
last_flush_time = 0.0
msg_count = 0

@app.on_event("startup")
async def startup_event():
    global redis_client, ws_task
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    ws_task = asyncio.create_task(binance_ws_listener())

@app.on_event("shutdown")
async def shutdown_event():
    if ws_task:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            logger.info("WS Task cancelled successfully")
            
    if redis_client:
        # Final flush
        if write_buffer:
            await flush_buffer()
        await redis_client.close()
        logger.info("Redis connection closed")

async def binance_ws_listener():
    """Connects to Binance WS and pushes data to Redis via Batch Buffer."""
    global last_flush_time
    last_flush_time = time.time()
    
    while True:
        try:
            async with websockets.connect(BINANCE_WS_URL) as ws:
                logger.info(f"Connected to Binance WS: {BINANCE_WS_URL}")
                while True:
                    # 1. Receive Message (Await)
                    # We might want a timeout here to ensure we flush even if no data comes ?
                    # But for BTC/ETH, data is constant.
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Idle heartbeat, check flush
                        await check_flush()
                        continue
                        
                    # Capture Receipt Time immediately
                    receipt_ts = time.time() * 1000 # ms
                    
                    data = json.loads(msg)
                    
                    # 2. Process & Buffer
                    await process_and_buffer(data, receipt_ts)
                    
                    # 3. Check Flush Contraints
                    await check_flush()

        except Exception as e:
            logger.error(f"WS Connection error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

async def check_flush():
    global last_flush_time
    now = time.time()
    if (len(write_buffer) >= BATCH_SIZE) or (now - last_flush_time >= FLUSH_INTERVAL and len(write_buffer) > 0):
        await flush_buffer()
        last_flush_time = now

async def flush_buffer():
    """Executes buffered commands in a Redis Pipeline."""
    if not redis_client: return
    
    pipeline = redis_client.pipeline()
    count = len(write_buffer)
    
    for item in write_buffer:
        # 1. XADD
        pipeline.xadd(
            "stream:ticks",
            {"json": item},
            maxlen=100000,
            approximate=True
        )
        # 2. Publish (Optional: Could reduce this volume if UI lags, but required for realtime)
        pipeline.publish("channel:updates", item)
        
    try:
        await pipeline.execute()
        # logger.debug(f"Flushed {count} ticks.")
    except Exception as e:
        logger.error(f"Redis Pipeline Error: {e}")
    finally:
        write_buffer.clear()

async def process_and_buffer(data: dict, receipt_ts: float):
    global msg_count
    """Normalizes message and adds to buffer."""
    try:
        event_type = data.get('e')
        # msg_count += 1
        # if msg_count % 1000 == 0:
        #     logger.info(f"Ingested {msg_count} messages. Last event: {event_type}")
        
        normalized_tick = None
        
        if event_type == 'trade':
            trade = Trade(**data)
            # Latency: receipt - T
            latency = receipt_ts - trade.T
            
            normalized_tick = Tick(
                type='trade',
                data=trade.dict(),
                timestamp=trade.T,
                receipt_timestamp=receipt_ts,
                latency=latency,
                symbol=trade.s
            )
        elif event_type == 'bookTicker':
            ticker = BookTicker(**data)
            latency = receipt_ts - ticker.T
            
            normalized_tick = Tick(
                type='bookTicker',
                data=ticker.dict(),
                timestamp=ticker.T,
                receipt_timestamp=receipt_ts,
                latency=latency,
                symbol=ticker.s
            )
            
        if normalized_tick:
            write_buffer.append(normalized_tick.json())

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Processing error: {e}")
