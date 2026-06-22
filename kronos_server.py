"""
Kronos MCP Server — Autonomous financial prediction for AI agents.

Exposes Kronos time-series foundation model predictions as MCP tools.
Deploy anywhere (HuggingFace Spaces, Modal, VPS) — zero maintenance.
"""

from fastmcp import FastMCP
import pandas as pd
import numpy as np
import httpx
import json
import os
import time
from typing import Optional

# ── MCP Server ────────────────────────────────────────────
mcp = FastMCP(
    "kronos-mcp",
    instructions="""Financial market prediction MCP server powered by Kronos 
    (AAAI 2026 foundation model for financial time series). 
    
    Provides cryptocurrency price forecasting using real-time Binance data.
    
    Tools:
    - forecast_btc: Predict BTC/USDT price movement (4h, 24h, 7d horizons)
    - forecast_eth: Predict ETH/USDT price movement
    - forecast_crypto: Predict any Binance trading pair
    - market_snapshot: Get current market conditions + Kronos sentiment
    """,
)

# ── Global model cache ────────────────────────────────────
_model = None
_tokenizer = None
_predictor = None

def get_kronos():
    """Lazy-load Kronos model (cached after first call)."""
    global _model, _tokenizer, _predictor
    
    if _predictor is not None:
        return _predictor
    
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        
        model_name = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
        tokenizer_name = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
        
        print(f"Loading Kronos model: {model_name}")
        
        _tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        _model = Kronos.from_pretrained(model_name)
        
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _predictor = KronosPredictor(_model, _tokenizer, device=device, max_context=512)
        
        print(f"Kronos loaded on {device}")
        return _predictor
    except ImportError:
        print("Kronos model not available locally. Run with model files.")
        return None

# ── Market Data ────────────────────────────────────────────

BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINE = f"{BINANCE_BASE}/api/v3/klines"

INTERVALS = {
    "4h": "4h",
    "1d": "1d",
    "1h": "1h",
    "15m": "15m",
    "5m": "5m",
}

HORIZON_MAP = {
    "1h": 12,    # 12 x 5min = 1h
    "4h": 48,    # 48 x 5min = 4h
    "24h": 288,  # 288 x 5min = 24h
    "7d": 2016,  # 2016 x 5min = 7d (but limited by context)
}


async def fetch_klines(symbol: str = "BTCUSDT", limit: int = 500) -> list:
    """Fetch OHLCV data from Binance."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            BINANCE_KLINE,
            params={"symbol": symbol, "interval": "5m", "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """Convert Binance klines to Kronos-compatible DataFrame."""
    records = []
    for k in klines:
        records.append({
            "timestamps": pd.Timestamp(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "amount": float(k[7]),  # Quote asset volume
        })
    return pd.DataFrame(records)


# ── MCP Tools ──────────────────────────────────────────────

@mcp.tool()
async def forecast_btc(horizon: str = "24h") -> dict:
    """Forecast BTC/USDT price for the given horizon.
    
    Args:
        horizon: Prediction horizon - '1h', '4h', '24h', or '7d'
    
    Returns:
        Current price, predicted price, direction, and confidence metrics.
    """
    return await forecast_crypto("BTCUSDT", horizon)


@mcp.tool()
async def forecast_eth(horizon: str = "24h") -> dict:
    """Forecast ETH/USDT price for the given horizon.
    
    Args:
        horizon: Prediction horizon - '1h', '4h', '24h', or '7d'
    
    Returns:
        Current price, predicted price, direction, and confidence metrics.
    """
    return await forecast_crypto("ETHUSDT", horizon)


@mcp.tool()
async def forecast_crypto(symbol: str = "BTCUSDT", horizon: str = "24h") -> dict:
    """Forecast any Binance trading pair using Kronos.
    
    Args:
        symbol: Binance trading pair (e.g. 'BTCUSDT', 'ETHUSDT', 'SOLUSDT')
        horizon: Prediction horizon - '1h', '4h', '24h', or '7d'
    
    Returns:
        Structured forecast with current price, predicted direction, 
        predicted close, and market context.
    """
    # Fetch live data
    klines = await fetch_klines(symbol, limit=500)
    df = klines_to_dataframe(klines)
    
    current_price = float(df["close"].iloc[-1])
    
    # Try Kronos prediction
    pred_result = None
    try:
        pred_len = HORIZON_MAP.get(horizon, 288)
        lookback = min(400, len(df) - 1)
        
        x_df = df.iloc[:lookback][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.iloc[:lookback]["timestamps"]
        
        # Predict end timestamp
        last_ts = df.iloc[lookback - 1]["timestamps"]
        y_ts = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=5),
            periods=pred_len,
            freq="5min"
        )
        
        predictor = get_kronos()
        if predictor:
            try:
                pred = predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=pred_len,
                    T=1.0,
                    top_p=0.9,
                    sample_count=3,
                )
                predicted_close = float(pred["close"].iloc[-1])
                predicted_high = float(pred["high"].max())
                predicted_low = float(pred["low"].min())
                
                direction = "bullish" if predicted_close > current_price else "bearish"
                change_pct = ((predicted_close - current_price) / current_price) * 100
                
                pred_result = {
                    "direction": direction,
                    "current_price": round(current_price, 2),
                    "predicted_close": round(predicted_close, 2),
                    "predicted_high": round(predicted_high, 2),
                    "predicted_low": round(predicted_low, 2),
                    "change_pct": round(change_pct, 2),
                    "horizon": horizon,
                    "confidence": "medium",
                    "model": "kronos"
                }
            except Exception as kronos_err:
                pred_result = _technical_fallback(df, current_price, horizon)
                pred_result["kronos_error"] = str(kronos_err)[:100]
        else:
            pred_result = _technical_fallback(df, current_price, horizon)
            pred_result["note"] = "Kronos model not loaded — using technical analysis"
    except Exception as e:
        pred_result = _technical_fallback(df, current_price, horizon)
        pred_result["error"] = str(e)[:100]
    
    # Add market context
    pred_result["symbol"] = symbol
    pred_result["timestamp"] = str(pd.Timestamp.now())
    pred_result["recent_volatility"] = _calc_volatility(df)
    pred_result["24h_change"] = _calc_24h_change(df)
    
    return pred_result


@mcp.tool()
async def market_snapshot() -> list:
    """Get a quick snapshot of major crypto markets with Kronos sentiment.
    
    Returns forecast for BTC, ETH, SOL, and one altcoin.
    """
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    results = []
    
    for pair in pairs:
        try:
            result = await forecast_crypto(pair, "24h")
            results.append({
                "symbol": pair.replace("USDT", ""),
                "price": result["current_price"],
                "direction": result["direction"],
                "change_24h": result.get("24h_change", 0),
                "predicted_change": result.get("change_pct", 0)
            })
        except Exception as e:
            results.append({
                "symbol": pair.replace("USDT", ""),
                "error": str(e)[:50]
            })
    
    return results


# ── Fallback Analysis ──────────────────────────────────────

def _technical_fallback(df: pd.DataFrame, current_price: float, horizon: str) -> dict:
    """Simple technical analysis fallback when Kronos model isn't loaded."""
    closes = df["close"].values
    lookback = min(50, len(closes))
    recent = closes[-lookback:]
    
    # Simple moving average trend
    sma_short = np.mean(recent[-10:])
    sma_long = np.mean(recent)
    trend = "bullish" if sma_short > sma_long else "bearish"
    
    # Volatility estimate
    returns = np.diff(recent) / recent[:-1]
    vol = float(np.std(returns))
    
    # Simple prediction (mean-reverting)
    predicted = current_price * (1 + (sma_long - current_price) / current_price * 0.1)
    
    return {
        "direction": trend,
        "current_price": round(current_price, 2),
        "predicted_close": round(predicted, 2),
        "change_pct": round((predicted - current_price) / current_price * 100, 2),
        "horizon": horizon,
        "confidence": "low",
        "method": "technical_fallback"
    }


def _calc_volatility(df: pd.DataFrame) -> float:
    """Calculate recent volatility (standard deviation of returns)."""
    closes = df["close"].values[-100:]
    if len(closes) < 10:
        return 0
    returns = np.diff(closes) / closes[:-1]
    return round(float(np.std(returns)) * 100, 2)


def _calc_24h_change(df: pd.DataFrame) -> float:
    """Calculate 24h price change percentage."""
    if len(df) < 288:
        return 0
    old = df["close"].iloc[-288]
    new = df["close"].iloc[-1]
    return round(((new - old) / old) * 100, 2)


# ── Run ────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
