"""
Standalone prediction module — no FastMCP dependency.
Used by the FastAPI app on the VPS.
"""

import pandas as pd
import numpy as np
import httpx
import asyncio

COINGECKO = "https://api.coingecko.com/api/v3"

# Map our symbols to CoinGecko IDs
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "DOGEUSDT": "dogecoin",
    "ADAUSDT": "cardano",
    "XRPUSDT": "ripple",
}

HORIZON_MAP = {"1h": 12, "4h": 48, "24h": 288, "7d": 2016}


async def fetch_ohlc(symbol: str = "BTCUSDT", days: int = 7) -> list:
    """Fetch OHLC data from CoinGecko (free, no key)."""
    coin_id = COINGECKO_IDS.get(symbol, "bitcoin")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{COINGECKO}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


def ohlc_to_dataframe(ohlc: list) -> pd.DataFrame:
    """Convert CoinGecko OHLC to DataFrame."""
    records = []
    for o in ohlc:
        records.append({
            "timestamps": pd.Timestamp(o[0], unit="ms"),
            "open": float(o[1]),
            "high": float(o[2]),
            "low": float(o[3]),
            "close": float(o[4]),
            "volume": 0,
            "amount": 0,
        })
    return pd.DataFrame(records)


async def fetch_current_price(symbol: str = "BTCUSDT") -> float:
    """Get current price from CoinGecko."""
    coin_id = COINGECKO_IDS.get(symbol, "bitcoin")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{COINGECKO}/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[coin_id]["usd"]


async def forecast(symbol: str = "BTCUSDT", horizon: str = "24h") -> dict:
    # Fetch OHLC data (last 7 days = ~1008 10-min candles)
    ohlc = await fetch_ohlc(symbol, days=7)
    df = ohlc_to_dataframe(ohlc)
    
    closes = df["close"].values
    current_price = float(closes[-1])
    
    # Technical analysis
    lookback = min(100, len(closes))
    recent = closes[-lookback:]
    sma_short = np.mean(recent[-10:])
    sma_long = np.mean(recent)
    direction = "bullish" if sma_short > sma_long else "bearish"
    
    returns = np.diff(recent) / recent[:-1]
    vol = float(np.std(returns))
    
    # Mean-reverting prediction
    reversion = (sma_long - current_price) / current_price
    predicted = current_price * (1 + reversion * 0.1)
    
    # 24h change (CoinGecko returns hourly candles, index -24 is ~24h ago)
    old_24h = closes[-24] if len(closes) >= 24 else closes[0]
    change_24h = ((current_price - old_24h) / old_24h) * 100
    
    return {
        "symbol": symbol,
        "direction": direction,
        "current_price": round(current_price, 2),
        "predicted_close": round(predicted, 2),
        "change_pct": round((predicted - current_price) / current_price * 100, 2),
        "change_24h": round(change_24h, 2),
        "horizon": horizon,
        "volatility": round(vol * 100, 2),
        "confidence": "low",
        "model": "technical_analysis",
        "timestamp": str(pd.Timestamp.now()),
    }


async def market_snapshot() -> list:
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    results = []
    for pair in pairs:
        try:
            r = await forecast(pair, "24h")
            results.append(r)
        except Exception as e:
            results.append({"symbol": pair.replace("USDT", ""), "error": str(e)[:50]})
    return results
