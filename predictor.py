"""
Standalone prediction module with Kronos AI model support.
Falls back to technical analysis when model unavailable.
"""

import pandas as pd
import numpy as np
import httpx
import asyncio
import sys
from pathlib import Path

COINGECKO = "https://api.coingecko.com/api/v3"
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "DOGEUSDT": "dogecoin", "ADAUSDT": "cardano", "XRPUSDT": "ripple",
}

# Lazy-loaded Kronos model
_kronos = None

def get_kronos():
    global _kronos
    if _kronos is not None:
        return _kronos
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from model import Kronos, KronosTokenizer, KronosPredictor
        import torch
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        model.eval()
        _kronos = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        print("Kronos model loaded (CPU)")
        return _kronos
    except Exception as e:
        print(f"Kronos not available: {e}")
        return None


async def fetch_ohlc(symbol: str = "BTCUSDT", days: int = 7) -> list:
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
    records = []
    for o in ohlc:
        records.append({
            "timestamps": pd.Timestamp(o[0], unit="ms"),
            "open": float(o[1]), "high": float(o[2]),
            "low": float(o[3]), "close": float(o[4]),
            "volume": 0, "amount": 0,
        })
    return pd.DataFrame(records)


async def forecast(symbol: str = "BTCUSDT", horizon: str = "24h") -> dict:
    ohlc = await fetch_ohlc(symbol, days=14)
    df = ohlc_to_dataframe(ohlc)
    closes = df["close"].values
    current_price = float(closes[-1])

    result = {"symbol": symbol, "current_price": round(current_price, 2)}
    
    # Try Kronos first
    predictor = get_kronos()
    if predictor:
        try:
            lookback = min(400, len(df))
            x_df = df.iloc[:lookback][["open", "high", "low", "close"]]
            x_ts = pd.Series(df.iloc[:lookback]["timestamps"])  # Series, not DatetimeIndex
                
            pred_len = 24
            last_ts = df.iloc[lookback - 1]["timestamps"]
            y_ts = pd.Series(pd.date_range(
                start=last_ts + pd.Timedelta(hours=1), periods=pred_len, freq="h"
            ))
            
            pred = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=3
            )
            predicted_close = float(pred["close"].iloc[-1])
            predicted_high = float(pred["high"].max())
            predicted_low = float(pred["low"].min())
            
            direction = "bullish" if predicted_close > current_price else "bearish"
            change = ((predicted_close - current_price) / current_price) * 100
            
            result.update({
                "direction": direction, "predicted_close": round(predicted_close, 2),
                "predicted_high": round(predicted_high, 2),
                "predicted_low": round(predicted_low, 2),
                "change_pct": round(change, 2), "horizon": horizon,
                "confidence": "high", "model": "kronos-ai",
            })
        except Exception as e:
            result.update(_technical(closes, current_price))
            result["kronos_error"] = str(e)[:80]
    else:
        result.update(_technical(closes, current_price))
        result["model"] = result.get("model", "technical_fallback")
    
    # 24h change
    change_24h = ((closes[-1] - closes[-24]) / closes[-24]) * 100 if len(closes) >= 24 else 0
    result["change_24h"] = round(change_24h, 2)
    result["timestamp"] = str(pd.Timestamp.now())
    return result


def _technical(closes: np.ndarray, current: float) -> dict:
    lookback = min(100, len(closes))
    recent = closes[-lookback:]
    sma_s = np.mean(recent[-10:])
    sma_l = np.mean(recent)
    direction = "bullish" if sma_s > sma_l else "bearish"
    reversion = (sma_l - current) / current
    predicted = current * (1 + reversion * 0.1)
    vol = float(np.std(np.diff(recent) / recent[:-1]))
    return {
        "direction": direction, "predicted_close": round(predicted, 2),
        "change_pct": round((predicted - current) / current * 100, 2),
        "horizon": "24h", "volatility": round(vol * 100, 2),
        "confidence": "low", "model": "technical_fallback",
    }


async def market_snapshot() -> list:
    results = []
    for pair in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]:
        try:
            results.append(await forecast(pair, "24h"))
        except Exception as e:
            results.append({"symbol": pair.replace("USDT", ""), "error": str(e)[:50]})
    return results
