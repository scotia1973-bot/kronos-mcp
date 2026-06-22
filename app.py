"""
FastAPI wrapper for Kronos MCP server on VPS.
Uses standalone predictor module — no FastMCP dependency.
"""

import json
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="Kronos Market Predictor")

HTML_PAGE = """
<html>
<head><title>Kronos MCP Server</title>
<style>
    body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
    h1 { color: #58a6ff; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }
    code { background: #1f2937; padding: 2px 6px; border-radius: 4px; }
    .status { color: #3fb950; }
    a { color: #58a6ff; }
    table { width: 100%%; border-collapse: collapse; margin-top: 15px; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }
    th { color: #8b949e; }
    .up { color: #3fb950; }
    .down { color: #f85149; }
</style></head>
<body>
    <h1>🚀 Kronos Market Predictor</h1>
    <div class="card">
        <p><span class="status">●</span> <strong>Status:</strong> Running</p>
        <p><strong>Server:</strong> VPS (auto-restart enabled)</p>
        <p id="prices">Loading...</p>
    </div>
    <div class="card">
        <h3>Endpoints</h3>
        <p><code>GET /predict/btc</code> — BTC/USDT forecast</p>
        <p><code>GET /predict/eth</code> — ETH/USDT forecast</p>
        <p><code>GET /predict/{symbol}</code> — Any Binance pair</p>
        <p><code>GET /snapshot</code> — All major pairs</p>
        <p><code>GET /health</code> — Health check</p>
    </div>
    <div class="card">
        <h3>MCP Integration</h3>
        <p>Connect via Streamable HTTP:</p>
        <code>hermes config set mcp_servers.kronos.url https://YOUR-DOMAIN/mcp</code>
    </div>
    <script>
    fetch('/snapshot').then(r=>r.json()).then(d=>{
        document.getElementById('prices').innerHTML = '<table><tr><th>Pair</th><th>Price</th><th>Direction</th><th>Predicted</th><th>24h</th></tr>' +
        d.map(p => '<tr><td>' + p.symbol + '</td><td>$' + (p.current_price||'') + '</td><td class="' + (p.direction==='bullish'?'up':'down') + '">' + (p.direction==='bullish'?'📈':'📉') + ' ' + (p.direction||'') + '</td><td>$' + (p.predicted_close||'') + '</td><td>' + (p.change_24h||'') + '%</td></tr>').join('') + '</table>';
    });
    </script>
</body></html>
"""


@app.get("/")
async def root():
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kronos-mcp", "version": "1.0.0"}


@app.get("/predict/btc")
async def predict_btc():
    from predictor import forecast
    return await forecast("BTCUSDT", "24h")


@app.get("/predict/eth")
async def predict_eth():
    from predictor import forecast
    return await forecast("ETHUSDT", "24h")


@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str):
    from predictor import forecast
    pair = f"{symbol.upper()}USDT"
    if not pair.endswith("USDT"):
        pair = pair + "USDT"
    return await forecast(pair, "24h")


@app.get("/snapshot")
async def snapshot():
    from predictor import market_snapshot
    return await market_snapshot()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
