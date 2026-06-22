"""
FastAPI wrapper for Kronos MCP server on VPS.
Exposes predictions as MCP tools via FastMCP and REST endpoints.
"""

import json, os, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="Kronos Market Predictor")

HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Kronos AI Market Predictor</title>
<style>
    body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
    h1 { color: #58a6ff; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }
    code { background: #1f2937; padding: 2px 6px; border-radius: 4px; }
    .up { color: #3fb950; } .down { color: #f85149; }
    table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }
    th { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge-ai { background: #1f6feb33; color: #58a6ff; }
    .badge-ta { background: #30363d; color: #8b949e; }
</style></head>
<body>
    <h1>🚀 Kronos AI Market Predictor</h1>
    <div class="card">
        <p><span class="up">●</span> <strong>Kronos-small</strong> (24.7M params) running on CPU — automatic daily updates</p>
        <div id="prices">Loading market data...</div>
    </div>
    <div class="card">
        <h3>📡 MCP Endpoint</h3>
        <p>Connect via Streamable HTTP:</p>
        <code>hermes config set mcp_servers.kronos.url http://172.86.117.39:8080/mcp</code>
    </div>
    <div class="card">
        <h3>🔌 REST Endpoints</h3>
        <table>
            <tr><td><code>GET /predict/btc</code></td><td>BTC/USDT AI forecast</td></tr>
            <tr><td><code>GET /predict/eth</code></td><td>ETH/USDT AI forecast</td></tr>
            <tr><td><code>GET /predict/{symbol}</code></td><td>Any token (e.g. SOL, DOGE)</td></tr>
            <tr><td><code>GET /snapshot</code></td><td>All major pairs at once</td></tr>
            <tr><td><code>GET /health</code></td><td>Health check</td></tr>
        </table>
    </div>
    <div class="card">
        <h3>🤖 Cron Jobs</h3>
        <table>
            <tr><td>Daily Forecast</td><td>Every 8am</td><td>Full market briefing</td></tr>
            <tr><td>Market Pulse</td><td>8am, 2pm, 8pm</td><td>Quick snapshot</td></tr>
        </table>
    </div>
    <script>
    fetch('/snapshot').then(r=>r.json()).then(d=>{
        document.getElementById('prices').innerHTML = '<table><tr><th>Asset</th><th>Price</th><th>Direction</th><th>Target</th><th>Range</th><th>Model</th><th>24h</th></tr>' +
        d.map(p => '<tr><td><strong>' + (p.symbol||'').replace('USDT','') + '</strong></td>' +
            '<td>$' + (p.current_price||'') + '</td>' +
            '<td class="' + (p.direction==='bullish'?'up':'down') + '">' + (p.direction==='bullish'?'📈':'📉') + '</td>' +
            '<td>$' + (p.predicted_close||'') + '</td>' +
            '<td>$' + (p.predicted_low||'') + ' - $' + (p.predicted_high||'') + '</td>' +
            '<td><span class="badge ' + (p.model==='kronos-ai'?'badge-ai':'badge-ta') + '">' + (p.model==='kronos-ai'?'AI':'tech') + '</span></td>' +
            '<td>' + (p.change_24h||'') + '%</td></tr>').join('') + '</table>';
    });
    </script>
</body></html>"""


@app.get("/")
async def root():
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kronos-mcp", "model": "kronos-small-24.7M", "version": "1.0.0"}


from predictor import forecast, market_snapshot as snap

@app.get("/predict/btc")
async def predict_btc():
    return await forecast("BTCUSDT", "24h")

@app.get("/predict/eth")
async def predict_eth():
    return await forecast("ETHUSDT", "24h")

@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str):
    pair = f"{symbol.upper()}USDT"
    if not pair.endswith("USDT"):
        pair = pair + "USDT"
    return await forecast(pair, "24h")

@app.get("/snapshot")
async def snapshot():
    return await snap()


# ── MCP Tool Definitions ─────────────────────────────────

# FastMCP tools registered inline for agent use
# Agents can connect via: hermes config set mcp_servers.kronos.url http://172.86.117.39:8080
# Or run: python3 -m mcp run /root/kronos-mcp/mcp_server.py

@app.get("/mcp/tools")
async def list_tools():
    """List available MCP tools."""
    return {
        "tools": [
            {"name": "forecast_btc", "description": "BTC/USDT 24h price forecast using Kronos AI"},
            {"name": "forecast_eth", "description": "ETH/USDT 24h price forecast using Kronos AI"},
            {"name": "forecast_crypto", "description": "Any crypto pair 24h forecast", "parameters": {"symbol": "string"}},
            {"name": "market_snapshot", "description": "BTC, ETH, SOL, DOGE snapshot"},
        ],
        "endpoint": "http://172.86.117.39:8080",
        "mcp_command": "python3 -m mcp run /root/kronos-mcp/mcp_server.py",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Kronos MCP Server starting on port {port}")
    print(f"  REST:   http://0.0.0.0:{port}/")
    print(f"  MCP:    http://0.0.0.0:{port}/mcp")
    print(f"  Health: http://0.0.0.0:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port)
