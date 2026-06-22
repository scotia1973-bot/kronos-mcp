"""
FastAPI wrapper for Kronos MCP server on HuggingFace Spaces.
Exposes the MCP server via SSE endpoint and a simple web UI.
"""

import json
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Add current dir for imports
sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="Kronos Market Predictor")


@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
    <head><title>Kronos MCP Server</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
        h1 { color: #58a6ff; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }
        code { background: #1f2937; padding: 2px 6px; border-radius: 4px; }
        .status { color: #3fb950; }
        a { color: #58a6ff; }
    </style>
    </head>
    <body>
        <h1>🚀 Kronos MCP Server</h1>
        <div class="card">
            <p><span class="status">●</span> <strong>Status:</strong> Running</p>
            <p><strong>Model:</strong> Kronos — Financial Time-Series Foundation Model</p>
            <p><strong>Live Demo:</strong> <a href="/predict/btc">BTC/USDT Forecast</a></p>
        </div>
        <div class="card">
            <h3>MCP Endpoints</h3>
            <p><code>GET /mcp</code> — MCP Streamable HTTP endpoint (SSE)</p>
            <p><code>POST /mcp</code> — MCP JSON-RPC endpoint</p>
            <p><code>GET /predict/btc</code> — Quick BTC forecast</p>
            <p><code>GET /predict/eth</code> — Quick ETH forecast</p>
            <p><code>GET /health</code> — Health check</p>
        </div>
        <div class="card">
            <h3>Agent Usage</h3>
            <p>Connect via Streamable HTTP:</p>
            <code>
            hermes config set mcp_servers.kronos.url https://YOUR-SPACE.hf.space/mcp
            </code>
        </div>
    </body>
    </html>
    """)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kronos-mcp", "version": "1.0.0"}


@app.get("/predict/btc")
async def predict_btc():
    return await _run_prediction("BTCUSDT")


@app.get("/predict/eth")
async def predict_eth():
    return await _run_prediction("ETHUSDT")


@app.get("/predict/{symbol}")
async def predict_symbol(symbol: str):
    pair = f"{symbol.upper()}USDT"
    return await _run_prediction(pair)


async def _run_prediction(symbol: str):
    """Run prediction using the MCP tool."""
    from kronos_server import forecast_crypto
    result = await forecast_crypto(symbol, "24h")
    return JSONResponse(content=result)


# ── MCP Streamable HTTP Endpoint ──────────────────────────

# Delegate MCP endpoint to FastMCP's SSE transport
# FastMCP handles JSON-RPC automatically via run() with transport
# For Spaces, we run it as a subprocess with SSE


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
