"""
Standalone MCP server for Kronos prediction tools.
Run: python3 mcp_server.py
Or via stdio: python3 -m mcp run mcp_server.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from predictor import forecast, market_snapshot as snap

# Import based on available MCP library
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("kronos-mcp")
    
    @mcp.tool()
    async def forecast_btc() -> dict:
        """Get BTC/USDT 24h price forecast using Kronos AI."""
        return await forecast("BTCUSDT", "24h")
    
    @mcp.tool()
    async def forecast_eth() -> dict:
        """Get ETH/USDT 24h price forecast using Kronos AI."""
        return await forecast("ETHUSDT", "24h")
    
    @mcp.tool()
    async def forecast_crypto(symbol: str = "BTCUSDT") -> dict:
        """Get 24h price forecast for any crypto pair (e.g. BTCUSDT, ETHUSDT, SOLUSDT)."""
        return await forecast(symbol, "24h")
    
    @mcp.tool()
    async def market_snapshot() -> list:
        """Get BTC, ETH, SOL, and DOGE forecasts in one call."""
        return await snap()
    
    if __name__ == "__main__":
        mcp.run()
        
except ImportError:
    # Fallback: stdio JSON-RPC
    import json, asyncio
    
    TOOLS = [
        {"name": "forecast_btc", "description": "BTC/USDT 24h price forecast", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "forecast_eth", "description": "ETH/USDT 24h price forecast", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "forecast_crypto", "description": "Forecast any crypto pair", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}}},
        {"name": "market_snapshot", "description": "All major pairs", "inputSchema": {"type": "object", "properties": {}}},
    ]
    
    async def handle_request(req):
        if req.get("method") == "tools/list":
            return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"tools": TOOLS}}
        elif req.get("method") == "tools/call":
            name = req.get("params", {}).get("name", "")
            args = req.get("params", {}).get("arguments", {})
            if name == "forecast_btc":
                result = await forecast("BTCUSDT", "24h")
            elif name == "forecast_eth":
                result = await forecast("ETHUSDT", "24h")
            elif name == "forecast_crypto":
                result = await forecast(args.get("symbol", "BTCUSDT"), "24h")
            elif name == "market_snapshot":
                result = await snap()
            else:
                return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32601, "message": "Tool not found"}}
            return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32601, "message": "Method not found"}}
    
    async def main():
        for line in sys.stdin:
            try:
                req = json.loads(line.strip())
                resp = await handle_request(req)
                print(json.dumps(resp), flush=True)
            except Exception as e:
                print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}), flush=True)
    
    if __name__ == "__main__":
        asyncio.run(main())
