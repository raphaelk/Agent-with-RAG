"""
Stock Market Search Tool: Analyzes and retrieves stocks with highest percentage
increase (top gainers) or lowest percentage decrease (top losers) based on query context.
"""
import sys
import json
from typing import List, Dict, Any

# Benchmark market snapshot with tickers, prices, daily changes, and sector
MARKET_SNAPSHOT = [
    {"ticker": "NVDA", "company": "NVIDIA Corporation", "price": 128.50, "change_pct": 6.84, "volume": "54.2M", "sector": "Semiconductors"},
    {"ticker": "SMCI", "company": "Super Micro Computer", "price": 48.20, "change_pct": 5.42, "volume": "18.1M", "sector": "Technology"},
    {"ticker": "PLTR", "company": "Palantir Technologies", "price": 36.90, "change_pct": 4.75, "volume": "32.0M", "sector": "Enterprise Software"},
    {"ticker": "TSLA", "company": "Tesla Inc.", "price": 242.15, "change_pct": 3.92, "volume": "41.6M", "sector": "Automotive & Energy"},
    {"ticker": "AMD", "company": "Advanced Micro Devices", "price": 156.30, "change_pct": 3.10, "volume": "22.5M", "sector": "Semiconductors"},
    {"ticker": "MSFT", "company": "Microsoft Corporation", "price": 428.10, "change_pct": 1.45, "volume": "19.3M", "sector": "Cloud & Software"},
    {"ticker": "AAPL", "company": "Apple Inc.", "price": 224.80, "change_pct": 0.85, "volume": "28.4M", "sector": "Consumer Electronics"},
    {"ticker": "GOOGL", "company": "Alphabet Inc.", "price": 164.20, "change_pct": -0.42, "volume": "14.7M", "sector": "Internet & Search"},
    {"ticker": "AMZN", "company": "Amazon.com Inc.", "price": 186.50, "change_pct": -1.15, "volume": "16.8M", "sector": "E-Commerce & Cloud"},
    {"ticker": "META", "company": "Meta Platforms Inc.", "price": 512.40, "change_pct": -1.82, "volume": "12.3M", "sector": "Social Media"},
    {"ticker": "INTC", "company": "Intel Corporation", "price": 19.80, "change_pct": -3.20, "volume": "45.0M", "sector": "Semiconductors"},
    {"ticker": "BA", "company": "The Boeing Company", "price": 152.60, "change_pct": -4.65, "volume": "11.2M", "sector": "Aerospace & Defense"},
    {"ticker": "NKE", "company": "Nike Inc.", "price": 78.40, "change_pct": -5.30, "volume": "15.9M", "sector": "Consumer Discretionary"},
    {"ticker": "WBA", "company": "Walgreens Boots Alliance", "price": 9.10, "change_pct": -7.12, "volume": "21.4M", "sector": "Healthcare Retail"}
]

def get_stocks_by_performance(mode: str = "gainers", limit: int = 5) -> Dict[str, Any]:
    """
    Retrieve stocks with either the highest percentage increase ('gainers')
    or lowest percentage decrease / biggest drop ('losers').
    """
    mode_clean = mode.lower().strip()
    if "loser" in mode_clean or "decrease" in mode_clean or "drop" in mode_clean or "down" in mode_clean or "fall" in mode_clean:
        sorted_stocks = sorted(MARKET_SNAPSHOT, key=lambda x: x["change_pct"])
        result_type = "Top Percentage Decliners (Losers)"
    else:
        sorted_stocks = sorted(MARKET_SNAPSHOT, key=lambda x: x["change_pct"], reverse=True)
        result_type = "Top Percentage Gainers"

    selected = sorted_stocks[:limit]
    return {
        "status": "success",
        "category": result_type,
        "count": len(selected),
        "stocks": selected
    }

def analyze_stock_query(user_query: str) -> Dict[str, Any]:
    """
    Determine whether user query asks for gainers or losers, and return corresponding stocks.
    """
    query_lower = user_query.lower()
    if any(w in query_lower for w in ["drop", "loser", "decrease", "decline", "fall", "worst", "down", "negative"]):
        return get_stocks_by_performance(mode="losers", limit=5)
    return get_stocks_by_performance(mode="gainers", limit=5)

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "gainers"
    print(json.dumps(analyze_stock_query(q), indent=2))
