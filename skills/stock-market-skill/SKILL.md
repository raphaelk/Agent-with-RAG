---
name: Stock Market Skill
description: Get the list of stocks with the highest percentage increase (top gainers) or lowest percentage decrease / biggest drops based on the chat question.
Trigger Queries:
  - Which stocks had the highest percentage increase today?
  - What are the top gaining stocks in the market?
  - Show me the stocks with the biggest drops or largest decrease
  - List the worst performing stocks today
  - Give me the top stock gainers and losers
---

# Stock Market Skill

## Overview
This skill analyzes stock market performance movements to rank and return equities demonstrating either the highest percentage appreciation (gainers) or the lowest percentage decrease / steepest drops (losers) in response to the user query.

## Standard Operating Procedure (SOP)
1. **Detect Intent (Gainers vs. Decliners)**:
   - Identify whether the user is querying for positive surges (gainers/increase) or negative contractions (losers/decrease/drop).
2. **Execute Stock Ranking Tool**:
   - Run `skills/stock-market-skill/scripts/stock_search.py` or invoke `analyze_stock_query(query)`.
3. **Format Response**:
   - Return ticker symbol, company name, latest price, percentage change, and trading volume in an intuitive tabular or structured summary.
