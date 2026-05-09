# NSE Intelligent Investor — Dataset Documentation

This document explains all 14 datasets used in the project. Written in simple language so anyone on the team can understand what each file contains and why it exists.

---

## Quick Overview

| # | File Name | Rows | What It Is |
|---|-----------|------|------------|
| 1 | bulk_deals_clean.csv | ~67,270 | Large block trades on NSE |
| 2 | cleaned_india_vix.csv | ~737 | India's market fear index |
| 3 | cleaned_nifty50_index.csv | ~740 | Nifty 50 daily index data |
| 4 | cleaned_nifty500_index.csv | ~737 | Nifty 500 daily index data |
| 5 | cleaned_usdinr.csv | ~784 | USD to INR exchange rate |
| 6 | corporate_announcements_nse.csv | ~321 | Dividends & corporate events |
| 7 | india_10y_gsec_complete.csv | ~967 | 10-year government bond yield |
| 8 | insider_trading_clean.csv | ~89,578 | Promoter/director trades |
| 9 | ipo_data_merged.csv | ~587 | IPO listing & performance data |
| 10 | mutual_fund_data.csv | ~16,350 | Mutual fund NAV & AUM data |
| 11 | nifty50_ohlcv_master.csv | ~1,153 | Full OHLCV for all 50 Nifty stocks |
| 12 | nifty50_prices.csv | ~740 | Daily closing prices — all 50 stocks |
| 13 | nifty50_returns.csv | ~739 | Daily % returns — all 50 stocks |
| 14 | nifty50_sector_mapping.csv | ~50 | Sector/industry label per stock |

---

## Detailed Breakdown

---

### 1. `bulk_deals_clean.csv`
**Size:** ~67,270 rows, 8 columns

**What it is:**
Every day, NSE publishes "bulk deals" — transactions where someone buys or sells a very large block of shares (typically more than 0.5% of total shares in one go). This file is a cleaned record of all such transactions.

**Columns:**
- `Date` — when the deal happened
- `Symbol` — stock ticker (e.g. RELIANCE)
- `Security` — full company name
- `Client` — name of the buyer or seller (often a fund, broker, or institution)
- `Transaction_Type` — BUY or SELL
- `Quantity Traded` — how many shares changed hands
- `Price` — at what price per share
- `Remarks` — any additional notes

**Used for:**
Module 1 (Opportunity Radar). When a large institutional player suddenly buys or sells a big chunk of a stock, it's a signal worth paying attention to. This data feeds the "Bulk" signal pill in the radar table.

---

### 2. `cleaned_india_vix.csv`
**Size:** ~737 rows, 6 columns

**What it is:**
India VIX is the "fear index" of the Indian market. A high VIX means the market is nervous and volatile. A low VIX means the market is calm. It's derived from Nifty options prices.

**Columns:**
- `Date` — trading date
- `Open`, `High`, `Low`, `Close` — VIX value at different points of the day
- `Volume` — always 0 (VIX is an index, not a tradeable instrument)

**Used for:**
Context layer for the risk engine. When VIX is high, the portfolio optimizer applies more caution. Also useful for the live top bar that shows current market conditions.

---

### 3. `cleaned_nifty50_index.csv`
**Size:** ~740 rows, 6 columns

**What it is:**
The daily price data for the Nifty 50 index itself (not individual stocks — just the overall index level).

**Columns:**
- `Date` — trading date
- `Open`, `High`, `Low`, `Close` — index value
- `Volume` — total market volume on that day

**Used for:**
Benchmark comparison. When the system calculates "Alpha vs Nifty," it compares your portfolio's returns against this index. Also powers the live Nifty level shown in the website's top bar.

---

### 4. `cleaned_nifty500_index.csv`
**Size:** ~737 rows, 6 columns

**What it is:**
Same structure as the Nifty 50 index file, but for the Nifty 500 — a broader index covering the top 500 companies. Represents a much wider slice of the Indian market.

**Columns:** Same as above — `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

**Used for:**
Alternative benchmark. Useful for evaluating performance against a broader market (not just the top 50 companies).

---

### 5. `cleaned_usdinr.csv`
**Size:** ~784 rows, 6 columns

**What it is:**
Daily USD to Indian Rupee exchange rate data. When the rupee weakens against the dollar, it affects IT companies (who earn in USD) differently from energy or banking companies.

**Columns:**
- `Date` — trading date
- `Open`, `High`, `Low`, `Close` — exchange rate for that day
- `Volume` — 0 (currency pairs don't have traditional volume)

**Used for:**
Macro context input. Currency movement is a background signal that can influence sector-level risk, particularly for export-heavy sectors like IT and Pharma.

---

### 6. `corporate_announcements_nse.csv`
**Size:** ~321 rows, 9 columns

**What it is:**
Official NSE announcements of corporate events — primarily dividends but also other board-level decisions like buybacks, splits, and board meetings.

**Columns:**
- `symbol` — stock ticker
- `COMPANY NAME` — full company name
- `SERIES` — share type (EQ = equity)
- `PURPOSE` — what the announcement is about (e.g. "Interim Dividend - Rs 6 Per Share")
- `FACE VALUE` — face value of the share
- `EX-DATE` — date from which the stock trades without the dividend
- `RECORD DATE` — date to determine eligible shareholders
- `BOOK CLOSURE START DATE` / `BOOK CLOSURE END DATE` — period when the share register is closed

**Used for:**
Module 1 (Opportunity Radar). A dividend announcement, board meeting, or other corporate action is a "Filing" signal. When this fires alongside Bulk and Insider signals, it creates a multi-source convergence — the strongest type of signal the radar can produce.

---

### 7. `india_10y_gsec_complete.csv`
**Size:** ~967 rows, 6 columns

**What it is:**
The yield on India's 10-year government bond (G-Sec). This is the "risk-free rate" — what you'd earn if you put your money in the safest possible investment (government debt). It's a critical benchmark for all return calculations.

**Columns:**
- `Date` — date
- `Price` — the bond yield (%) for that day
- `Open`, `High`, `Low` — yield range during the day
- `Change %` — daily change in yield

**Used for:**
Risk engine calculations. The Sharpe Ratio formula requires subtracting the risk-free rate from portfolio returns. This dataset provides that rate. It's the foundation for metrics like Alpha, Sharpe, and Sortino shown in Module 2's risk dashboard.

---

### 8. `insider_trading_clean.csv`
**Size:** ~89,578 rows, 12 columns

**What it is:**
SEBI-mandated disclosures: every time a company director, promoter, or Key Managerial Personnel (KMP) buys or sells shares of their own company, they must report it. This dataset captures all such disclosures.

**Columns:**
- `Symbol` — stock ticker
- `Company` — company name
- `Insider_Name` — name of the person trading
- `Category` — their role (Director, KMP, Promoter, etc.)
- `Quantity` — shares traded
- `Value` — total transaction value in ₹
- `Transaction_Type` — Buy or Sell
- `Acquisition_Date` — when the trade actually happened
- `Intimation_Date` — when they informed the exchange
- `Broadcast_Date` — when NSE published it
- `Action` — BUY or SELL (simplified)
- `Is_Priority_Insider` — True if this is a high-credibility insider (director/KMP/promoter)

**Used for:**
Module 1 (Opportunity Radar). When a company's own director or promoter buys shares — especially in large quantities — it's a strong signal that insiders believe in the company's future. This fires the "Insider" signal pill in the radar table.

---

### 9. `ipo_data_merged.csv`
**Size:** ~587 rows, 13 columns

**What it is:**
Data on IPOs listed on NSE and BSE, including their listing performance (did the stock open higher or lower than the issue price?) and current price.

**Columns:**
- `Company` — company name
- `Opening Date` — IPO subscription open date
- `Listing Date` — when the stock started trading
- `Listing At` — which exchange (NSE or BSE)
- `ISIN` — unique stock identifier
- `BSE Scrip Code` / `NSE Symbol` — exchange-specific codes
- `Issue Price (Rs.)` — price at which shares were sold in IPO
- `Close Price on Listing (Rs.)` — first-day closing price
- `% Gain/Loss (Issue price vs close price on Listing)` — listing day return
- `Current Price at BSE / NSE` — present market price
- `Gain / Loss (%)` — total return since issue price

**Used for:**
Supplementary data for Opportunity Radar. Recently listed stocks with unusual post-listing behaviour can be flagged as signals worth watching. Also useful for the Dashboard's market overview.

---

### 10. `mutual_fund_data.csv`
**Size:** ~16,350 rows, 16 columns

**What it is:**
A comprehensive database of Indian mutual fund schemes — their NAV (Net Asset Value, essentially the price per unit), AUM (how much money they manage), and scheme details.

**Columns:**
- `Scheme_Code` — unique AMFI code for the scheme
- `Scheme_Name` — full name of the fund
- `AMC` — fund house (e.g. Aditya Birla Sun Life AMC)
- `Scheme_Type` — Open Ended / Close Ended
- `Scheme_Category` — e.g. Large Cap, Mid Cap, Debt, etc.
- `Scheme_NAV_Name` — variant name (Growth, IDCW, etc.)
- `Scheme_Min_Amt` — minimum investment amount (₹)
- `NAV` — current NAV per unit
- `Latest_NAV_Date` — when the NAV was last updated
- `Average_AUM_Cr` — average assets under management in crores
- `AAUM_Quarter` — which quarter the AUM data is from
- `ISIN_Div_Payout/Growth` / `ISIN_Div_Reinvestment` — ISINs for different plan variants
- `Launch_Date` / `Closure_Date` — fund lifetime dates

**Used for:**
Context enrichment and future feature expansion. Useful for comparing a stock's performance against equivalent mutual funds. Also relevant for users who want to benchmark whether their direct stock portfolio beats a simple mutual fund investment.

---

### 11. `nifty50_ohlcv_master.csv`
**Size:** ~1,153 rows, 251 columns

**What it is:**
The most detailed price file in the dataset. Contains full OHLCV (Open, High, Low, Close, Volume) data for all 50 Nifty stocks, in a wide format where each stock gets 5 columns.

**Structure:**
- First column (`Price`) acts as a date/ticker header
- Then for each of the 50 stocks: `TICKER_Close`, `TICKER_High`, `TICKER_Low`, `TICKER_Open`, `TICKER_Volume`
- Covers all 50 Nifty 50 constituents (ADANIENT, TCS, HDFCBANK, RELIANCE, INFY, etc.)

**Used for:**
Module 3 (Chart Pattern Intelligence). Pattern detection — breakouts, double tops, head & shoulders, support bounces, etc. — requires the full price history including intraday High/Low, which this file provides. The LSTM model also uses this for technical feature engineering.

---

### 12. `nifty50_prices.csv`
**Size:** ~740 rows, 51 columns

**What it is:**
A clean, simple table of daily closing prices for all 50 Nifty stocks. One row per day, one column per stock.

**Columns:**
- `Date` — trading date
- One column per stock (50 total) — daily closing price in ₹

**Used for:**
Core input to the LSTM model and Portfolio Optimizer (Module 2). The LSTM is trained on this price history to predict future returns. Also used to calculate portfolio value over time.

---

### 13. `nifty50_returns.csv`
**Size:** ~739 rows, 51 columns

**What it is:**
The daily percentage change in closing price for each of the 50 Nifty stocks. Derived from `nifty50_prices.csv` but pre-calculated so the backend doesn't have to compute it every time.

**Columns:**
- `Date` — trading date
- One column per stock (50 total) — that day's return as a decimal (e.g. 0.0081 = +0.81%)

**Used for:**
Risk engine (Module 2). All risk metrics — Sharpe Ratio, Sortino Ratio, CVaR, Volatility, Max Drawdown — are calculated from return distributions. This file is the direct input for those calculations. Also feeds the Efficient Frontier chart.

---

### 14. `nifty50_sector_mapping.csv`
**Size:** ~50 rows, 3 columns

**What it is:**
A simple lookup table that maps each Nifty 50 stock to its sector and industry.

**Columns:**
- `Symbol` — stock ticker (e.g. TCS.NS)
- `sector` — broad sector (e.g. IT, Financials, Energy, Auto)
- `industry` — more specific industry (e.g. IT Services, Private Banks, Refineries)

**Used for:**
Everywhere. The sector filter in Module 1's Opportunity Radar uses this. The sector column in the signals table uses this. Portfolio-level sector concentration analysis in Module 2 uses this. Without this mapping, the system would have no way to group or filter stocks by category.

---

## How the Datasets Connect to Each Module

```
Module 1 — Opportunity Radar
├── bulk_deals_clean.csv          → Bulk signal
├── insider_trading_clean.csv     → Insider signal  
├── corporate_announcements_nse.csv → Filing signal
└── nifty50_sector_mapping.csv    → Sector filter

Module 2 — LSTM Portfolio Optimizer
├── nifty50_prices.csv            → LSTM training data
├── nifty50_returns.csv           → Risk metric calculations
├── india_10y_gsec_complete.csv   → Risk-free rate (Sharpe/Alpha)
├── cleaned_nifty50_index.csv     → Benchmark (Alpha vs Nifty)
└── nifty50_sector_mapping.csv    → Sector concentration display

Module 3 — Chart Pattern Intelligence
├── nifty50_ohlcv_master.csv      → Pattern detection (needs H/L/V)
├── nifty50_prices.csv            → Price context
└── nifty50_sector_mapping.csv    → Stock categorisation

Background / Context
├── cleaned_india_vix.csv         → Market fear level
├── cleaned_usdinr.csv            → Currency macro context
├── cleaned_nifty500_index.csv    → Broader market benchmark
├── ipo_data_merged.csv           → New listing signals
└── mutual_fund_data.csv          → Benchmark comparison
```

---

## Date Coverage Summary

Most datasets cover approximately **August 2021 to mid-2024**, giving the LSTM roughly 3 years of training data. The G-Sec and bulk deal files extend slightly further back to January 2021. Mutual fund NAV data includes records up to March 2026 (live NAV updates).

---

*Last updated: May 2026*
