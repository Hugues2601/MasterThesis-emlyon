import pandas as pd
import yfinance as yf
from datetime import datetime

""" --------------- 10Y Treasury Yield --------------"""

def get_trasury_yield(symbol: str = "^TNX") -> float:
    data = yf.download(symbol, period="1d", interval="1d")
    treasury_yield = (data["Close"].iloc[0])/100
    return treasury_yield

""" --------------------- Options Data from Yahoo Finance ---------------------"""

def get_yfinance_data(symbol: str):
    stock = yf.Ticker(symbol)
    spot_price = stock.history(period="1d")['Close'].iloc[-1]

    expirations = stock.options
    all_calls = pd.DataFrame()

    for expiration_date in expirations:
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls
        calls["expiration"] = expiration_date
        calls["spot_price"] = spot_price
        calls["moneyness"] = calls["spot_price"]/calls["strike"]

        expiration_datetime = datetime.strptime(expiration_date, "%Y-%m-%d")
        today = datetime.today()
        days_to_maturity = (expiration_datetime - today).days
        timetomaturity = days_to_maturity / 252.0
        calls["timetomaturity"] = timetomaturity

        all_calls = pd.concat([all_calls, calls], ignore_index=True)

    calls_list = all_calls[
        (all_calls["moneyness"] > 0.8) & (all_calls["moneyness"] < 1.2) &
        (all_calls["timetomaturity"] > 0.2) & (all_calls["timetomaturity"] < 4) &
        (all_calls["volume"] > 0) &
        (all_calls["openInterest"] > 5) &
        (all_calls["impliedVolatility"] < 1) & (all_calls["impliedVolatility"] > 0.01)
    ]

    columns_to_keep = ["contractSymbol",
                       'lastPrice',
                       'strike',
                       'impliedVolatility',
                       'moneyness',
                       'timetomaturity',
                       'spot_price',
                       ]
    calls_list = calls_list[columns_to_keep]
    calls_list.reset_index(drop=True, inplace=True)

    lastPrice = calls_list["lastPrice"].tolist()
    strike = calls_list["strike"].tolist()
    impliedVolatility = calls_list["impliedVolatility"].tolist()
    moneyness = calls_list["moneyness"].tolist()
    timetomaturity = calls_list["timetomaturity"].tolist()
    spot_price = calls_list["spot_price"].tolist()

    return calls_list, lastPrice, timetomaturity, impliedVolatility, strike, spot_price



""" ------------------------ Options Data from other API --------------------------"""