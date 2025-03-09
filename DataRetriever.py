import pandas as pd
import yfinance as yf
from datetime import datetime
import requests

""" --------------- 10Y Treasury Yield --------------"""

def get_treasury_yield() -> float:
    data = yf.download("^TNX", period="1d", interval="1d")
    treasury_yield = (data["Close"].iloc[0])/100
    return treasury_yield


""" --------------------- Options Data from Yahoo Finance ---------------------"""

def get_yfinance_data(symbol: str, to_csv: bool = False):
    stock = yf.Ticker(symbol)
    spot_price = stock.history(period="1d")['Close'].iloc[-1]

    #riskfree
    rate = yf.download("^TNX", period="1d", interval="1d")
    treasury_yield = rate["Close"].iloc[-1].item() / 100
    print(treasury_yield)

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
        (all_calls["openInterest"] > 10) &
        (all_calls["impliedVolatility"] < 1) & (all_calls["impliedVolatility"] > 0.05)
    ]

    calls_list["r"] = treasury_yield

    columns_to_keep = ["contractSymbol",
                       'lastPrice',
                       'strike',
                       'impliedVolatility',
                       'moneyness',
                       'timetomaturity',
                       'expiration',
                       'spot_price',
                       'r'
                       ]
    calls_list = calls_list[columns_to_keep]
    calls_list.reset_index(drop=True, inplace=True)

    if to_csv:
        calls_list.to_csv(f"C:\\Users\\hugue\\Desktop\\Master Thesis\\Data\\{datetime.now().strftime('%Y%m%d')}_{symbol}_clean.csv", index=False)


    lastPrice = calls_list["lastPrice"].tolist()
    strike = calls_list["strike"].tolist()
    impliedVolatility = calls_list["impliedVolatility"].tolist()
    moneyness = calls_list["moneyness"].tolist()
    timetomaturity = calls_list["timetomaturity"].tolist()
    spot_price = calls_list["spot_price"].tolist()
    risk_free_rate = calls_list["r"].tolist()

    return calls_list, lastPrice, timetomaturity, impliedVolatility, strike, spot_price, risk_free_rate

def agg_strikes_and_maturities(symbol:str):
    calls_list, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(symbol)
    # Créer un DataFrame avec les données
    df = pd.DataFrame({
        'strike': strike,
        'timetomaturity': timetomaturity,
        'impliedVolatility': impliedVolatility
    })

    df_cleaned = df.groupby(['strike', 'timetomaturity'], as_index=False).mean()

    strike_cleaned = df_cleaned['strike'].tolist()
    timetomaturity_cleaned = df_cleaned['timetomaturity'].tolist()
    impliedVolatility_cleaned = df_cleaned['impliedVolatility'].tolist()

    return strike_cleaned, timetomaturity_cleaned, impliedVolatility_cleaned

def store_to_csv():
    tickers = ["SPY", "^NDX", "^SPX", "^RUT", "AAPL", "NVDA", "NFLX", "XOM", "MSFT", "META"]
    for ticker in tickers:
        get_yfinance_data(ticker, to_csv=True)
        print(f"{ticker} data saved as of {datetime.now().strftime('%Y%m%d')}")
    print("all done")




# La fonction est prête à être testée avec des données.


""" ------------------------ Options Data from other API --------------------------"""