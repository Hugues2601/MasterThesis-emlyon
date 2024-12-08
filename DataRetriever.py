import yfinance as yf

""" --------------- 10Y Treasury Yield --------------"""

def get_trasury_yield(symbol: str = "^TNX") -> float:
    data = yf.download(symbol, period="1d", interval="1d")
    treasury_yield = (data["Close"].iloc[0])/100
    return treasury_yield

