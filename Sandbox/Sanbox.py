# %%
"""
Created on Jan 2 2019
The Heston model and pricing of forward start options
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum
import scipy.optimize as optimize
import yfinance as yf
import time

stock = yf.Ticker("TSLA")
spot_price = stock.history(period="1d")['Close'].iloc[-1]
print(spot_price)