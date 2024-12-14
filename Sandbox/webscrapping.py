import pandas as pd
from datetime import datetime



def process():
    df = pd.read_csv(r"C:\Users\hugue\Downloads\spx_quotedata.csv")
    df_filtered = df[
        (df["Last Sale"]>3) &
        (df["Open Interest"]>5) &
        (df["IV"] > 0.01) &
        (df["IV"]<1)
    ]
    
    df_filtered["Expiration Date"] = pd.to_datetime(df_filtered["Expiration Date"], format='%a %b %d %Y')
    today = datetime.today()
    df_filtered["timetomaturity"] = (df_filtered["Expiration Date"] - today).dt.days/252
    
    df_filtered = df_filtered[
        (df_filtered["timetomaturity"] > 0.2)
    ]

    current_spx_price = 6051.09
    df_filtered["current_spx_price"] = current_spx_price
    df_filtered["moneyness"] = df_filtered["current_spx_price"]/df_filtered["Strike"]

    df_filtered = df_filtered[
        (df_filtered["moneyness"]>0.75) &
        (df_filtered["moneyness"]<1.25)
    ]

    lastPrice = df_filtered["Last Sale"].tolist()
    strike = df_filtered["Strike"].tolist()
    impliedVolatility = df_filtered["IV"].tolist()
    moneyness = df_filtered["moneyness"].tolist()
    timetomaturity = df_filtered["timetomaturity"].tolist()
    spot_price = df_filtered["current_spx_price"].tolist()


    return df_filtered, lastPrice, timetomaturity, impliedVolatility, strike, spot_price

