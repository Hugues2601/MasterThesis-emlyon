import torch
import matplotlib.pyplot as plt
from Calibrator import calibrate
from HestonModel.ForwardStart import ForwardStart
from HestonModel.Vanilla import VanillaHestonPrice
from DataRetriever import get_yfinance_data, get_treasury_yield, store_to_csv
from config import CONFIG
from datetime import datetime
from DisplayFactory.DisplayGreeks import DisplayGreeks

def run(args):
    action = args.get("action", None)
    calibrator_ticker = args.get("calibrator_ticker", None)
    ticker = args.get("ticker", None)

    if "GET_TREASURY_YIELD" in action:
        r = get_treasury_yield()
        print(f"10y treasury yield as of {datetime.today()}: {r*100}%")

    if "GET_VANILLA_PRICE" in action:
        price = VanillaHestonPrice(100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7).heston_price()
        print(f"Vanilla Heston price: {price.item()}")

    if "GET_FS_PRICE" in action:
        price = ForwardStart(100.0, 1.0, 0.0, 1.0, 3.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7).heston_price()
        print(f"Forward Start price: {price.item()}")

    if "GET_AND_STORE_DATA" in action:
        store_to_csv()

    if "GET_SINGLE_FS_GREEK" in action:
        FS = ForwardStart(100.0, 1.05, 0.0, 1.0, 3.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7)
        delta = FS.compute_first_order_greek("vega", plot=False)
        print(delta)

    if "DISPLAY_FS_GREEKS" in action:
        DisplayGreeks(100.0, 1.05, 0.0, 1.0, 3.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7).display()



    # df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data("SPY")
    # print(f"Nb of options: {len(lastPrice)}")
    # calls_mean = sum(lastPrice) / len(lastPrice)
    # print(f"mean price of options: {calls_mean}")
    # S0 = spot_price[0]
    # initial_guess = CONFIG.initial_guess
    # calibrated_params = calibrate(S0, lastPrice, strike, timetomaturity, r, initial_guess, max_epochs=3000, loss_threshold=0.03*calls_mean)
    # print("Calibrated Parameters:")
    # print(calibrated_params)





if __name__ == '__main__':
    input = {
        "action": ["DISPLAY_FS_GREEKS"],
        "ticker": "^RUT"
    }

    run(input)
