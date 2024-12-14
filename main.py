from Calibrator.Calibrator import calibrate
from HestonModel.ForwardStart import ForwardStart
from HestonModel.Vanilla import VanillaHestonPrice
from DataRetriever import get_treasury_yield, store_to_csv, get_yfinance_data
from config import CONFIG
from datetime import datetime
from DisplayFactory.DisplayGreeks import DisplayGreeks
from Sandbox.webscrapping import process
from Calibrator.Calibrator import calibrate
from Calibrator.HMCalibration import heston_price

def run(args):
    action = args.get("action", None)
    calibrator_ticker = args.get("calibrator_ticker", None)
    ticker = args.get("ticker", None)
    params_fs = args.get("params_fs", None)
    params_vanilla = args.get("params_vanilla", None)

    if "GET_TREASURY_YIELD" in action:
        r = get_treasury_yield()
        print(f"10y treasury yield as of {datetime.today()}: {r*100}%")

    if "GET_VANILLA_PRICE" in action:
        price = VanillaHestonPrice(*params_vanilla).heston_price()
        print(f"Vanilla Heston price: {price.item()}")

    if "GET_FS_PRICE" in action:
        price = ForwardStart(*params_fs).heston_price()
        print(f"Forward Start price: {price.item()}")

    if "GET_AND_STORE_DATA" in action:
        store_to_csv()

    if "GET_SINGLE_FS_GREEK" in action:
        FS = ForwardStart(*params_fs)
        delta = FS.compute_first_order_greek("vega")
        print(delta)

    if "DISPLAY_FS_GREEKS" in action:
        DisplayGreeks(*params_fs).display()

    if "CALIBRATE_HESTON_MODEL" in action:
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        # df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = process()
        print(f"Nb of options: {len(lastPrice)}")
        calls_mean = sum(lastPrice) / len(lastPrice)
        print(f"mean price of options: {calls_mean}")
        S0 = spot_price[0]
        initial_guess = CONFIG.initial_guess
        calibrated_params = calibrate(S0, lastPrice, strike, timetomaturity, r, initial_guess, max_epochs=5000, loss_threshold=0.03*calls_mean)
        print("Calibrated Parameters:")
        print(calibrated_params)





if __name__ == '__main__':
    input = {
        "action": ["DISPLAY_FS_GREEKS"],
        "ticker": "^NDX",
        "params_fs" : [100.0, 1.0, 0.0, 1.0, 3.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7],
        "params_vanilla" : [100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7]
    }

    run(input)
