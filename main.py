from BlackScholes.FSImpliedVol import ImpliedVolCalculator
from imports import *

def run(args):
    action = args.get("action", None)
    calibrator_ticker = args.get("calibrator_ticker", None)
    ticker = args.get("ticker", None)
    params_fs = list(args.get("params_fs", {}).values())
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

    if "DISPLAY_TICKER_SURFACE" in action:
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        calc_IV = ImpliedVolCalculator(0, 0).VanillaImpliedVol(strike, timetomaturity)
        DisplayVolSurface(ticker).display_comparison(calc_IV)

    if "CALIBRATE_HESTON_MODEL" in action:
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        S0 = spot_price[0]
        calibrated_params = Calibrator(S0, lastPrice, strike, timetomaturity, 0.0430).calibrate(max_epochs=2000)
        print(f"Calibrated Parameters: {calibrated_params}")

    if "GET_YF_IV":
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        vol = implied_vol(strike, timetomaturity, lastPrice)



# test de push

if __name__ == '__main__':
    input = {
        "action": ["GET_TREASURY_YIELD", "CALIBRATE_HESTON_MODEL"],
        "ticker": "AMZN",
        "params_fs" : {"S0": 237.0, "k": 1.0, "t0": 0.0, "T1": 1.0, "T2": 3.0, "r": 0.0456, "kappa": 2.64059, "v0": 0.07878, "theta": 0.05544, "sigma": 0.215834, "rho": -0.40317},
        "params_vanilla" : [100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7]
    }

    run(input)
