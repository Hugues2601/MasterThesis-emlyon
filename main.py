from BlackScholes.FSImpliedVol import ImpliedVolCalculator
from Calibrator.Calibrator import plot_heston_vs_market
from imports import *
import json

def run(args):
    args = json.loads(args)
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
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price, risk_free_rate = get_yfinance_data(ticker, to_csv=True)
        S0 = spot_price[0]
        r = risk_free_rate[0]
        calibrated_params = Calibrator(S0, lastPrice, strike, timetomaturity, r).calibrate(max_epochs=2000)
        plot_heston_vs_market(S0, lastPrice, strike, timetomaturity, r, calibrated_params)
        print(f"Calibrated Parameters: {calibrated_params}")

    if "GET_YF_IV" in action:
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        vol = implied_vol(strike, timetomaturity, lastPrice)



if __name__ == '__main__':
    input = {
        "action": ["CALIBRATE_HESTON_MODEL"],
        "ticker": "^SPX",
        "params_fs" : {"S0": 237.0, "k": 1.0, "t0": 0.0, "T1": 1.0, "T2": 3.0, "r": 0.0456, "kappa": 2.64059, "v0": 0.07878, "theta": 0.05544, "sigma": 0.215834, "rho": -0.40317},
        "params_vanilla" : [100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7]
    }

    input_json = json.dumps(input)

    run(input_json)
