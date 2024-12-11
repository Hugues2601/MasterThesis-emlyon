from Calibrator import calibrate
from DataRetriever import get_yfinance_data, get_treasury_yield, store_to_csv
from config import CONFIG
from datetime import datetime
from Greeks.FS_Greeks_AAD import GreeksFS

def run(input):
    action = input.get("action", None)
    calibrator_ticker = input.get("calibrator_ticker", None)

    if "GET_TREASURY_YIELD" in action:
        r = get_treasury_yield()
        print(f"10y treasury yield as of {datetime.today()}: {r*100}%")

    if "GET_AND_STORE_DATA" in action:
        store_to_csv()

    if "GET_FS_PRICE" in action:
        pass

    if "RUN_SINGLE_FS_GREEK" in action:
        FS = GreeksFS(100.0, 1, 0.0, 1.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.6)
        print(FS.calculate_greek("rho"))

    if "PLOT_ALL_FS_GREEKS" in action:
        FS = GreeksFS(100.0, 1, 0.0, 1.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.6)
        delta_plot = FS.plot_greek("delta")
        vega_plot = FS.plot_greek("vega")
        theta_plot = FS.plot_greek("theta")
        rho_plot = FS.plot_greek("rho")



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
        "action" : ["RUN_SINGLE_FS_GREEK",],
    }


    run(input)