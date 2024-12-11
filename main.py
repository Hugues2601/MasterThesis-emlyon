from Calibrator import calibrate
from DataRetriever import get_yfinance_data, get_treasury_yield, store_to_csv
from config import CONFIG
from datetime import datetime

def run(input):
    action = input.get("action", None)
    calibrator_ticker = input.get("calibrator_ticker", None)

    if "GET_TREASURY_YIELD" in action:
        r = get_treasury_yield()
        print(f"10y treasury yield as of {datetime.today()}: {r*100}%")

    if "GET_AND_STORE_DATA" in action:
        store_to_csv()

    if "RUN_FS_GREEKS" in action:
        pass


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
        "actions" : ["GET_AND_STORE_DATA", "GET_TREASURY_YIELD"]
    }


    run(input)