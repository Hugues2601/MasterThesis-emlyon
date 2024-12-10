from Calibrator import calibrate
from DataRetriever import get_yfinance_data, get_trasury_yield, store_to_csv
from config import CONFIG


def run(input):
    action = input["action"]

    # r = get_trasury_yield()
    # df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data("SPY")
    # print(f"Nb of options: {len(lastPrice)}")
    # calls_mean = sum(lastPrice) / len(lastPrice)
    # print(f"mean price of options: {calls_mean}")
    # S0 = spot_price[0]
    # initial_guess = CONFIG.initial_guess
    # calibrated_params = calibrate(S0, lastPrice, strike, timetomaturity, r, initial_guess, max_epochs=3000, loss_threshold=0.03*calls_mean)
    # print("Calibrated Parameters:")
    # print(calibrated_params)

    if action == "GET_AND_STORE_DATA":
        store_to_csv()



if __name__ == '__main__':

    input = {
        "action" : "GET_AND_STORE_DATA"
    }


    run(input)