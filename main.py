from BlackScholes.FSImpliedVol import ImpliedVolCalculatorFS
from BlackScholes.VanillaImpliedVolSmile import ImpliedVolCalculatorVanilla
from Calibrator.Calibrator import plot_heston_vs_market
from imports import *
from PnL_Analysis.Path_Simulation import ForwardStartAnalysis
import json

def run(args):
    args = json.loads(args)
    action = args.get("action", None)
    calibrator_ticker = args.get("calibrator_ticker", None)
    ticker = args.get("ticker", None)
    params_fs = list(args.get("params_fs", {}).values())
    params_vanilla = args.get("params_vanilla", None)

    if "GET_VANILLA_PRICE" in action:
        price = VanillaHestonPrice(*params_vanilla).heston_price()
        print(f"Vanilla Heston price: {price.item()}")

    if "GET_FS_PRICE" in action:
        price = ForwardStart(*params_fs).heston_price()
        print(f"Forward Start price: {price.item()}")

    if "GET_SINGLE_FS_GREEK" in action:
        FS = ForwardStart(*params_fs)
        delta = FS.compute_first_order_greek("vega")
        print(f"value {delta}")

    if "DISPLAY_FS_GREEKS" in action:
        DisplayGreeks(*params_fs).display()

    if "CALIBRATE_HESTON_MODEL" in action:
        # Recuperation et traitement des données sur yf
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price, risk_free_rate = get_yfinance_data(ticker, to_csv=True, filter_data=True)
        S0 = spot_price[0]
        r = risk_free_rate[0]
        # Calibration de Heston
        calibrated_params = Calibrator(S0, lastPrice, strike, timetomaturity, r).calibrate(max_epochs=1500)
        # Display graphique des prix du marché vs des prix calculés avec Heston et avec les parametres calibrés
        plot_heston_vs_market(S0, lastPrice, strike, timetomaturity, r, calibrated_params)
        print(f"Calibrated Parameters: {calibrated_params}")
        # Display du la vol implicite pour les Calls Forward Start
        ImpliedVolCalculatorFS(S0=S0,
                             k_values=[],
                             T0=0.0, T1=1.0,
                             T2=[], r=r,
                             kappa=calibrated_params["kappa"],
                             v0=calibrated_params["v0"],
                             theta=calibrated_params["theta"],
                             sigma=calibrated_params["sigma"],
                             rho=calibrated_params["rho"]).plot_IV_smile()

        #Display de la vol implicite smile pour les Calls Vanille
        ImpliedVolCalculatorVanilla(S0=S0,
                                    k_values=[],
                                    T=1.0,
                                    r=r,
                                    kappa=calibrated_params["kappa"],
                                    v0=calibrated_params["v0"],
                                    theta=calibrated_params["theta"],
                                    sigma=calibrated_params["sigma"],
                                    rho=calibrated_params["rho"]).plot_IV_smile()

        # Affichage des Grecques
        DisplayGreeks(S0=S0,
                      k=1.0,
                      T0=0.0,
                      T1=1.0,
                      T2=3.0,
                      r=r,
                      kappa=calibrated_params["kappa"],
                      v0=calibrated_params["v0"],
                      theta=calibrated_params["theta"],
                      sigma=calibrated_params["sigma"],
                      rho=calibrated_params["rho"]).display()

        # PnL analysis
        params = {
            "S0": S0,  # Prix initial du sous-jacent
            "r": r,  # Taux sans risque
            "kappa": calibrated_params["kappa"],  # Vitesse de réversion
            "theta": calibrated_params["theta"],  # Niveau de variance de long-terme
            "sigma": calibrated_params["sigma"],  # Volatilité de la variance (vol of vol)
            "rho": calibrated_params["rho"],  # Corrélation entre Wt^S et Wt^v
            "v0": calibrated_params["v0"],  # Variance initiale
            "T": 1.0,  # Horizon de simulation (1 an)
            "dt": 1 / 252,  # Pas de temps journalier
            "device": "cuda"  # Exécution sur GPU
        }

        analysis = ForwardStartAnalysis(params)
        PnL_inexpliqué_list = analysis.compute_pnl_analysis(num_simulations=2000)
        analysis.analyze_results(PnL_inexpliqué_list)

    if "GET_YF_IV" in action:
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        vol = implied_vol(strike, timetomaturity, lastPrice)



if __name__ == '__main__':
    input = {
        "action": ["CALIBRATE_HESTON_MODEL"],
        "ticker": "AMZN",
        "params_fs" : {"S0": 575.92, "k": 1.0, "t0": 0.0, "T1": 1.0, "T2": 3.0, "r": 0.04316, "kappa": 4, "v0": 0.0222788, "theta": 0.00426840169840379, "sigma": 0.11711648513095249, "rho": -0.616869574660294},
        "params_vanilla" : [100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7]
    }

    input_json = json.dumps(input)

    run(input_json)
