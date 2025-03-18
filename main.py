from BlackScholes.FSImpliedVol import ImpliedVolCalculatorFS
from BlackScholes.VanillaImpliedVolSmile import ImpliedVolCalculatorVanilla
from Calibrator.Calibrator import plot_heston_vs_market
from imports import *
import json
import numpy as np
import matplotlib.pyplot as plt
from PnL_Analysis.Path_Simulation import pnl_analysis
from DisplayFactory.Plot_Market_IV import plot_implied_volatility

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
        delta = FS.compute_greek("theta")
        print(f"value {delta}")

    if "DISPLAY_FS_GREEKS" in action:
        DisplayGreeks(*params_fs).display()

    if "CALIBRATE_HESTON_MODEL" in action:

        # Recuperation et traitement des données sur yf
        print("\n" + "=" * 50)
        print(" DES DONNÉES AVEC YAHOO FINANCE")
        print("=" * 50 + "\n")
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price, risk_free_rate = get_yfinance_data(ticker, get_csv_name="SPX_DATA_polygon_20250313_CLEANED")
        S0 = spot_price[0]
        r = risk_free_rate[0]

        # Calibration de Heston
        print("\n" + "=" * 50)
        print(" CALIBRATION DU MODÈLE HESTON")
        print("=" * 50 + "\n")
        calibrated_params = Calibrator(S0, lastPrice, strike, timetomaturity, r).calibrate(max_epochs=2000)
        print("\nParamètres calibrés :")
        print(calibrated_params)

        # Display graphique des prix du marché vs des prix calculés avec Heston et avec les parametres calibrés
        print("\n" + "=" * 50)
        print(" COMPARAISON PRIX DU MARCHÉ VS HESTON")
        print("=" * 50 + "\n")
        plot_heston_vs_market(S0, lastPrice, strike, timetomaturity, r, calibrated_params)

        print("\n" + "=" * 50)
        print(" VARIATIONS DES PARAMÈTRES CALIBRÉS (HESTON)")
        print("=" * 50 + "\n")

        fs_option = ForwardStart(S0=S0, k=0.75, T0=0.0, T1=1.0, T2=3.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
                                 sigma=calibrated_params["sigma"], rho=calibrated_params["rho"])
        fs_option.sensitivity_analysis_all(S0, r, calibrated_params)

        print("\n" + "=" * 50)
        print("  🏦 IMPLICIT VOLATILITY SMILE (CALLS FORWARD START)")
        print("=" * 50 + "\n")
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

        print("\n" + "=" * 50)
        print("IMPLICIT VOLATILITY SMILE (CALLS VANILLE)")
        print("=" * 50 + "\n")
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

        plot_implied_volatility(strike, impliedVolatility, timetomaturity)

        print("\n" + "=" * 50)
        print("CALCUL DES GRECS")
        print("=" * 50 + "\n")
        # Affichage des Grecques
        DisplayGreeks(S0=S0,
                      k=1.0,
                      T0=0.25,
                      T1=1.0,
                      T2=2.0,
                      r=r,
                      kappa=calibrated_params["kappa"],
                      v0=calibrated_params["v0"],
                      theta=calibrated_params["theta"],
                      sigma=calibrated_params["sigma"],
                      rho=calibrated_params["rho"]).display()

        print("\n" + "=" * 50)
        print("ANALYSE DU PNL INEXPLIQUÉ")
        print("=" * 50 + "\n")

        pnl_analysis(S0=S0,
                    k=1.0,
                    T0=0.25,
                    T1=1.0,
                    T2=2.0,
                    r=r,
                    kappa=calibrated_params["kappa"],
                    v0=calibrated_params["v0"],
                    theta=calibrated_params["theta"],
                    sigma=calibrated_params["sigma"],
                     rho=calibrated_params["rho"])

        # Analyse du PnL unexplained : on génére genre 10 000 chemins avec les parametres calibrés
        # puis on calcule pnl total entre deux instant de chaque chemin, pnl expliqué avec les grecs
        # On calcule a chaque fois le pnl inexpliqué et on regarde sa répartition


    if "GET_YF_IV" in action:
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data(ticker)
        vol = implied_vol(strike, timetomaturity, lastPrice)



if __name__ == '__main__':
    input = {
        "action": ["CALIBRATE_HESTON_MODEL"],
        "ticker": "TSLA",
        "params_fs" : {"S0": 5712.1011, "k": 1.0, "t0": 0.25, "T1": 1.0, "T2": 2.0, "r": 0.05, "kappa": 0.77576, "v0": 0.007, "theta": 0.03395, "sigma": 0.68546, "rho": -0.864341},
        "params_vanilla" : [100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7]
    }

    input_json = json.dumps(input)

    run(input_json)
