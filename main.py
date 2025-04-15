from BlackScholes.FSImpliedVol import ImpliedVolCalculatorFS
from BlackScholes.VanillaImpliedVolSmile import ImpliedVolCalculatorVanilla, plot_implied_volatility, plot_comparative_IV_smile
from Calibrator.Calibrator import plot_heston_vs_market, plot_residuals_heston
from HestonModel.ForwardStart import plot_forward_start_vs_vanilla_price_multi_maturity, \
    plot_forward_start_price_t0_variation
from imports import *
import json
import numpy as np
import matplotlib.pyplot as plt
from PnL_Analysis.Path_Simulation import pnl_analysis
import torch
import time

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
        torch.set_default_dtype(torch.float32)
        print("\n" + "=" * 50)
        print(" DES DONNÉES AVEC YAHOO FINANCE")
        print("=" * 50 + "\n")
        df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price, risk_free_rate = get_yfinance_data(ticker, get_csv_name="SPX_DATA_polygon_20250321_CLEANED_r_0_03927")
        S0 = spot_price[0]
        r = risk_free_rate[0]

        calibrated_params = {'kappa': 2.41300630569458, 'v0': 0.029727613553404808, 'theta': 0.04138144478201866,
                             'sigma': 0.3084869682788849, 'rho': -0.8905978202819824}

        # k_values = np.linspace(0.6, 1.4, 100)
        # plot_forward_start_vs_vanilla_price_multi_maturity(
        #     S0=S0,
        #     k_range=k_values,
        #     r=r,
        #     kappa=calibrated_params["kappa"],
        #     v0=calibrated_params["v0"],
        #     theta=calibrated_params["theta"],
        #     sigma=calibrated_params["sigma"],
        #     rho=calibrated_params["rho"]
        # )
        #
        # plot_forward_start_price_t0_variation(
        #     S0=S0,
        #     k_range=k_values,
        #     r=r,
        #     kappa=calibrated_params["kappa"],
        #     v0=calibrated_params["v0"],
        #     theta=calibrated_params["theta"],
        #     sigma=calibrated_params["sigma"],
        #     rho=calibrated_params["rho"]
        # )

        # price = ForwardStart(S0=S0, k=1.0, T0=0.0, T1=1.0, T2=2.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
        #                      sigma=calibrated_params["sigma"], rho=calibrated_params["rho"]).heston_price()
        # print(f"Vanilla Heston price: {price.item()}")
        #
        # price = ForwardStart(S0=S0, k=1.0, T0=0.5, T1=1.0, T2=2.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
        #                      sigma=calibrated_params["sigma"], rho=calibrated_params["rho"]).heston_price()
        # print(f"Vanilla Heston price: {price.item()}")
        #
        # price = ForwardStart(S0=S0, k=1.0, T0=0.75, T1=1.0, T2=2.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
        #                      sigma=calibrated_params["sigma"], rho=calibrated_params["rho"]).heston_price()
        # print(f"Vanilla Heston price: {price.item()}")

        # start_time = time.time()
        #
        # # Calibration de Heston
        # print("\n" + "=" * 50)
        # print(" CALIBRATION DU MODÈLE HESTON")
        # print("=" * 50 + "\n")
        # calibrated_params = Calibrator(S0, lastPrice, strike, timetomaturity, r).calibrate(max_epochs=7000)
        # print("\nParamètres calibrés :")
        # print(calibrated_params)
        #
        #
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        #
        # print(f"Temps d'exécution de la calibration: {elapsed_time:.2f} secondes")

        # # Display graphique des prix du marché vs des prix calculés avec Heston et avec les parametres calibrés
        # print("\n" + "=" * 50)
        # print(" COMPARAISON PRIX DU MARCHÉ VS HESTON")
        # print("=" * 50 + "\n")
        # plot_heston_vs_market(S0, lastPrice, strike, timetomaturity, r, calibrated_params)
        #
        # plot_residuals_heston(
        #     VanillaHestonPrice,
        #     S0=S0,
        #     K_list=strike,
        #     T_list=timetomaturity,
        #     r=r,
        #     market_prices=lastPrice,
        #     kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"], sigma=calibrated_params["sigma"], rho=calibrated_params["rho"],
        #     plot_type="strike"  # ou "maturity"
        # )

        #
        print("\n" + "=" * 50)
        print(" VARIATIONS DES PARAMÈTRES CALIBRÉS (HESTON)")
        print("=" * 50 + "\n")

        fs_option = ForwardStart(S0=S0, k=1.0, T0=0.0, T1=.75, T2=1.5, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
                                 sigma=calibrated_params["sigma"], rho=calibrated_params["rho"])
        fs_option.sensitivity_analysis_all(S0, r, calibrated_params)
        #
        print("\n" + "=" * 50)


        # print(" IMPLICIT VOLATILITY SMILE (CALLS FORWARD START)")
        # print("=" * 50 + "\n")
        # # Display du la vol implicite pour les Calls Forward Start
        # ImpliedVolCalculatorFS(S0=S0,
        #                      k_values=[],
        #                      T0=0.0, T1=0.75,
        #                      T2=[], r=r,
        #                      kappa=calibrated_params["kappa"],
        #                      v0=calibrated_params["v0"],
        #                      theta=calibrated_params["theta"],
        #                      sigma=calibrated_params["sigma"],
        #                      rho=calibrated_params["rho"]).plot_IV_smile()
        #
        # print("\n" + "=" * 50)


        # """ PLOT DE SURFACES DE VOLATILITE """
        #
        # ImpliedVolCalculatorFS(S0=S0,
        #                        k_values=[],
        #                        T0=0.0, T1=1.0,
        #                        T2=[], r=r,
        #                        kappa=calibrated_params["kappa"],
        #                        v0=calibrated_params["v0"],
        #                        theta=calibrated_params["theta"],
        #                        sigma=calibrated_params["sigma"],
        #                        rho=calibrated_params["rho"]).plot_IV_surface()

        # print("IMPLICIT VOLATILITY SMILE (CALLS VANILLE)")
        # print("=" * 50 + "\n")
        # #Display de la vol implicite smile pour les Calls Vanille
        # ImpliedVolCalculatorVanilla(S0=S0,
        #                             k_values=[],
        #                             T=1.0,
        #                             r=r,
        #                             kappa=calibrated_params["kappa"],
        #                             v0=calibrated_params["v0"],
        #                             theta=calibrated_params["theta"],
        #                             sigma=calibrated_params["sigma"],
        #                             rho=calibrated_params["rho"]).plot_IV_smile()
        #



        # plot_implied_volatility(strike, impliedVolatility, timetomaturity)
        #
        # plot_comparative_IV_smile(
        #     S0=S0,
        #     strike_market=strike,
        #     iv_market=impliedVolatility,
        #     timetomarket=timetomaturity,
        #     r=r,
        #     kappa=calibrated_params["kappa"],
        #     v0=calibrated_params["v0"],
        #     theta=calibrated_params["theta"],
        #     sigma=calibrated_params["sigma"],
        #     rho=calibrated_params["rho"]
        # )

        #
        # print("\n" + "=" * 50)
        # print("CALCUL DES GRECS")
        # print("=" * 50 + "\n")
        # # Affichage des Grecques
        # DisplayGreeks(S0=S0,
        #               k=1.0,
        #               T0=0.25,
        #               T1=1.0,
        #               T2=2.0,
        #               r=r,
        #               kappa=calibrated_params["kappa"],
        #               v0=calibrated_params["v0"],
        #               theta=calibrated_params["theta"],
        #               sigma=calibrated_params["sigma"],
        #               rho=calibrated_params["rho"]).display()

        # print("\n" + "=" * 50)
        # print("ANALYSE DU PNL INEXPLIQUÉ")
        # print("=" * 50 + "\n")
        #
        # calibrated_params={'kappa': 7.6402974128723145, 'v0': 0.03784135729074478, 'theta': 0.034553635865449905,
        #  'sigma': 0.19902223348617554, 'rho': -0.8408819437026978}
        # S0=5667.65
        # r=0.043
        # pnl_analysis(S0=S0,
        #             k=1.0,
        #             T0=0.25,
        #             T1=1.0,
        #             T2=2.0,
        #             r=r,
        #             kappa=calibrated_params["kappa"],
        #             v0=calibrated_params["v0"],
        #             theta=calibrated_params["theta"],
        #             sigma=calibrated_params["sigma"],
        #              rho=calibrated_params["rho"])




if __name__ == '__main__':
    input = {
        "action": ["CALIBRATE_HESTON_MODEL"],
        "ticker": "AMZN",
        "params_fs" : {"S0": 5712.1011, "k": 1.0, "t0": 0.25, "T1": 1.0, "T2": 2.0, "r": 0.05, "kappa": 0.77576, "v0": 0.007, "theta": 0.03395, "sigma": 0.68546, "rho": -0.864341},
        "params_vanilla" : [100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7]
    }

    input_json = json.dumps(input)

    run(input_json)
