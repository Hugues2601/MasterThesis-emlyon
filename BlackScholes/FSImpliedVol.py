import torch
from BlackScholes.ForwardStartBlackScholes import FSBlackScholes
from HestonModel.ForwardStart import ForwardStart
from config import CONFIG
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from DataRetriever import get_yfinance_data, agg_strikes_and_maturities
from BlackScholes.VanillaBlackScholes import VanillaBlackScholes, implied_vol
from HestonModel.Vanilla import VanillaHestonPrice

class ImpliedVolCalculator:
    def __init__(self, k_values, T):
        self.k_values = k_values
        self.T = T

    def FSImpliedVol(self):
        theta = torch.tensor([0.2] * len(self.k_values), device=CONFIG.device, requires_grad=True)
        k_values = torch.tensor(self.k_values, device=CONFIG.device, dtype=torch.float64)  # Conversion en tenseur GPU

        T2_tensor = torch.full_like(k_values, fill_value=self.T)

        optimizer = torch.optim.Adam([theta], lr=0.1)

        for step in range(500):
            sigma = torch.exp(theta)

            FSHeston = ForwardStart(
                S0=227.46, k=k_values, T0=0.0, T1=1.0, T2=T2_tensor, r=0.0410,
                kappa=0.372638, v0=0.066769, theta=0.102714, sigma=0.405173, rho=-0.30918
            ).heston_price()

            FSBS = FSBlackScholes(
                S0=227.46, k=k_values, T1=1.0, T2=self.T, r=0.05, sigma=sigma
            ).price()

            loss = torch.mean((FSHeston - FSBS) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.exp(theta).detach().cpu().numpy()

    def VanillaImpliedVol(self, strikes, maturities):

        k_values = torch.tensor(strikes, device=CONFIG.device, dtype=torch.float64)
        T_values = torch.tensor(maturities, device=CONFIG.device, dtype=torch.float64)

        theta = torch.tensor([0.2] * len(strikes), device=CONFIG.device, requires_grad=True)

        optimizer = torch.optim.Adam([theta], lr=0.1)

        for step in range(500):
            sigma = torch.exp(theta)

            # Prix avec le modèle Heston
            VHeston = VanillaHestonPrice(
                S0=604.0, K=k_values, T=T_values, r=0.0430,
                kappa=2.6947430389551794, v0=0.012811702890798056, theta=0.007170008673845381, sigma=0.11938463973365486, rho=-0.4284149052958534
            ).heston_price()

            VBS = VanillaBlackScholes(
                S0=604.0, K=k_values, T=T_values, r=0.0430, sigma=sigma
            ).price()

            loss = torch.mean((VHeston - VBS) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.exp(theta).detach().cpu().numpy()


# df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data("XOM")
# vol = implied_vol(strike, timetomaturity, lastPrice)
#
# print(timetomaturity, vol)
# calc_IV = ImpliedVolCalculator(0, 0).VanillaImpliedVol(strike, timetomaturity)
# plt.plot(strike, calc_IV)
# plt.plot(strike, vol)
# plt.show()


# # Différentes maturités T
# T = [1.2, 1.4, 1.6, 1.8, 2, 3]
#
# # Génération des strikes k
# k_values = np.linspace(0.2, 1.9, 500)
#
# # Initialisation de la figure
# plt.figure(figsize=(12, 8))
#
# # Définir un fond gris
# plt.gca().set_facecolor('#e6e6e6')  # Couleur de fond gris clair
# plt.gca().grid(True, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
#
# for t in T:
#     IV_T = ImpliedVolCalculator(k_values, t).FSImpliedVol()
#     plt.plot(k_values, IV_T, label=f'T2 = {t}', linewidth=2)
#
# # Personnalisation du graphique
# plt.xlabel('Strike (k)', fontsize=14)
# plt.ylabel('Implied Volatility (IV)', fontsize=14)
# plt.title('Implied Volatility vs Strike (k) for Different Maturities (T)', fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
#
# # Affichage du graphique avec bordure grise
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_color('#4f4f4f')  # Bordure gauche
# plt.gca().spines['bottom'].set_color('#4f4f4f')  # Bordure inférieure
# plt.gca().tick_params(colors='black')  # Couleur des ticks
#
# # Affichage
# plt.show()






#
#
# # Différentes maturités T
# T = [.2, .4, .6, .8, 1, 2]
#
# # Génération des strikes k
# k_values = np.linspace(170, 280, 500)
#
# # Initialisation de la figure
# plt.figure(figsize=(12, 8))
#
# # Définir un fond gris
# plt.gca().set_facecolor('#e6e6e6')  # Couleur de fond gris clair
# plt.gca().grid(True, color='white', linestyle='--', linewidth=0.7, alpha=0.8)
#
# for t in T:
#     IV_T = ImpliedVolCalculator(k_values, t).VanillaImpliedVol()
#     plt.plot(k_values, IV_T, label=f'T2 = {t}', linewidth=2)
#
# # Personnalisation du graphique
# plt.xlabel('Strike (k)', fontsize=14)
# plt.ylabel('Implied Volatility (IV)', fontsize=14)
# plt.title('Implied Volatility vs Strike (k) for Different Maturities (T)', fontsize=16)
# plt.legend(fontsize=12)
# plt.tight_layout()
#
# # Affichage du graphique avec bordure grise
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_color('#4f4f4f')  # Bordure gauche
# plt.gca().spines['bottom'].set_color('#4f4f4f')  # Bordure inférieure
# plt.gca().tick_params(colors='black')  # Couleur des ticks
#
# # Affichage
# plt.show()
#
#



