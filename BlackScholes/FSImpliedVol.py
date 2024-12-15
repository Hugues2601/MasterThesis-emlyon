import torch
from BlackScholes.ForwardStartBlackScholes import FSBlackScholes
from HestonModel.ForwardStart import ForwardStart
from config import CONFIG
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from DataRetriever import get_yfinance_data
from BlackScholes.VanillaBlackScholes import VanillaBlackScholes
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

            # Calcul de FSBS pour chaque valeur de k en batch
            FSBS = FSBlackScholes(
                S0=227.46, k=k_values, T1=1.0, T2=self.T, r=0.05, sigma=sigma
            ).price()

            # Perte moyenne sur toutes les valeurs de k
            loss = torch.mean((FSHeston - FSBS) ** 2)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.exp(theta).detach().cpu().numpy()

    def VanillaImpliedVol(self):


        theta = torch.tensor([0.2] * len(self.k_values), device=CONFIG.device, requires_grad=True)
        k_values = torch.tensor(self.k_values, device=CONFIG.device, dtype=torch.float64)

        T2_tensor = torch.full_like(k_values, fill_value=self.T)

        optimizer = torch.optim.Adam([theta], lr=0.1)

        for step in range(500):
            sigma = torch.exp(theta)

            VHeston = VanillaHestonPrice(
                S0=227.46, K=k_values, T=T2_tensor, r=0.0410,
                kappa=0.372638, v0=0.066769, theta=0.102714, sigma=0.405173, rho=-0.30918
            ).heston_price()

            # Calcul de FSBS pour chaque valeur de k en batch
            VBS = VanillaBlackScholes(
                S0=227.46, K=k_values, T=self.T, r=0.05, sigma=sigma
            ).price()

            # Perte moyenne sur toutes les valeurs de k
            loss = torch.mean((VHeston - VBS) ** 2)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.exp(theta).detach().cpu().numpy()




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


def plot_implied_vol_comparison_subset(df, spot_price, impliedVolCalculator):
    import matplotlib.pyplot as plt

    # Extraire les données nécessaires du DataFrame
    strikes = df['strike'].values
    market_iv = df['impliedVolatility'].values
    maturities = df['timetomaturity'].values

    # Indices des strikes spécifiques
    import random

    # Générer 10 indices aléatoires uniques entre 1 et 300
    indices_to_plot = random.sample(range(1, 223), 10)
    print(indices_to_plot)

    # Initialiser l'objet de calcul des implied vol
    impliedVolCalculator.k_values = strikes

    # Calculer les implied vol avec la méthode VanillaImpliedVol pour les strikes spécifiques
    heston_iv = []
    selected_strikes = []
    selected_market_iv = []

    for i in indices_to_plot:
        impliedVolCalculator.T = maturities[i]
        heston_iv.append(impliedVolCalculator.VanillaImpliedVol()[i])
        selected_strikes.append(strikes[i])
        selected_market_iv.append(market_iv[i])

    # Créer le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(selected_strikes, selected_market_iv, label="Market IV", marker='o', linestyle='--')
    plt.plot(selected_strikes, heston_iv, label="Heston IV", marker='x', linestyle='-')

    # Ajouter des titres et légendes
    plt.title("Comparison of Implied Volatilities (Selected Strikes)")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid()

    # Afficher le graphique
    plt.show()

# Exemple d'appel avec les données et l'objet
df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data("AMZN")
impliedVolCalculator = ImpliedVolCalculator([], 0)  # Initialisation avec des valeurs temporaires
plot_implied_vol_comparison_subset(df, spot_price, impliedVolCalculator)



df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data("AMZN")
impliedVolCalculator = ImpliedVolCalculator([], 0)  # Initialisation avec des valeurs temporaires
plot_implied_vol_comparison(df, spot_price, impliedVolCalculator)











# # Calcul des implied volatilities pour différentes valeurs de T2
# IV_T2_2 = [FSImpliedVol(k, T2=2.0) for k in k_values]
# IV_T2_1_5 = [FSImpliedVol(k, T2=1.5) for k in k_values]
# IV_T2_1_25 = [FSImpliedVol(k, T2=1.25) for k in k_values]
#
# # Tracé des courbes
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, IV_T2_2, label='T2 = 2.0', linewidth=2)
# plt.plot(k_values, IV_T2_1_5, label='T2 = 1.5', linewidth=2)
# plt.plot(k_values, IV_T2_1_25, label='T2 = 1.25', linewidth=2)
#
# # Mise en forme du graphique
# plt.xlabel('Strike (k)')
# plt.ylabel('Implied Volatility (sigma)')
# plt.title('Implied Volatility vs Strike (k) for Different T2')
# plt.grid(True)
# plt.legend()
# plt.show()






