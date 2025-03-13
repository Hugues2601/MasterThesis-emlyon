import torch
import numpy as np
import matplotlib.pyplot as plt

from BlackScholes.VanillaBlackScholes import VanillaBlackScholes
from HestonModel.Vanilla import VanillaHestonPrice
from config import CONFIG


class ImpliedVolCalculatorVanilla:
    def __init__(self, S0, k_values, T, r, kappa, v0, theta, sigma, rho):
        self.S0 = S0
        self.k_values = k_values
        self.T = T
        self.r = r
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def VanillaImpliedVol(self):
        # Conversion des listes en tenseurs GPU
        k_values_tensor = torch.tensor(self.k_values, device=CONFIG.device, dtype=torch.float64)
        T_tensor = torch.tensor(self.T, device=CONFIG.device, dtype=torch.float64)

        # Initialisation des paramètres optimisables (theta utilisé pour sigma implicite)
        theta = torch.tensor([0.2] * len(self.k_values), device=CONFIG.device, requires_grad=True)

        # Optimiseur Adam
        optimizer = torch.optim.Adam([theta], lr=0.1)

        for step in range(500):
            sigma = torch.exp(theta)  # On impose une positivité à sigma avec l'exponentielle

            # Calcul du prix avec Heston
            VanillaHeston = VanillaHestonPrice(
                S0=self.S0, K=k_values_tensor, T=T_tensor, r=self.r,
                kappa=self.kappa, v0=self.v0, theta=self.theta, sigma=self.sigma, rho=self.rho, type="put"
            ).heston_price()

            # Calcul du prix avec Black-Scholes (sigma optimisé)
            VanillaBS = VanillaBlackScholes(
                S0=self.S0, K=k_values_tensor, T=T_tensor, r=self.r, sigma=sigma, type="put"
            ).price()

            # Calcul de la loss
            loss = torch.mean((VanillaHeston - VanillaBS) ** 2)

            # Backpropagation et mise à jour des paramètres
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.exp(theta).detach().cpu().numpy()

    def plot_IV_smile(self, T_values=None, num_strikes=100):
        """
        Génère et affiche le graphique de l'IV en fonction des strikes pour différentes maturités.

        :param T_values: Liste des maturités T à tester
        :param num_strikes: Nombre de strikes à générer (défaut : 500)
        """
        k_values = np.linspace(self.S0 * 0.4, self.S0*1.6, num_strikes).tolist()

        if T_values is None:
            T_values = [.4, .8, 1.0, 1.5, 2, 3]

        plt.figure(figsize=(12, 8))
        plt.gca().set_facecolor('#e6e6e6')
        plt.gca().grid(True, color='white', linestyle='--', linewidth=0.7, alpha=0.8)

        for T in T_values:
            IV_T = ImpliedVolCalculatorVanilla(
                S0=self.S0,
                k_values=k_values,
                T=[T]*len(k_values),
                r=self.r,
                kappa=self.kappa,
                v0=self.v0,
                theta=self.theta,
                sigma=self.sigma,
                rho=self.rho
            ).VanillaImpliedVol()

            plt.plot(k_values, IV_T, label=f'T = {T}', linewidth=2)

        # Personnalisation du graphique
        plt.xlabel('Strike (K)', fontsize=14)
        plt.ylabel('Implied Volatility (IV)', fontsize=14)
        plt.title('Vanilla Heston Calls - Implied Volatility for Different Maturities (T)', fontsize=14)
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#4f4f4f')
        plt.gca().spines['bottom'].set_color('#4f4f4f')
        plt.gca().tick_params(colors='black')

        plt.show()



# # Définition des paramètres
# S0 = 100.0  # Prix initial du sous-jacent
# k_values = [] # Liste des strikes entre 80 et 120
# T = [] # Maturité
# r = 0.05  # Taux sans risque
# kappa = 3.0  # Paramètre Heston
# v0 = 0.04  # Volatilité initiale sous Heston
# theta = 0.04  # Moyenne de la volatilité sous Heston
# sigma = 0.2  # Volatilité du processus de variance sous Heston
# rho = -0.7  # Corrélation entre le sous-jacent et la variance
#
# # Instanciation de la classe
# iv_calculator = ImpliedVolCalculatorVanilla(S0, k_values, T, r, kappa, v0, theta, sigma, rho).plot_IV_smile()
#
# # Appel de la méthode pour tracer le smile de volatilité
# print(iv_calculator.VanillaImpliedVol())