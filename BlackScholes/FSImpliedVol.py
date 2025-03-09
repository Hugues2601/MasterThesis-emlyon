import torch
from BlackScholes.ForwardStartBlackScholes import FSBlackScholes
from HestonModel.ForwardStart import ForwardStart
from config import CONFIG
import numpy as np
import matplotlib.pyplot as plt
from BlackScholes.VanillaBlackScholes import VanillaBlackScholes, implied_vol
from HestonModel.Vanilla import VanillaHestonPrice

import torch

class ImpliedVolCalculatorFS:
    def __init__(self, S0, k_values, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        self.S0 = S0
        self.k_values = k_values
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2
        self.r = r
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def FSImpliedVol(self):
        # Conversion des listes en tenseurs GPU
        k_values_tensor = torch.tensor(self.k_values, device=CONFIG.device, dtype=torch.float64)
        T2_tensor = torch.tensor(self.T2, device=CONFIG.device, dtype=torch.float64)

        # Initialisation des paramètres optimisables (theta utilisé pour sigma implicite)
        theta = torch.tensor([0.2] * len(self.k_values), device=CONFIG.device, requires_grad=True)

        # Optimiseur Adam
        optimizer = torch.optim.Adam([theta], lr=0.1)

        for step in range(500):
            sigma = torch.exp(theta)  # On impose une positivité à sigma avec l'exponentielle

            # Calcul du prix avec Heston
            FSHeston = ForwardStart(
                S0=self.S0, k=k_values_tensor, T0=self.T0, T1=self.T1, T2=T2_tensor, r=self.r,
                kappa=self.kappa, v0=self.v0, theta=self.theta, sigma=self.sigma, rho=self.rho
            ).heston_price()

            # Calcul du prix avec Black-Scholes (sigma optimisé)
            FSBS = FSBlackScholes(
                S0=self.S0, k=k_values_tensor, T1=self.T1, T2=T2_tensor, r=self.r, sigma=sigma
            ).price()

            # Calcul de la loss
            loss = torch.mean((FSHeston - FSBS) ** 2)

            # Backpropagation et mise à jour des paramètres
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return torch.exp(theta).detach().cpu().numpy()

    def plot_IV_smile(self, T_values=None, num_strikes=500):
        """
        Génère et affiche le graphique de l'IV en fonction des strikes pour différentes maturités.

        :param T_values: Liste des maturités T2 à tester
        :param num_strikes: Nombre de strikes à générer (défaut : 500)
        """
        k_values = np.linspace(0.2, 1.9, num_strikes).tolist()

        if T_values is None:
            T_values = [1.4, 1.6, 1.8, 2, 2.5, 3]


        plt.figure(figsize=(12, 8))

        # Définir un fond gris
        plt.gca().set_facecolor('#e6e6e6')
        plt.gca().grid(True, color='white', linestyle='--', linewidth=0.7, alpha=0.8)

        for T in T_values:
            IV_T = ImpliedVolCalculatorFS(
                S0=self.S0,
                k_values=k_values,
                T0=self.T0,
                T1=self.T1,
                T2=[T] * len(k_values),  # Liste de T2 de même longueur que k_values
                r=self.r,
                kappa=self.kappa,
                v0=self.v0,
                theta=self.theta,
                sigma=self.sigma,
                rho=self.rho
            ).FSImpliedVol()

            plt.plot(k_values, IV_T, label=f'T2-T1 = {T-self.T1}', linewidth=2)

        # Personnalisation du graphique
        plt.xlabel('Strike (k)', fontsize=14)
        plt.ylabel('Implied Volatility (IV)', fontsize=14)
        plt.title('Forward Start Heston Calls - Implied Volatility for Different Maturities (T)', fontsize=14)
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        # Amélioration des bordures du graphique
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#4f4f4f')
        plt.gca().spines['bottom'].set_color('#4f4f4f')
        plt.gca().tick_params(colors='black')

        # Affichage
        plt.show()
