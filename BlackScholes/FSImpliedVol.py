import torch
from BlackScholes.ForwardStartBlackScholes import FSBlackScholes
from HestonModel.ForwardStart import ForwardStart
from config import CONFIG
import numpy as np
import matplotlib.pyplot as plt
from BlackScholes.VanillaBlackScholes import VanillaBlackScholes, implied_vol
from HestonModel.Vanilla import VanillaHestonPrice
import torch
from mpl_toolkits.mplot3d import Axes3D  # pour les graphs 3D
from matplotlib import cm  # pour les colormaps

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
        k_values_tensor = torch.tensor(self.k_values, device=CONFIG.device, dtype=torch.float32)
        T2_tensor = torch.tensor(self.T2, device=CONFIG.device, dtype=torch.float32)

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

    def plot_IV_smile(self, num_strikes=100):
        """
        Trace les implied volatilities de Forward Start options pour différents couples (T0, T1, T2),
        avec couleurs pastel et courbe T0=0.5 en noir pointillé.
        """

        import matplotlib.pyplot as plt
        import numpy as np

        k_values = np.linspace(0.6, 1.4, num_strikes).tolist()

        # Triplets dans l'ordre souhaité (les T1=0.75 sont à la fin)
        maturity_triplets = [
            (0.0, 0.5, 1.0),  # turquoise
            (0.0, 1.0, 2.0),  # orange
            (0.5, 1.0, 2.0),  # noir pointillé
            (0.0, 0.75, 1.5),  # vert
            (0.0, 0.75, 2.0),  # rose
            (0.0, 0.75, 2.5),  # lavande
        ]

        pastel_colors = [
            "#8fbcbb",  # turquoise
            "#d08770",  # orange
            "black",  # noir
            "#a3be8c",  # vert
            "#e09ec7",  # rose
            "#b48ead"  # lavande
        ]

        linestyles = [
            '-', '-', '--', '-', '-', '-'
        ]

        # Setup du plot
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.7)

        # Tracé des courbes
        for i, (T0, T1, T2) in enumerate(maturity_triplets):
            IV_T = ImpliedVolCalculatorFS(
                S0=self.S0,
                k_values=k_values,
                T0=T0,
                T1=T1,
                T2=[T2] * len(k_values),
                r=self.r,
                kappa=self.kappa,
                v0=self.v0,
                theta=self.theta,
                sigma=self.sigma,
                rho=self.rho
            ).FSImpliedVol()

            label = f"T0={T0}, T1={T1}, T2={T2}"
            ax.plot(
                k_values,
                IV_T,
                label=label,
                color=pastel_colors[i],
                linestyle=linestyles[i],
                linewidth=2
            )

        # Axes et légende
        ax.set_xlabel('Strike (k)', fontsize=14)
        ax.set_ylabel('Implied Volatility (IV)', fontsize=14)
        ax.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', edgecolor='gray')

        # Bordures noires autour du graphique
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        plt.tight_layout()
        plt.show()

    def plot_IV_surface(self, T_values=None, num_strikes=100):
        """
        Affiche la surface de volatilité implicite (IV) en fonction du strike (k) et de la maturité T2.

        :param T_values: Liste des maturités T2 à tester
        :param num_strikes: Nombre de strikes à générer (défaut : 100)
        """
        k_values = np.linspace(0.4, 1.6, num_strikes).tolist()

        if T_values is None:
            T_values = [1.4, 1.8, 2.0, 2.5, 3.0, 4.0]

        # Préparer les matrices pour la surface
        K_mesh, T_mesh = np.meshgrid(k_values, T_values)
        IV_surface = np.zeros_like(K_mesh)

        for i, T in enumerate(T_values):
            IV_T = ImpliedVolCalculatorFS(
                S0=self.S0,
                k_values=k_values,
                T0=self.T0,
                T1=self.T1,
                T2=[T] * len(k_values),
                r=self.r,
                kappa=self.kappa,
                v0=self.v0,
                theta=self.theta,
                sigma=self.sigma,
                rho=self.rho
            ).FSImpliedVol()

            IV_surface[i, :] = IV_T

        # Tracer la surface
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(K_mesh, T_mesh, IV_surface, cmap=cm.viridis, edgecolor='none', alpha=0.9)

        ax.set_xlabel('Strike (k)', fontsize=12)
        ax.set_ylabel('Maturity $T_2$', fontsize=12)
        ax.set_zlabel('Implied Volatility (IV)', fontsize=12)
        ax.set_title('Implied Volatility Surface - Heston Forward Start Options', fontsize=14)

        fig.colorbar(surf, shrink=0.5, aspect=10, label='Implied Vol')

        plt.tight_layout()
        plt.show()
