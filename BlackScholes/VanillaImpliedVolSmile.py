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
        k_values_tensor = torch.tensor(self.k_values, device=CONFIG.device, dtype=torch.float32)
        T_tensor = torch.tensor(self.T, device=CONFIG.device, dtype=torch.float32)

        # Initialisation des paramètres optimisables (theta utilisé pour sigma implicite)
        theta = torch.tensor([0.2] * len(self.k_values), device=CONFIG.device, requires_grad=True)

        # Optimiseur Adam
        optimizer = torch.optim.Adam([theta], lr=0.1)

        for step in range(500):
            sigma = torch.exp(theta)  # On impose une positivité à sigma avec l'exponentielle

            # Calcul du prix avec Heston
            VanillaHeston = VanillaHestonPrice(
                S0=self.S0, K=k_values_tensor, T=T_tensor, r=self.r,
                kappa=self.kappa, v0=self.v0, theta=self.theta, sigma=self.sigma, rho=self.rho, type="call"
            ).heston_price()

            # Calcul du prix avec Black-Scholes (sigma optimisé)
            VanillaBS = VanillaBlackScholes(
                S0=self.S0, K=k_values_tensor, T=T_tensor, r=self.r, sigma=sigma, type="call"
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
        k_values = np.linspace(4500, 7200, num_strikes).tolist()

        if T_values is None:
            T_values = [.44, .60, .79, 1.05, 1.29, 1.83,]

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


def plot_implied_volatility(strike, implied_volatility, timetomaturity):
    """
    Plots implied volatility against strike price for different maturities.

    Parameters:
    - strike: List or array of strike prices.
    - implied_volatility: List or array of implied volatilities.
    - timetomaturity: List or array of time to maturity (same length as strike and implied_volatility).

    Returns:
    - A plot of implied volatility vs. strike price, grouped by maturity.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))

    # Convert to numpy array for easier processing
    strike = np.array(strike)
    implied_volatility = np.array(implied_volatility)
    timetomaturity = np.array(timetomaturity)

    # Get unique maturities and plot each one
    unique_maturities = np.unique(timetomaturity)
    for maturity in sorted(unique_maturities):
        mask = timetomaturity == maturity
        plt.plot(strike[mask], implied_volatility[mask], label=f"T={maturity:.2f}")

    # Labels and title
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Market Implied Volatility Smile")
    plt.legend(title="Maturity (Years)", loc="upper right")
    plt.grid(True)

    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_comparative_IV_smile(
    S0,
    strike_market,
    iv_market,
    timetomarket,
    r,
    kappa, v0, theta, sigma, rho,
    selected_maturities=[.44, .60, .79, 1.05, 1.29, 1.83],
    num_strikes=100
):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    strike_range = np.linspace(4500, 7200, num_strikes).tolist()

    # Palette pastel personnalisée (peut être étendue si besoin)
    pastel_colors = [
        "black",
        "#8fbcbb",
        "#a3be8c",
        "#d08770",
        "#e09ec7",
        "#b48ead"
    ]

    for idx, T in enumerate(selected_maturities):
        color = pastel_colors[idx % len(pastel_colors)]

        # ---- IV du modèle Heston ----
        model_IV = ImpliedVolCalculatorVanilla(
            S0=S0,
            k_values=strike_range,
            T=[T] * len(strike_range),
            r=r,
            kappa=kappa,
            v0=v0,
            theta=theta,
            sigma=sigma,
            rho=rho
        ).VanillaImpliedVol()

        # ---- IV du marché (filtrée sur la même maturité) ----
        mask = (np.isclose(timetomarket, T, atol=0.05)) & (np.array(strike_market) >= 4500)
        strike_T = np.array(strike_market)[mask]
        iv_T = np.array(iv_market)[mask]

        # ---- Tracé ----
        plt.plot(strike_range, model_IV, label=f"Heston — T={T:.2f}", color=color, linewidth=2)
        if len(strike_T) > 0:
            sorted_indices = np.argsort(strike_T)
            plt.plot(
                strike_T[sorted_indices],
                iv_T[sorted_indices],
                linestyle='--',
                color=color,
                linewidth=1.5,
                label=f"Market — T={T:.2f}"
            )
        else:
            print(f"⚠️ No market data for T ≈ {T:.2f}")

    plt.xlabel("Strike", fontsize=13)
    plt.ylabel("Implied Volatility", fontsize=13)
    plt.legend(fontsize=10, loc='upper right', frameon=True)
    plt.tight_layout()
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