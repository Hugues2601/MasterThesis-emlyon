import torch

from DataRetriever import get_yfinance_data
from config import CONFIG
from Calibrator.HMCalibration import heston_price
import numpy as np
from HestonModel.Vanilla import VanillaHestonPrice
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Calibrator:
    def __init__(self, S0, market_prices, K, T, r):
        self.S0 = S0
        self.market_prices = market_prices
        self.K = K
        self.T = T
        self.r = r
        self.initial_guess = CONFIG.initial_guess

    def calibrate(self, max_epochs=10000, lr=0.001, loss_threshold=1e-4):
        device = CONFIG.device
        print(f"Nb of options: {len(self.market_prices)}")
        calls_mean = sum(self.market_prices) / len(self.market_prices)
        print(f"mean price of options: {calls_mean}")
        print(f"Standard Deviation of Options Prices: {np.std(self.market_prices)}")
        print(f"spot price: {self.S0}")

        S0 = torch.tensor(self.S0, dtype=torch.float32, device=device)
        K = torch.tensor(self.K, dtype=torch.float32, device=device)
        T = torch.tensor(self.T, dtype=torch.float32, device=device)
        market_prices = torch.tensor(self.market_prices, dtype=torch.float32, device=device)
        r = torch.tensor(self.r, dtype=torch.float32, device=device)

        raw_kappa = torch.tensor(self.initial_guess['kappa'], dtype=torch.float32, device=device, requires_grad=True)
        raw_v0 = torch.tensor(self.initial_guess['v0'], dtype=torch.float32, device=device, requires_grad=True)
        raw_theta = torch.tensor(self.initial_guess['theta'], dtype=torch.float32, device=device, requires_grad=True)
        raw_sigma = torch.tensor(np.log(self.initial_guess['sigma']), dtype=torch.float32, device=device, requires_grad=True)
        raw_rho = torch.tensor(self.initial_guess['rho'], dtype=torch.float32, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([raw_kappa, raw_v0, raw_theta, raw_sigma, raw_rho], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1000)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            kappa = 0.5 + 7.5 * torch.sigmoid(raw_kappa)

            v0 = 0.01 + 0.49 * torch.sigmoid(raw_v0)  # [0.01, 0.5]
            theta = 0.01 + 0.49 * torch.sigmoid(raw_theta)  # [0.01, 0.5]
            sigma = 0.05 + 0.95 * torch.sigmoid(raw_sigma)

            # 🔥 Correction finale de theta pour assurer Feller
  # Toujours positif
            theta = torch.log1p(theta + (sigma ** 2) / (2 * kappa) + 1e-5)  # Transformation lisse

            rho = -0.9 + 1.8 * torch.sigmoid(raw_rho)

            model_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
            loss = torch.sqrt(torch.mean((model_prices - market_prices) ** 2))

            # 🔥 Ajout d’une pénalisation exponentielle pour renforcer Feller
            feller_penalty = torch.exp(-10 * (2 * kappa * theta - sigma ** 2))  # Évite de frôler la limite
            total_loss = loss + feller_penalty
            total_loss.backward()

            optimizer.step()
            scheduler.step(loss)
            current_lr = scheduler.optimizer.param_groups[0]['lr']

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {current_lr}, Feller Penalty: {feller_penalty.item()}")

            if loss.item() < loss_threshold:
                print(f"Converged at epoch {epoch} with loss {loss.item()}")
                break

        calibrated_params = {
            'kappa': kappa.item(),
            'v0': v0.item(),
            'theta': theta.item(),
            'sigma': sigma.item(),
            'rho': rho.item()
        }

        print(f"Final Feller Condition: {2 * calibrated_params['kappa'] * calibrated_params['theta']} > {calibrated_params['sigma']**2}")
        return calibrated_params



def plot_heston_vs_market(S0, lastPrice, strike, timetomaturity, r, calibrated_params):
    """
    Plots the market prices vs Heston model prices using calibrated parameters.

    Parameters:
    - S0: Spot price of the underlying asset.
    - lastPrice: List of observed market prices of the options.
    - strike: List of strike prices.
    - timetomaturity: List of times to maturity.
    - r: Risk-free interest rate.
    - calibrated_params: Calibrated parameters from the Heston model (kappa, v0, theta, sigma, rho).
    """

    kappa = float(calibrated_params["kappa"])
    v0 = float(calibrated_params["v0"])
    theta = float(calibrated_params["theta"])
    sigma = float(calibrated_params["sigma"])
    rho = float(calibrated_params["rho"])

    heston_prices = []

    for K, T in zip(strike, timetomaturity):
        model = VanillaHestonPrice(S0, K, T, r, kappa, v0, theta, sigma, rho, type="call")
        price = model.heston_price().item()
        heston_prices.append(price)

    # Calcul du R² global
    R2_global = r2_score(lastPrice, heston_prices)

    # Plot global
    plt.figure(figsize=(10, 6))
    plt.scatter(strike, lastPrice, color='black', label='Market Options Prices', s=12)
    plt.scatter(strike, heston_prices, color='#4682B4', marker='x', label='Heston Model Options Prices', s=12)
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('')
    plt.legend()
    plt.grid()

    # Affichage du R² global sur le plot
    plt.text(min(strike), max(lastPrice), f'R² = {R2_global:.4f}', fontsize=12,
             color='black', bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

    # Plot pour chaque maturité séparément
    unique_maturities = sorted(set(timetomaturity))

    for T in unique_maturities:
        plt.figure(figsize=(10, 6))
        indices = [i for i, t in enumerate(timetomaturity) if t == T]
        strikes_T = [strike[i] for i in indices]
        lastPrices_T = [lastPrice[i] for i in indices]
        hestonPrices_T = [heston_prices[i] for i in indices]

        # Calcul du R² par maturité
        R2_T = r2_score(lastPrices_T, hestonPrices_T)

        plt.scatter(strikes_T, lastPrices_T, color='black', label='Market Prices', s=10)
        plt.scatter(strikes_T, hestonPrices_T, color='gray', marker='x', label='Heston Model Prices', s=10)
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.title(f'Observed Options Prices vs Heston Model Prices (T={T:.2f})')
        plt.legend()
        plt.grid()

        # Affichage du R² spécifique à cette maturité
        plt.text(min(strikes_T), max(lastPrices_T), f'R² = {R2_T:.4f}', fontsize=12,
                 color='black', bbox=dict(facecolor='white', alpha=0.8))

        plt.show()




def plot_residuals_heston(
    model_class,
    S0,
    K_list,
    T_list,
    r,
    market_prices,
    kappa, v0, theta, sigma, rho,
    plot_type="strike"  # ou "maturity"
):
    model_residuals = []
    Ks, Ts = [], []

    for K, T, market_price in zip(K_list, T_list, market_prices):
        model = model_class(S0, K, T, r, kappa, v0, theta, sigma, rho, type="call")
        price = model.heston_price().item()
        residual = price - market_price

        model_residuals.append(residual)
        Ks.append(K)
        Ts.append(T)

    if plot_type == "strike":
        x = Ks
        xlabel = "Strike"
    elif plot_type == "maturity":
        x = Ts
        xlabel = "Time to Maturity (Years)"
    else:
        raise ValueError("plot_type must be 'strike' or 'maturity'")

    plt.figure(figsize=(10, 5))
    plt.scatter(x, model_residuals, alpha=0.6, color='black')  # points en noir
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("Residuals (Model Price - Market Price)")
    plt.grid(True)
    plt.show()



