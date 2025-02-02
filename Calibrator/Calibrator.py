import torch

from DataRetriever import get_yfinance_data
from config import CONFIG
from Calibrator.HMCalibration import heston_price
import numpy as np

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
        print(f"spot price: {self.S0}")
        S0 = torch.tensor(self.S0, dtype=torch.float64, device=device)
        K = torch.tensor(self.K, dtype=torch.float64, device=device)
        T = torch.tensor(self.T, dtype=torch.float64, device=device)
        market_prices = torch.tensor(self.market_prices, dtype=torch.float64, device=device)
        r = torch.tensor(self.r, dtype=torch.float64, device=device)

        raw_kappa = torch.tensor(self.initial_guess['kappa'], dtype=torch.float64, device=device, requires_grad=True)
        raw_v0 = torch.tensor(self.initial_guess['v0'], dtype=torch.float64, device=device, requires_grad=True)
        raw_theta = torch.tensor(self.initial_guess['theta'], dtype=torch.float64, device=device, requires_grad=True)
        raw_sigma = torch.tensor(np.log(self.initial_guess['sigma']), dtype=torch.float64, device=device, requires_grad=True)
        raw_rho = torch.tensor(self.initial_guess['rho'], dtype=torch.float64, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([raw_kappa, raw_v0, raw_theta, raw_sigma, raw_rho], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1000)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            kappa = raw_kappa
            sigma = torch.sigmoid(raw_sigma)
            v0 = raw_v0
            theta = raw_theta
            rho = -torch.sigmoid(raw_rho)

            model_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
            loss = torch.sqrt(torch.mean((model_prices - market_prices) ** 2))
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            current_lr = scheduler.optimizer.param_groups[0]['lr']

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}, LR: {current_lr}")

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
        return calibrated_params


