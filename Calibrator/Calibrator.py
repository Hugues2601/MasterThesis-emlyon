import torch
from config import CONFIG
from Calibrator.HMCalibration import heston_price
import numpy as np


def calibrate(S0, market_prices, K, T, r, initial_guess, max_epochs=10000, lr=0.001, loss_threshold=1e-4):
    device = CONFIG.device
    S0 = torch.tensor(S0, dtype=torch.float64, device=device)
    K = torch.tensor(K, dtype=torch.float64, device=device)
    T = torch.tensor(T, dtype=torch.float64, device=device)
    market_prices = torch.tensor(market_prices, dtype=torch.float64, device=device)
    r = torch.tensor(r, dtype=torch.float64, device=device)

    raw_kappa = torch.tensor(initial_guess['kappa'], dtype=torch.float64, device=device, requires_grad=True)
    raw_v0 = torch.tensor(initial_guess['v0'], dtype=torch.float64, device=device, requires_grad=True)
    raw_theta = torch.tensor(initial_guess['theta'], dtype=torch.float64, device=device, requires_grad=True)
    raw_sigma = torch.tensor(np.log(initial_guess['sigma']), dtype=torch.float64, device=device, requires_grad=True)
    raw_rho = torch.tensor(initial_guess['rho'], dtype=torch.float64, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([raw_kappa, raw_v0, raw_theta, raw_sigma, raw_rho], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1000, verbose=True)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        kappa = 5*torch.sigmoid(raw_kappa)
        sigma = 0.6*torch.sigmoid(raw_sigma)
        v0 = torch.sigmoid(raw_v0)
        theta = torch.sigmoid(raw_theta)
        rho = -torch.sigmoid(raw_rho)

        model_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
        loss = torch.sqrt(torch.mean((model_prices - market_prices) ** 2))
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

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

def calibrate_data(ticker):
    pass