import torch
from config import CONFIG
from Calibrator.HMCalibration import heston_price
import numpy as np


def calibrate(S0, market_prices, K, T, r, initial_guess, max_epochs=10000, lr=0.01, loss_threshold=1e-4):
    # Ensure all inputs are torch tensors on the GPU (with double precision)
    device = CONFIG.device
    S0 = torch.tensor(S0, dtype=torch.float64, device=device)
    K = torch.tensor(K, dtype=torch.float64, device=device)
    T = torch.tensor(T, dtype=torch.float64, device=device)
    market_prices = torch.tensor(market_prices, dtype=torch.float64, device=device)
    r = torch.tensor(r, dtype=torch.float64, device=device)

    # Paramètres Heston à optimiser
    kappa = torch.tensor(initial_guess['kappa'], dtype=torch.float64, device=device, requires_grad=True)
    v0 = torch.tensor(initial_guess['v0'], dtype=torch.float64, device=device, requires_grad=True)
    theta = torch.tensor(initial_guess['theta'], dtype=torch.float64, device=device, requires_grad=True)
    # On paramètre sigma sous forme exponentielle
    raw_sigma = torch.tensor(np.log(initial_guess['sigma']), dtype=torch.float64, device=device, requires_grad=True)
    raw_rho = torch.tensor(initial_guess['rho'], dtype=torch.float64, device=device, requires_grad=True)

    # Optimiseur
    optimizer = torch.optim.Adam([kappa, v0, theta, raw_sigma, raw_rho], lr=lr)
    # Scheduler pour réduire le taux d'apprentissage si la perte stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, verbose=True)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Reconstruction de sigma
        sigma = torch.exp(raw_sigma)
        rho = -torch.sigmoid(raw_rho)

        # Calcul des prix modèles
        model_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
        # Loss = Root Mean Squared Error
        loss = torch.sqrt(torch.mean((model_prices - market_prices) ** 2))

        # Rétro-propagation
        loss.backward()
        optimizer.step()

        # Ajustement du lr selon la perte
        scheduler.step(loss)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Test de convergence
        if loss.item() < loss_threshold:
            print(f"Converged at epoch {epoch} with loss {loss.item()}")
            break

    # Paramètres calibrés finaux
    calibrated_params = {
        'kappa': kappa.item(),
        'v0': v0.item(),
        'theta': theta.item(),
        'sigma': sigma.item(),  # Ici, sigma est toujours > 0
        'rho': rho.item()
    }
    return calibrated_params


def calibrate_data(ticker):
    pass