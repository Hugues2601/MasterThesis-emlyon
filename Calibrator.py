import torch
from config import CONFIG
from HestonModel.Vanilla import heston_price


def calibrate(S0, market_prices, K, T, r, initial_guess, max_epochs=10000, lr=0.01, loss_threshold=3):
    # Ensure that all inputs are torch tensors on the GPU
    S0 = torch.tensor(S0, dtype=torch.float64, device=CONFIG.device)
    K = torch.tensor(K, dtype=torch.float64, device=CONFIG.device)
    T = torch.tensor(T, dtype=torch.float64, device=CONFIG.device)
    market_prices = torch.tensor(market_prices, dtype=torch.float64, device=CONFIG.device)
    r = torch.tensor(r, dtype=torch.float64, device=CONFIG.device)

    # Initialize parameters to optimize
    kappa = torch.tensor(initial_guess['kappa'], dtype=torch.float64, device=CONFIG.device, requires_grad=True)
    v0 = torch.tensor(initial_guess['v0'], dtype=torch.float64, device=CONFIG.device, requires_grad=True)
    theta = torch.tensor(initial_guess['theta'], dtype=torch.float64, device=CONFIG.device, requires_grad=True)
    sigma = torch.tensor(initial_guess['sigma'], dtype=torch.float64, device=CONFIG.device, requires_grad=True)
    rho = torch.tensor(initial_guess['rho'], dtype=torch.float64, device=CONFIG.device, requires_grad=True)

    # Set up optimizer
    optimizer = torch.optim.Adam([kappa, v0, theta, sigma, rho], lr=lr)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Compute model prices
        model_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
        # Compute loss (Mean Squared Error)
        loss = torch.sqrt(torch.mean((model_prices - market_prices) ** 2))
        # Backpropagation
        loss.backward()
        # Update parameters
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Check for convergence
        if loss.item() < loss_threshold:
            print(f"Converged at epoch {epoch} with loss {loss.item()}")
            break

    # Return the calibrated parameters
    calibrated_params = {
        'kappa': kappa.item(),
        'v0': v0.item(),
        'theta': theta.item(),
        'sigma': sigma.item(),
        'rho': rho.item()
    }
    return calibrated_params