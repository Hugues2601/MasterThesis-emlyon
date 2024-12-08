import torch


def heston_cf(u, t, tau, hestonParams):

    kappa, theta, sigma, rho, v0 = hestonParams

    d = torch.sqrt((kappa - rho * sigma * u) * (kappa - rho * sigma * u) + u * (1. - u) * sigma * sigma)

    Gamma = (kappa - rho * sigma * u - d) / (kappa - rho * sigma * u + d)

    A = ((kappa * theta) / (sigma * sigma)) * (
                (kappa - rho * sigma * u - d) * tau - 2. * torch.log((1. - Gamma * torch.exp(-d * tau)) / (1. - Gamma)))

    B = (1. / (sigma * sigma)) * (kappa - rho * sigma * u - d) * (
                (1. - torch.exp(-d * tau)) / (1. - Gamma * torch.exp(-d * tau)))

    BBb = ((sigma * sigma) / (4. * kappa)) * (1. - torch.exp(-kappa * t))

    cf_val = torch.exp(A
                       + (B / (1. - 2. * BBb * B)) * (v0 * torch.exp(-kappa * t))
                       - ((2. * kappa * theta) / (sigma * sigma)) * torch.log(1. - 2. * BBb * B))
    return cf_val


def heston_price(S0, K, t, tau, hestonParams, precision=20.0, N=2000):

    kappa, theta, sigma, rho, v0 = hestonParams
    alpha = 0.5
    k = torch.log(K)

    pp = heston_cf(torch.tensor(1.0, dtype=torch.complex64), t, tau, hestonParams).real

    w = torch.linspace(-precision, precision, N, dtype=torch.float32)
    dw = w[1] - w[0]

    u = alpha + 1j * w
    cf_val = heston_cf(u, t, tau, hestonParams)

    denom = (u * (1. - alpha - 1j * w))
    intPhi_val = torch.exp(-k * u) * cf_val / denom
    intPhi_val_real = intPhi_val.real

    integral = torch.trapz(intPhi_val_real, w)

    price = S0*(pp - K / (2. * torch.pi) * integral)

    return price.real


kappa = torch.tensor(2.1, dtype=torch.float64)
theta = torch.tensor(0.03, dtype=torch.float64)
sigma = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
rho   = torch.tensor(-0.2, dtype=torch.float64)
v0    = torch.tensor(0.05, dtype=torch.float64)
S0 = torch.tensor(100, dtype=torch.float64, requires_grad=True)

hestonParams = (kappa, theta, sigma, rho, v0)

k    = torch.tensor(0.7, dtype=torch.float64)
t    = torch.tensor(1, dtype=torch.float64)
tau  = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
precision = 20.0

price = heston_price(S0, k, t, tau, hestonParams, precision=precision)

price.backward()

print("dPrice/dS0:", S0.grad.item())
print("dPrice/dsigma:", sigma.grad.item())
print("dPrice/dtau:", tau.grad.item())
print("Le prix de l'option call forward selon le modèle de Heston est:", price.item())