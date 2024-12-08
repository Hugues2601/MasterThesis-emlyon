import torch


def heston_cf(u, t, tau, hestonParams):
    # hestonParams = (kappa, theta, sigma, rho, v0)
    # Toutes les variables doivent être des tenseurs torch, u peut être complexe : u = alpha + 1j*w
    kappa, theta, sigma, rho, v0 = hestonParams

    # Calcul de d(u)
    d = torch.sqrt((kappa - rho * sigma * u) * (kappa - rho * sigma * u) + u * (1. - u) * sigma * sigma)

    # LittleGamma(u)
    Gamma = (kappa - rho * sigma * u - d) / (kappa - rho * sigma * u + d)

    # BigA(u, tau)
    A = ((kappa * theta) / (sigma * sigma)) * (
                (kappa - rho * sigma * u - d) * tau - 2. * torch.log((1. - Gamma * torch.exp(-d * tau)) / (1. - Gamma)))

    # BigB(u, tau)
    B = (1. / (sigma * sigma)) * (kappa - rho * sigma * u - d) * (
                (1. - torch.exp(-d * tau)) / (1. - Gamma * torch.exp(-d * tau)))

    # LittleB(t)
    BBb = ((sigma * sigma) / (4. * kappa)) * (1. - torch.exp(-kappa * t))

    # cfHestonFwd(u, t, tau)
    cf_val = torch.exp(A
                       + (B / (1. - 2. * BBb * B)) * (v0 * torch.exp(-kappa * t))
                       - ((2. * kappa * theta) / (sigma * sigma)) * torch.log(1. - 2. * BBb * B))
    return cf_val


def heston_price(S0, K, t, tau, hestonParams, precision=20.0, N=2000):
    # On calcule le prix de l'option call forward sous Heston via l'intégration numérique
    # hestonForwardCallPrice(K, t, tau) = pp - K/(2π)*int_{-precision}^{precision} intPhi(w) dw
    # avec pp = cfHestonFwd(1, t, tau).real

    # Paramètres
    kappa, theta, sigma, rho, v0 = hestonParams
    alpha = 0.5
    k = torch.log(K)

    # pp = Re(cfHestonFwd(1., t, tau))
    pp = heston_cf(torch.tensor(1.0, dtype=torch.complex64), t, tau, hestonParams).real

    # Construction de la grille pour w
    w = torch.linspace(-precision, precision, N, dtype=torch.float32)
    dw = w[1] - w[0]

    # intPhi(w) = Re( e^{ -k*(alpha + i w) } * cfHestonFwd(alpha + i w) / ((alpha + i w)*(1 - alpha - i w)) )
    # On calcule cfHestonFwd(alpha + i w)
    u = alpha + 1j * w
    cf_val = heston_cf(u, t, tau, hestonParams)

    denom = (u * (1. - alpha - 1j * w))  # = (alpha + i w)*(1 - alpha - i w)
    intPhi_val = torch.exp(-k * u) * cf_val / denom
    # On ne garde que la partie réelle
    intPhi_val_real = intPhi_val.real

    # Intégration numérique par trapèzes sur w
    # trapz = sum((f(w_i) + f(w_{i+1}))/2 * dw)
    # Ici, torch.trapz existe, sinon on peut le coder :
    integral = torch.trapz(intPhi_val_real, w)

    # heston price
    # hestonForwardCallPrice = pp - K/(2π)*intPhi
    price = S0*(pp - K / (2. * torch.pi) * integral)

    return price.real


kappa = torch.tensor(2.0, dtype=torch.float64)
theta = torch.tensor(0.04, dtype=torch.float64)
sigma = torch.tensor(0.3, dtype=torch.float64)
rho   = torch.tensor(-0.7, dtype=torch.float64)
v0    = torch.tensor(0.04, dtype=torch.float64)
S0 = torch.tensor(100, dtype=torch.float64)

hestonParams = (kappa, theta, sigma, rho, v0)

# Autres paramètres
k    = torch.tensor(1, dtype=torch.float64)
t    = torch.tensor(1, dtype=torch.float64)
tau  = torch.tensor(1.0, dtype=torch.float64)
precision = 20.0

# Appel de la fonction
price = heston_price(S0, k, t, tau, hestonParams, precision=precision)

print("Le prix de l'option call forward selon le modèle de Heston est:", price.item())