from HestonModelSuperClass import HestonModel
import torch
from config import *

""" ----------------------- Characteristic Function --------------------"""

class VanillaHestonPrice(HestonModel):
    def __init__(self, S0, K, T, r, kappa, v0, theta, sigma, rho):
        super().__init__(S0, K, T, r, kappa, v0, theta, sigma, rho)

    def _heston_cf(self, phi):
        # Ensure that phi is a torch tensor on the GPU
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.complex128, device=CONFIG.device)
        else:
            phi = phi.to(CONFIG.device).type(torch.complex128)


        S0 = self.S0.to(CONFIG.device).type(torch.float64)
        T = self.T.to(CONFIG.device).type(torch.float64)
        r = self.r.to(CONFIG.device).type(torch.float64)

        a = -0.5 * phi ** 2 - 0.5 * 1j * phi
        b = self.kappa - self.rho * self.sigma * 1j * phi

        g = ((b - torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2) / ((b + torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2)

        C = self.kappa * (((b - torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2) * self.T - 2 / self.sigma ** 2 * torch.log(
            (1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * T)) / (1 - g)))

        D = ((b - torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a)) / self.sigma ** 2) * (
                    1 - torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * T)) / (
                        1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * T))

        cf = torch.exp(C * self.theta + D * self.v0 + 1j * phi * torch.log(self.S0 * torch.exp(r * T)))

        return cf


    """ ---------------------- Pricing Formula ----------------------"""

    def heston_price(self):

        # Vérification de la taille de K et T
        assert self.K.dim() == 1, "K doit être un tenseur 1D"
        assert self.T.dim() == 1, "T doit être un tenseur 1D"
        assert len(self.K) == len(self.T), "K et T doivent avoir la même taille"


        umax = 50
        n = 100
        if n % 2 == 0:
            n += 1

        phi_values = torch.linspace(1e-5, umax, n, device=self.K.device)
        du = (umax - 1e-5) / (n - 1)

        phi_values = phi_values.unsqueeze(1).repeat(1, len(self.K))

        factor1 = torch.exp(-1j * phi_values * torch.log(self.K))
        denominator = 1j * phi_values


        cf1 = self._heston_cf(phi_values - 1j) / self._heston_cf(-1j)
        temp1 = factor1 * cf1 / denominator
        integrand_P1_values = 1 / torch.pi * torch.real(temp1)


        cf2 = self._heston_cf(phi_values)
        temp2 = factor1 * cf2 / denominator
        integrand_P2_values = 1 / torch.pi * torch.real(temp2)

        weights = torch.ones(n, device=self.K.device)
        weights[1:-1:2] = 4
        weights[2:-2:2] = 2
        weights *= du / 3
        weights = weights.unsqueeze(1).repeat(1, len(self.K))

        integral_P1 = torch.sum(weights * integrand_P1_values, dim=0)
        integral_P2 = torch.sum(weights * integrand_P2_values, dim=0)

        P1 = torch.tensor(0.5, device=self.K.device) + integral_P1
        P2 = torch.tensor(0.5, device=self.K.device) + integral_P2
        price = self.S0 * P1 - torch.exp(-self.r * self.T) * self.K * P2
        return price

price = VanillaHestonPrice(100.0, 100.0, 2.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7)
price2 = price.heston_price()
print(price2)