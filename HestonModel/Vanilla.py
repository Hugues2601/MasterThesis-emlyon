from HestonModel.HestonModelSuperClass import HestonModel
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
                    1 - torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * self.T)) / (
                        1 - g * torch.exp(-torch.sqrt(b ** 2 - 2 * self.sigma ** 2 * a) * self.T))

        cf = torch.exp(C * self.theta + D * self.v0 + 1j * phi * torch.log(self.S0 * torch.exp(self.r * self.T)))

        return cf

    def heston_price(self):
        P1, P2 = self._compute_integrals()
        price = self.S0 * P1 - self.K * torch.exp(-self.r * self.T) * P2
        return price

    def _compute_theta(self):
        if self.T.grad is not None:
            self.T.grad.zero_()

        price = self.heston_price()
        price.backward()
        greek = self.T.grad.item()
        return greek
