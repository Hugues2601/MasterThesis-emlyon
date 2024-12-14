from abc import ABC, abstractmethod
import torch
from config import CONFIG
import matplotlib.pyplot as plt

class HestonModel(ABC):
    def __init__(self, S0, K, T, r, kappa, v0, theta, sigma, rho):
        self.S0 = torch.tensor(S0, device=CONFIG.device, requires_grad=True)
        self.K = torch.tensor([K], device=CONFIG.device)
        self.T = torch.tensor([T], device=CONFIG.device, requires_grad=True)
        self.r = torch.tensor(r, device=CONFIG.device, requires_grad=True)
        self.kappa = torch.tensor(kappa, device=CONFIG.device)
        self.v0 = torch.tensor(v0, device=CONFIG.device)
        self.theta = torch.tensor(theta, device=CONFIG.device)
        self.sigma = torch.tensor(sigma, device=CONFIG.device, requires_grad=True)
        self.rho = torch.tensor(rho, device=CONFIG.device)

    @abstractmethod
    def _heston_cf(self, phi):
        pass


    def _compute_integrals(self):

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

        return P1, P2

    @abstractmethod
    def heston_price(self):
        pass

    @abstractmethod
    def _compute_theta(self):
        pass

    def compute_first_order_greek(self, greek_name):
        greeks = {
            "delta" : self.S0,
            "vega" : self.sigma,
            "rho" : self.rho
        }

        if greek_name == "theta":
            greek = self._compute_theta()
            return greek

        elif greek_name in greeks:
            variable = greeks[greek_name]
            if variable.grad is not None:
                variable.grad.zero_()

            price = self.heston_price()
            price.backward()
            greek = variable.grad.item()
            return greek

    def plot_first_order_greek(self, greek_name, k_range):

        greek_list = []

        for k in k_range:
            self.K = torch.tensor([k], device=CONFIG.device)
            greek = self.compute_first_order_greek(greek_name)
            greek_list.append(greek)

        plt.plot(k_range, greek_list)
        plt.title(greek_name)
        plt.xlabel("Strike")
        plt.ylabel(f"{greek_name} value")
        plt.grid(True)
        plt.show()



    def compute_second_order_greek(self, greek_name, plot=False):
        pass





