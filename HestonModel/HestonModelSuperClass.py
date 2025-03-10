from abc import ABC, abstractmethod
import torch
from config import CONFIG
import matplotlib.pyplot as plt

class HestonModel(ABC):
    def __init__(self, S0, K, T, r, kappa, v0, theta, sigma, rho, type="call"):
        self.S0 = torch.tensor(S0, device=CONFIG.device, requires_grad=True)
        self.K = self._ensure_1d_tensor(torch.tensor(K, device=CONFIG.device))
        self.T = self._ensure_1d_tensor(torch.tensor(T, device=CONFIG.device))
        self.r = torch.tensor(r, device=CONFIG.device, requires_grad=True)
        self.kappa = torch.tensor(kappa, device=CONFIG.device)
        self.v0 = torch.tensor(v0, device=CONFIG.device)
        self.theta = torch.tensor(theta, device=CONFIG.device)
        self.sigma = torch.tensor(sigma, device=CONFIG.device, requires_grad=True)
        self.rho = torch.tensor(rho, device=CONFIG.device)
        self.type = type

    @abstractmethod
    def _heston_cf(self, phi):
        pass


    def _compute_integrals(self):

        # Vérification de la taille de K et T
        assert self.K.dim() == 1, "K doit être un tenseur 1D"
        assert self.T.dim() == 1, "T doit être un tenseur 1D"
        assert len(self.K) == len(self.T), "K et T doivent avoir la même taille"


        umax = 175
        n = 1501
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
            "delta": self.S0,
            "vega": self.sigma,
            "rho": self.r
        }

        if greek_name == "theta":
            return self._compute_theta()

        elif greek_name in greeks:
            variable = greeks[greek_name]
            if variable.grad is not None:
                variable.grad.zero_()

            price = self.heston_price()
            price.backward()
            return variable.grad.item()

        elif greek_name == "vanna":
            # Compute Vanna: d(Delta) / d(sigma)
            if self.sigma.grad is not None:
                self.sigma.grad.zero_()
            if self.S0.grad is not None:
                self.S0.grad.zero_()

            price = self.heston_price()
            price.backward(create_graph=True)

            delta = self.S0.grad.clone()  # Stocke Delta
            delta.backward(retain_graph=True)  # Backward pour obtenir Vanna
            return self.sigma.grad.item()

        elif greek_name == "volga":
            # Compute Volga: d²P / dσ²
            if self.sigma.grad is not None:
                self.sigma.grad.zero_()

            price = self.heston_price()
            price.backward(create_graph=True)  # Conserve le graphe pour dérivée 2ème ordre

            vega = self.sigma.grad.clone()  # Stocke Vega
            vega.backward(retain_graph=True)  # Backward pour Volga sans libérer le graphe
            return self.sigma.grad.item()

    def compute_second_order_greek(self, greek_name, plot=False):
        pass

    def _ensure_1d_tensor(self, tensor):
        """
        Assure que l'entrée est un tenseur 1D.
        Si l'entrée est un scalaire encapsulé dans un tenseur, elle sera transformée en 1D.
        """
        if tensor.dim() == 0:  # Si c'est un scalaire encapsulé
            return tensor.unsqueeze(0)  # Convertir en tenseur 1D
        return tensor




