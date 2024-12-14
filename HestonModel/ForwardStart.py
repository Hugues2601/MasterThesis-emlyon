import torch
from HestonModel.HestonModelSuperClass import HestonModel
from config import CONFIG

class ForwardStart(HestonModel):
    def __init__(self, S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        super().__init__(S0=S0, K=k, T=T2, r=r, kappa=kappa, v0=v0, theta=theta, sigma=sigma, rho=rho)
        self.k = self._ensure_1d_tensor(torch.tensor(k, device=CONFIG.device))
        self.T0 = torch.tensor(T0, device=CONFIG.device, requires_grad=True)
        self.T1 = torch.tensor(T1, device=CONFIG.device)
        self.T2 = torch.tensor(T2, device=CONFIG.device)


    def _heston_cf(self, phi):
        # Ensure that phi is a torch tensor on the GPU
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=torch.complex128, device=CONFIG.device)
        else:
            phi = phi.to(CONFIG.device).type(torch.complex128)


        S0 = self.S0.to(CONFIG.device).type(torch.float64)
        T0 = self.T0.to(CONFIG.device).type(torch.float64)
        T1 = self.T1.to(CONFIG.device).type(torch.float64)
        T2 = self.T2.to(CONFIG.device).type(torch.float64)
        r = self.r.to(CONFIG.device).type(torch.float64)

        tau = T2-T1

        delta = 4*self.kappa*self.theta/self.sigma**2
        little_c_bar = self.sigma**2/(4*self.kappa) * (1 - torch.exp(-self.kappa*(T1-T0)))
        kappa_bar = (4*self.kappa*self.v0*torch.exp(-self.kappa*(T1-T0))) / (self.sigma**2 * (1-torch.exp(-self.kappa*(T1-T0))))
        d = torch.sqrt((self.kappa-self.rho*self.sigma*1j*phi)**2 + self.sigma**2 * (phi**2 + 1j * phi))
        g = (self.kappa - self.rho*self.sigma*1j*phi-d)/(self.kappa-self.rho*self.sigma*1j*phi+d)

        A_bar = (
                self.r * 1j * phi * tau
                + (self.kappa * self.theta * tau / (self.sigma ** 2)) * (self.kappa - self.sigma * self.rho * 1j * phi - d)
                - (2 * self.kappa * self.theta / (self.sigma ** 2)) * torch.log((1.0 - g * torch.exp(-d * tau)) / (1.0 - g))
        )

        C_bar = (1-torch.exp(-d*tau))/(self.sigma**2 * (1-g*torch.exp(-d*tau))) * (self.kappa-self.rho*self.sigma*1j*phi - d)

        cf = torch.exp(A_bar + (C_bar * little_c_bar*kappa_bar)/(1 - 2*C_bar*little_c_bar)) * (1/(1-2*C_bar*little_c_bar))**(delta/2)
        return cf

    def heston_price(self):
        P1, P2 = self._compute_integrals()
        price = self.S0 * (P1 - self.k * torch.exp(-self.r * (self.T2 - self.T1)) * P2)
        return price

    def _compute_theta(self):
        if self.T0.grad is not None:
            self.T0.grad.zero_()

        price = self.heston_price()
        price.backward()
        greek = self.T0.grad.item()
        return greek