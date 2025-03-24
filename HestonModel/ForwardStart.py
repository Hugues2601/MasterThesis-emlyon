import torch
from HestonModel.HestonModelSuperClass import HestonModel
from config import CONFIG
import matplotlib.pyplot as plt
import numpy as np

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


        S0 = self.S0.to(CONFIG.device).type(torch.float32)
        T0 = self.T0.to(CONFIG.device).type(torch.float32)
        T1 = self.T1.to(CONFIG.device).type(torch.float32)
        T2 = self.T2.to(CONFIG.device).type(torch.float32)
        r = self.r.to(CONFIG.device).type(torch.float32)

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

    def compute_heston_prices(self, S_paths, v_paths, t):
        """
        Version vectorisée de la simulation des prix Heston pour tous les chemins en parallèle sur GPU.

        Args:
        - S_paths (torch.Tensor): Matrice des prix simulés de taille (n_paths, n_steps).
        - v_paths (torch.Tensor): Matrice des variances simulées de taille (n_paths, n_steps).
        - t (int): Indice temporel t pour lequel calculer les prix.

        Returns:
        - prices_t (torch.Tensor): Tensor des prix Heston au temps t.
        - prices_t1 (torch.Tensor): Tensor des prix Heston au temps t+1.
        """
        n_paths = S_paths.shape[0]

        # Déplacer les données sur GPU pour accélérer le calcul
        S_t = S_paths[:, t].to(CONFIG.device)
        v_t = v_paths[:, t].to(CONFIG.device)
        S_t1 = S_paths[:, t + 1].to(CONFIG.device)
        v_t1 = v_paths[:, t + 1].to(CONFIG.device)

        # Recréer un objet ForwardStart mais avec batch S_t et v_t
        forward_start_t = ForwardStart(S0=S_t, k=self.k, T0=self.T0, T1=self.T1, T2=self.T2,
                                       r=self.r, kappa=self.kappa, v0=v_t, theta=self.theta,
                                       sigma=self.sigma, rho=self.rho)

        forward_start_t1 = ForwardStart(S0=S_t1, k=self.k, T0=self.T0 - 1/252, T1=self.T1 - 1/252, T2=self.T2 - 1/252,
                                        r=self.r, kappa=self.kappa, v0=v_t1, theta=self.theta,
                                        sigma=self.sigma, rho=self.rho)


        # Calculer les prix Heston en batch
        prices_t = forward_start_t.heston_price()
        prices_t1 = forward_start_t1.heston_price()
        pnl_total = prices_t1 - prices_t  # PnL = prix_t+1 - prix_t

        return prices_t, prices_t1, pnl_total

    def compute_explained_pnl(self, S_paths, v_paths, t, dt, dt_path):
        """
        Calcule le PnL expliqué à l'instant t en batch pour chaque chemin.

        Args:
        - S_paths (torch.Tensor): Matrice des prix simulés (n_paths, n_steps).
        - v_paths (torch.Tensor): Matrice des variances simulées (n_paths, n_steps).
        - t (int): Indice temporel pour lequel calculer le PnL expliqué.
        - dt (float): Pas de temps écoulé entre t et t+1.

        Returns:
        - explained_pnl (torch.Tensor): Tensor du PnL expliqué pour chaque chemin.
        """
        # Déplacer les données sur GPU
        S_t = S_paths[:, t].to(CONFIG.device)
        S_t1 = S_paths[:, t + 1].to(CONFIG.device)
        v_t = v_paths[:, t].to(CONFIG.device)
        v_t1 = v_paths[:, t + 1].to(CONFIG.device)
        dt_t = dt_path[:, t].to(CONFIG.device)
        T0_t = self.T0.expand_as(S_t).to(CONFIG.device)

        print("min de v_t", torch.min(v_paths))
        print("max de v_t", torch.max(v_paths))
        print("min de S_t", torch.min(S_paths))
        print("max de S_t", torch.max(S_paths))

        # Création d'instances batch ForwardStart pour calculer les prix et grecs
        forward_start_t = ForwardStart(S0=S_t, k=self.k, T0=T0_t, T1=self.T1, T2=self.T2,
                                       r=self.r, kappa=self.kappa, v0=v_t, theta=self.theta,
                                       sigma=self.sigma, rho=self.rho)

        # Calculer le prix de l'option à t
        price_t = forward_start_t.heston_price()

        # Calcul des Grecs
        delta = forward_start_t.compute_greek("delta", batch=True)
        vega = forward_start_t.compute_greek("vega", batch=True)
        # vanna = forward_start_t.compute_greek("vanna", batch=True)
        # vomma = forward_start_t.compute_greek("vomma", batch=True)
        theta = forward_start_t.compute_greek("theta", batch=True)

        # Calcul des variations des variables
        dS = S_t1 - S_t
        dv = v_t1 - v_t
        dT = 1/252

        # explained_pnl = delta * dS + vega * dv + theta * dT + 0.5 * vanna * dS * dv + 0.5 * vomma * dv**2
        explained_pnl = delta * dS + vega * dv + theta * dT

        return explained_pnl

    def compute_greek(self, greek_name, batch=False):
        greeks = {
            "delta": self.S0,
            "vega": self.v0,
            "rho": self.r,
            "theta": self.T0,
            "gamma": (self.S0, self.S0),  # d²V/dS0²
            "vanna": (self.S0, self.v0),  # d²V/dS0 dv0
            "vomma": (self.v0, self.v0),  # d²V/dv0²
        }

        variable = greeks[greek_name]
        price = self.heston_price()

        if batch:
            price = price.sum()  # Ensure a scalar output for batch mode

        if isinstance(variable, tuple):  # Second-order Greeks (Gamma, Vanna, Vomma)
            var1, var2 = variable
            first_derivative, = torch.autograd.grad(price, var1, create_graph=True)


            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, var2)

            return second_derivative if batch else second_derivative.item()
        else:  # First-order Greeks (Delta, Vega, Rho, Theta)
            first_derivative, = torch.autograd.grad(price, variable)
            return first_derivative if batch else first_derivative.item()

    def sensitivity_analysis(self, param_name, param_range):
        """
        Analyse la sensibilité du prix de l'option Forward Start par rapport à un paramètre donné.
        :param param_name: Nom du paramètre à faire varier ("kappa", "theta", "v0", "sigma", "rho")
        :param param_range: Liste ou np.array contenant les valeurs du paramètre à tester.
        """
        prices = []

        for value in param_range:
            setattr(self, param_name, torch.tensor(value, device=CONFIG.device, dtype=torch.float32))
            price = self.heston_price().item()
            prices.append(price)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(param_range, prices, linestyle='-', label=f"Impact of {param_name} for k={self.k.item()}", color="black")
        plt.xlabel(param_name)
        plt.ylabel("Forward Start Option Price")
        plt.title(f"Sensitivity of Forward Start Option Price to {param_name}")
        plt.legend()
        plt.grid()
        plt.show()

    def sensitivity_analysis_all(self, S0, r, calibrated_params):
        fs_option = ForwardStart(S0=S0, k=0.75, T0=0.0, T1=1.0, T2=3.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
                                 sigma=calibrated_params["sigma"], rho=calibrated_params["rho"])

        kappa_range = np.linspace(0.05, 4, 500)
        fs_option.sensitivity_analysis("kappa", kappa_range)

        v0_range = np.linspace(0.01, 0.2, 500)
        fs_option.sensitivity_analysis("v0", v0_range)

        # Faire varier theta de 0.01 à 0.2
        theta_range = np.linspace(0.01, 0.2, 500)
        fs_option.sensitivity_analysis("theta", theta_range)

        # Faire varier sigma de 0.1 à 1.0
        sigma_range = np.linspace(0.1, 1.0, 500)
        fs_option.sensitivity_analysis("sigma", sigma_range)

        # Faire varier rho de -0.9 à 0.9
        rho_range = np.linspace(-0.9, 0.9, 500)
        fs_option.sensitivity_analysis("rho", rho_range)

        # Exemple d'utilisation (en gardant les autres paramètres fixes)
        fs_option = ForwardStart(S0=S0, k=1.0, T0=0.0, T1=1.0, T2=3.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
                                 sigma=calibrated_params["sigma"], rho=calibrated_params["rho"])

        kappa_range = np.linspace(0.05, 4, 500)
        fs_option.sensitivity_analysis("kappa", kappa_range)

        v0_range = np.linspace(0.01, 0.2, 500)
        fs_option.sensitivity_analysis("v0", v0_range)

        # Faire varier theta de 0.01 à 0.2
        theta_range = np.linspace(0.01, 0.2, 500)
        fs_option.sensitivity_analysis("theta", theta_range)

        # Faire varier sigma de 0.1 à 1.0
        sigma_range = np.linspace(0.1, 1.0, 500)
        fs_option.sensitivity_analysis("sigma", sigma_range)

        # Faire varier rho de -0.9 à 0.9
        rho_range = np.linspace(-0.9, 0.9, 500)
        fs_option.sensitivity_analysis("rho", rho_range)

        # Exemple d'utilisation (en gardant les autres paramètres fixes)
        fs_option = ForwardStart(S0=S0, k=1.25, T0=0.0, T1=1.0, T2=3.0, r=r, kappa=calibrated_params["kappa"], v0=calibrated_params["v0"], theta=calibrated_params["theta"],
                                 sigma=calibrated_params["sigma"], rho=calibrated_params["rho"])

        kappa_range = np.linspace(0.05, 4, 500)
        fs_option.sensitivity_analysis("kappa", kappa_range)

        v0_range = np.linspace(0.01, 0.2, 500)
        fs_option.sensitivity_analysis("v0", v0_range)

        # Faire varier theta de 0.01 à 0.2
        theta_range = np.linspace(0.01, 0.2, 500)
        fs_option.sensitivity_analysis("theta", theta_range)

        # Faire varier sigma de 0.1 à 1.0
        sigma_range = np.linspace(0.1, 1.0, 500)
        fs_option.sensitivity_analysis("sigma", sigma_range)

        # Faire varier rho de -0.9 à 0.9
        rho_range = np.linspace(-0.9, 0.9, 500)
        fs_option.sensitivity_analysis("rho", rho_range)
