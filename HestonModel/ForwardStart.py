import torch
from HestonModel.HestonModelSuperClass import HestonModel
from HestonModel.Vanilla import VanillaHestonPrice
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
            "gamma": self.S0,
            "vanna": (self.S0, self.v0),
            "volga": self.v0,
        }

        variable = greeks[greek_name]
        price = self.heston_price()

        if batch:
            price = price.sum()

        if greek_name == "volga":
            first_derivative, = torch.autograd.grad(price, self.v0, create_graph=True)
            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, self.v0)
            volga = 2 * first_derivative + 4 * self.v0 * second_derivative
            return volga if batch else volga.item()

        if isinstance(variable, tuple):
            var1, var2 = variable
            first_derivative, = torch.autograd.grad(price, var1, create_graph=True)
            second_derivative, = torch.autograd.grad(first_derivative.sum() if batch else first_derivative, var2)
            return second_derivative if batch else second_derivative.item()

        else:
            derivative, = torch.autograd.grad(price, variable)

            if greek_name == "vega":
                adjusted_vega =2 * torch.sqrt(self.v0) * derivative
                return adjusted_vega if batch else adjusted_vega.item()

            return derivative if batch else derivative.item()

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

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    def sensitivity_analysis_all(self, S0, r, calibrated_params):
        """
        Affiche les sensibilités du prix de l'option Forward Start à plusieurs paramètres du modèle de Heston,
        avec 5 subplots (3 colonnes x 2 lignes), 3 courbes par subplot (3 maturités),
        une légende globale dans la cellule vide, et une bordure noire autour de chaque subplot.
        """

        # Paramètres et labels LaTeX
        param_info = {
            "kappa": (np.linspace(0.05, 4, 300), r"$\kappa$"),
            "v0": (np.linspace(0.01, 0.2, 300), r"$v_0$"),
            "theta": (np.linspace(0.01, 0.2, 300), r"$\theta$"),
            "sigma": (np.linspace(0.1, 1.0, 300), r"$\sigma$"),
            "rho": (np.linspace(-0.9, 0.9, 300), r"$\rho$"),
        }

        # Maturités et styles associés
        maturities = [
            (0.5, 1.0, 'black', r"$T_1=0.5,\ T_2=1.0$"),
            (0.75, 1.5, '#d08770', r"$T_1=0.75,\ T_2=1.5$"),
            (1.0, 2.0, '#8fbcbb', r"$T_1=1.0,\ T_2=2.0$"),
        ]

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()

        for idx, (param_name, (param_range, latex_label)) in enumerate(param_info.items()):
            ax = axs[idx]

            for T1, T2, color, label in maturities:
                prices = []

                fs_option = ForwardStart(
                    S0=S0,
                    k=1.0,
                    T0=0.0,
                    T1=T1,
                    T2=T2,
                    r=r,
                    kappa=calibrated_params["kappa"],
                    v0=calibrated_params["v0"],
                    theta=calibrated_params["theta"],
                    sigma=calibrated_params["sigma"],
                    rho=calibrated_params["rho"]
                )

                for value in param_range:
                    setattr(fs_option, param_name, torch.tensor(value, device=CONFIG.device, dtype=torch.float32))
                    price = fs_option.heston_price().item()
                    prices.append(price)

                ax.plot(param_range, prices, color=color, label=label)

            ax.set_title(f"Impact of {latex_label}", fontsize=14)
            ax.set_xlabel(latex_label, fontsize=12)
            ax.set_ylabel("Option Price", fontsize=12)
            ax.grid(True)

            # Encadrer chaque subplot avec une bordure noire
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

        # Légende dans la dernière cellule vide
        legend_ax = axs[-1]
        legend_ax.axis('off')
        handles, labels = axs[0].get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            loc='center',
            fontsize=16,
            handlelength=4,
            handletextpad=3
        )

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

pastel_colors = [
    "black",        # utilisé pour la première paire
    "#8fbcbb",      # bleu-vert clair
    "#a3be8c",      # vert pastel
    "#d08770",      # orange pastel
    "#e09ec7",      # rose
    "#b48ead"       # violet pastel
]


def plot_forward_start_vs_vanilla_price_multi_maturity(
    S0,
    k_range,  # valeurs de k entre 0.6 et 1.4 typiquement
    r,
    kappa, v0, theta, sigma, rho
):
    # Définir les différentes maturités
    maturities = [
        (1.75, 2.0, 2.5, .5),
        (0.0, 0.75, 1.5, .75),
        (0.0, 1.0, 2.0, 1.0)
    ]

    # Utiliser les couleurs définies
    color_pairs = [
        (pastel_colors[0], pastel_colors[0]),
        (pastel_colors[2], pastel_colors[2]),
        (pastel_colors[3], pastel_colors[3])
    ]

    plt.figure(figsize=(10, 5))

    for idx, (T0, T1, T2, T_vanilla) in enumerate(maturities):
        fs_prices = []
        vanilla_prices = []

        for k in k_range:
            # --- Forward Start Pricing ---
            model_fs = ForwardStart(
                S0=S0,
                k=k,
                T0=T0,
                T1=T1,
                T2=T2,
                r=r,
                kappa=kappa,
                v0=v0,
                theta=theta,
                sigma=sigma,
                rho=rho
            )
            fs_price = model_fs.heston_price().item()
            fs_prices.append(fs_price)

            # --- Vanilla Pricing ---
            K_vanilla = S0 * k
            model_vanilla = VanillaHestonPrice(
                S0=S0,
                K=K_vanilla,
                T=T_vanilla,
                r=r,
                kappa=kappa,
                v0=v0,
                theta=theta,
                sigma=sigma,
                rho=rho,
                type="call"
            )
            vanilla_price = model_vanilla.heston_price().item()
            vanilla_prices.append(vanilla_price)

        label_fs = f"Forward Start t1={T1}, t2={T2}"
        label_vanilla = f"Vanilla T={T_vanilla}"

        fs_color, vanilla_color = color_pairs[idx]

        plt.plot(k_range, fs_prices, label=label_fs, linewidth=2, color=fs_color)
        plt.plot(k_range, vanilla_prices, label=label_vanilla, linewidth=2, linestyle='--', color=vanilla_color)

    # Axes
    plt.xlabel("k (Strike Coefficient for FS — Moneyness for Vanilla)", fontsize=13)
    plt.ylabel("Option Price", fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_forward_start_price_t0_variation(
    S0,
    k_range,  # ex: np.linspace(0.6, 1.4, 50)
    r,
    kappa, v0, theta, sigma, rho
):
    # Valeurs fixes pour T1 et T2
    T1 = 2.0
    T2 = 3.0

    # Valeurs de T0 à tester
    T0_values = [0.0, 0.5, 1.0, 1.75]

    # Couleurs personnalisées si tu veux
    colors = pastel_colors[:4]  # ou mets ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] par exemple

    plt.figure(figsize=(10, 5))

    for idx, T0 in enumerate(T0_values):
        fs_prices = []

        for k in k_range:
            model_fs = ForwardStart(
                S0=S0,
                k=k,
                T0=T0,
                T1=T1,
                T2=T2,
                r=r,
                kappa=kappa,
                v0=v0,
                theta=theta,
                sigma=sigma,
                rho=rho
            )
            fs_price = model_fs.heston_price().item()
            fs_prices.append(fs_price)

        label = f"FS Option (T0={T0}, T1={T1}, T2={T2})"
        plt.plot(k_range, fs_prices, label=label, linewidth=2, color=colors[idx])

    # Axes
    plt.xlabel("k (Strike Coefficient at T1)", fontsize=13)
    plt.ylabel("Forward Start Option Price", fontsize=13)
    plt.title("Forward Start Option Prices for Varying T0", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





