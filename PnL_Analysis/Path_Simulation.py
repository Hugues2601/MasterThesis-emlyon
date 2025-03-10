import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
from HestonModel.ForwardStart import ForwardStart


class ForwardStartAnalysis:
    def __init__(self, params, device="cuda"):
        self.params = params
        self.device = device
        self.simulator = self.HestonSimulator(**params)

    class HestonSimulator:
        def __init__(self, S0, r, kappa, theta, sigma, rho, v0, T, dt, device="cuda"):
            self.S0 = S0
            self.r = r
            self.kappa = kappa
            self.theta = theta
            self.sigma = sigma
            self.rho = rho
            self.v0 = v0
            self.T = T
            self.dt = dt
            self.device = device
            self.n_steps = int(T / dt)

        def simulate(self, n_paths):
            """ Simulation vectorisée avec génération de 10 000 chemins en une seule passe """
            dt_tensor = torch.tensor(self.dt, device=self.device, dtype=torch.float32)

            S = torch.full((n_paths, self.n_steps), self.S0, device=self.device, dtype=torch.float32)
            v = torch.full((n_paths, self.n_steps), self.v0, device=self.device, dtype=torch.float32)

            dW_S = torch.randn(n_paths, self.n_steps - 1, device=self.device, dtype=torch.float32) * torch.sqrt(
                dt_tensor)
            dW_v = self.rho * dW_S + torch.sqrt(torch.tensor(1 - self.rho ** 2, device=self.device)) * torch.randn(
                n_paths, self.n_steps - 1, device=self.device, dtype=torch.float32) * torch.sqrt(dt_tensor)

            # Simulation en batch (full GPU, plus de boucle sur le temps)
            v[:, 1:] = torch.maximum(
                v[:, :-1] + self.kappa * (self.theta - v[:, :-1]) * self.dt + self.sigma * torch.sqrt(
                    torch.maximum(v[:, :-1], torch.tensor(0.0, device=self.device))) * dW_v,
                torch.tensor(0.0, device=self.device)
            )
            S[:, 1:] = S[:, :-1] * torch.exp((self.r - 0.5 * v[:, :-1]) * dt_tensor + torch.sqrt(
                torch.maximum(v[:, :-1], torch.tensor(0.0, device=self.device))) * dW_S)

            return S, v

    def compute_pnl_analysis(self, num_simulations=10_000):
        """ Optimisation du calcul du PnL inexpliqué sur 10 000 chemins en full GPU """
        k, T0, T1, T2 = 1.0, 0.0, 1.0, 2.0
        r_t = 0.02

        # 🔥 Simulation en batch de 10 000 chemins
        S_traj, v_traj = self.simulator.simulate(n_paths=num_simulations)

        # 🔥 Sélection des valeurs à t et t+1 en batch
        t = 100
        S_t, v_t = S_traj[:, t], v_traj[:, t]
        S_t1, v_t1 = S_traj[:, t + 1], v_traj[:, t + 1]

        # 🔥 Pricing en batch (full GPU)
        price_t = torch.tensor(
            [ForwardStart(S0=s.item(), k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v.item(), theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"]).heston_price().item()
             for s, v in zip(S_t, v_t)], device=self.device
        )

        price_t1 = torch.tensor(
            [ForwardStart(S0=s.item(), k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v.item(), theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"]).heston_price().item()
             for s, v in zip(S_t1, v_t1)], device=self.device
        )

        # 🔥 Calcul du PnL total en batch
        PnL_total = price_t1 - price_t

        # 🔥 Calcul des variations des facteurs de risque
        dS, dV = S_t1 - S_t, v_t1 - v_t
        dr = torch.zeros(num_simulations, device=self.device)  # r reste constant
        dT = -self.params["dt"] * torch.ones(num_simulations, device=self.device)

        # 🔥 Calcul des Grecs en batch
        delta_t = torch.tensor(
            [ForwardStart(S0=s.item(), k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v.item(), theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"]).compute_first_order_greek("delta")
             for s, v in zip(S_t, v_t)], device=self.device
        )

        vega_t = torch.tensor(
            [ForwardStart(S0=s.item(), k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v.item(), theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"]).compute_first_order_greek("vega")
             for s, v in zip(S_t, v_t)], device=self.device
        )

        rho_t = torch.tensor(
            [ForwardStart(S0=s.item(), k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v.item(), theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"]).compute_first_order_greek("rho")
             for s, v in zip(S_t, v_t)], device=self.device
        )

        theta_t = torch.tensor(
            [ForwardStart(S0=s.item(), k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v.item(), theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"]).compute_first_order_greek("theta")
             for s, v in zip(S_t, v_t)], device=self.device
        )

        # 🔥 Calcul du PnL expliqué en batch
        PnL_explained = delta_t * dS + vega_t * dV + rho_t * dr + theta_t * dT

        # 🔥 Calcul du PnL inexpliqué en batch
        PnL_inexpliqué_list = PnL_total - PnL_explained

        return PnL_inexpliqué_list

    def analyze_results(self, PnL_inexpliqué_list):
        mean_pnl_inexpliqué = torch.mean(PnL_inexpliqué_list).item()
        std_pnl_inexpliqué = torch.std(PnL_inexpliqué_list).item()
        skewness = torch.mean((PnL_inexpliqué_list - mean_pnl_inexpliqué) ** 3) / std_pnl_inexpliqué ** 3
        kurtosis = torch.mean((PnL_inexpliqué_list - mean_pnl_inexpliqué) ** 4) / std_pnl_inexpliqué ** 4

        print(f"Moyenne du PnL inexpliqué : {mean_pnl_inexpliqué:.6f}")
        print(f"Écart-type du PnL inexpliqué : {std_pnl_inexpliqué:.6f}")
        print(f"Skewness : {skewness.item():.6f}")
        print(f"Kurtosis : {kurtosis.item():.6f}")

        plt.figure(figsize=(10, 5))
        plt.hist(PnL_inexpliqué_list.cpu().tolist(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(mean_pnl_inexpliqué, color='red', linestyle='dashed', linewidth=2, label="Moyenne")
        plt.xlabel("PnL inexpliqué")
        plt.ylabel("Fréquence")
        plt.title("Distribution du PnL inexpliqué")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(8, 6))
        stats.probplot(PnL_inexpliqué_list.cpu().numpy(), dist="norm",
                       sparams=(mean_pnl_inexpliqué, std_pnl_inexpliqué), plot=plt)
        plt.title("Q-Q Plot du PnL inexpliqué")
        plt.grid()
        plt.show()

#
# # Définition des paramètres de simulation
# params = {
#     "S0": 400,      # Prix initial du sous-jacent
#     "r": 0.02,      # Taux sans risque
#     "kappa": 3.0,   # Vitesse de réversion
#     "theta": 0.04,  # Niveau de variance de long-terme
#     "sigma": 0.5,   # Volatilité de la variance (vol of vol)
#     "rho": -0.7,    # Corrélation entre Wt^S et Wt^v
#     "v0": 0.04,     # Variance initiale
#     "T": 1.0,       # Horizon de simulation (1 an)
#     "dt": 1/252,    # Pas de temps journalier
#     "device": "cuda" # Exécution sur GPU
# }
#
# # Instanciation de la classe d'analyse
# analysis = ForwardStartAnalysis(params)
#
# # Exécution de l'analyse sur 10 000 simulations (full GPU)
# PnL_inexpliqué_list = analysis.compute_pnl_analysis(num_simulations=2000)
#
# # 🔥 Résultats : Distribution du PnL inexpliqué
# analysis.analyze_results(PnL_inexpliqué_list)
