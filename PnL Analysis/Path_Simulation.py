import torch
import matplotlib.pyplot as plt

from HestonModel.ForwardStart import ForwardStart


# Définition de la classe HestonSimulator
import torch


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
        """
        Simulate the Heston model for a given number of paths.

        Args:
        - n_paths (int): Number of Monte Carlo simulation paths.

        Returns:
        - S (torch.Tensor): Simulated asset price paths of shape (n_paths, n_steps).
        - v (torch.Tensor): Simulated variance paths of shape (n_paths, n_steps).
        """
        S = torch.zeros(n_paths, self.n_steps, device=self.device)
        v = torch.zeros(n_paths, self.n_steps, device=self.device)
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        dW_S = torch.randn(n_paths, self.n_steps - 1, device=self.device) * torch.sqrt(
            torch.tensor(self.dt, device=self.device))
        dW_v = self.rho * dW_S + torch.sqrt(torch.tensor(1 - self.rho ** 2)) * torch.randn(n_paths, self.n_steps - 1,
                                                                                           device=self.device) * torch.sqrt(
            torch.tensor(self.dt, device=self.device))

        for t in range(1, self.n_steps):
            v[:, t] = torch.maximum(
                v[:, t - 1] + self.kappa * (self.theta - v[:, t - 1]) * self.dt +
                self.sigma * torch.sqrt(torch.maximum(v[:, t - 1], torch.tensor(0.0, device=self.device))) * dW_v[:,
                                                                                                             t - 1],
                torch.tensor(0.0, device=self.device)
            )
            S[:, t] = S[:, t - 1] * torch.exp(
                (self.r - 0.5 * v[:, t - 1]) * self.dt +
                torch.sqrt(torch.maximum(v[:, t - 1], torch.tensor(0.0, device=self.device))) * dW_S[:, t - 1]
            )

        return S, v  # Returns tensors directly

    def plot_paths(self, S, v, num_paths=10):
        """
        Affiche les trajectoires simulées pour S et v.

        Args:
        - S (torch.Tensor): Matrice des prix simulés (n_paths, n_steps).
        - v (torch.Tensor): Matrice des variances simulées (n_paths, n_steps).
        - num_paths (int): Nombre de chemins à afficher pour une meilleure lisibilité.
        """
        S_cpu, v_cpu = S[:num_paths].cpu().numpy(), v[:num_paths].cpu().numpy()
        time_grid = torch.linspace(0, self.T, steps=self.n_steps).cpu().numpy()

        # Graphique des prix S_t
        plt.figure(figsize=(12, 5))
        for i in range(num_paths):
            plt.plot(time_grid, S_cpu[i], alpha=0.6)
        plt.title(f"Simulated Heston Price Paths ({num_paths} paths)")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price S_t")
        plt.grid()
        plt.show()

        # Graphique des variances v_t
        plt.figure(figsize=(12, 5))
        for i in range(num_paths):
            plt.plot(time_grid, v_cpu[i], alpha=0.6)
        plt.title(f"Simulated Heston Variance Paths ({num_paths} paths)")
        plt.xlabel("Time (Years)")
        plt.ylabel("Variance v_t")
        plt.grid()
        plt.show()


# Simuler les chemins Heston
simulator = HestonSimulator(S0=100.0, r=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, T=1, dt=1/252)
S_paths, v_paths = simulator.simulate(n_paths=20000)

# Instancier ForwardStart
forward_start = ForwardStart(S0=100.0, k=1.1, T0=0.25, T1=0.5, T2=1,
                             r=0.05, kappa=2.0, v0=0.04, theta=0.04,
                             sigma=0.3, rho=-0.7)

# Calcul des prix à t=100 et t+1=101
prices_t, prices_t1, pnl_tot = forward_start.compute_heston_prices(S_paths, v_paths, t=100)

# Afficher les 5 premiers résultats
print("Prix à t :", prices_t[:5])
print(len(prices_t))
print("Prix à t+1 :", prices_t1[:5])
print("PnL total :", pnl_tot)



# # Paramètres de simulation
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
#     "n_paths": 1,   # Une seule trajectoire pour simplifier
#     "device": "cuda" # Exécution sur CPU pour affichage
# }
#
# # Instanciation du simulateur Heston
# simulator = HestonSimulator(**params)
#
# # Simulation d'une trajectoire
# S_traj, v_traj = simulator.simulate()
#
#
# t = 100
# S_t = S_traj[0, t]      # Prix du sous-jacent à t
# v_t = v_traj[0, t]      # Variance à t
# r_t = 0.02      # On suppose que le taux sans risque reste constant
#
# # Valeurs à t+1
# S_t1 = S_traj[0, t+1]
# v_t1 = v_traj[0, t+1]
# r_t1 = 0.02
#
# # Définition des paramètres du forward start
# k = 1.0   # Ratio de strike par rapport au sous-jacent
# T0 = 0.0  # Temps actuel
# T1 = 1.0  # Début de l'option
# T2 = 2.0  # Expiration de l'option
#
# # Calcul du prix de l'option Forward Start à t
# fs_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
#                     kappa=params["kappa"], v0=v_t, theta=params["theta"],
#                     sigma=params["sigma"], rho=params["rho"])
# price_t = fs_t.heston_price().item()
#
# # Calcul du prix de l'option Forward Start à t+1
# fs_t1 = ForwardStart(S0=S_t1, k=k, T0=T0, T1=T1, T2=T2, r=r_t1,
#                      kappa=params["kappa"], v0=v_t1, theta=params["theta"],
#                      sigma=params["sigma"], rho=params["rho"])
# price_t1 = fs_t1.heston_price().item()
#
# print(f"Prix de l'option Forward Start à t={t}: {price_t:.4f}")
# print(f"Prix de l'option Forward Start à t+1={t+1}: {price_t1:.4f}")
#
# # Variations des facteurs de risque
# dS = S_t1 - S_t    # Variation du sous-jacent
# dV = v_t1 - v_t    # Variation de la variance (on suppose sigma = sqrt(variance))
# dr = r_t1 - r_t    # Variation du taux sans risque (souvent 0)
# dT = -params["dt"] # Le temps avance donc T diminue
#
# # Création des objets ForwardStart à t et t+1
# FS_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
#                     kappa=params["kappa"], v0=v_t, theta=params["theta"],
#                     sigma=params["sigma"], rho=params["rho"])
#
# FS_t1 = ForwardStart(S0=S_t1, k=k, T0=T0, T1=T1, T2=T2, r=r_t1,
#                      kappa=params["kappa"], v0=v_t1, theta=params["theta"],
#                      sigma=params["sigma"], rho=params["rho"])
#
# # Calcul des Grecs à t
# delta_t = FS_t.compute_first_order_greek("delta")
# vega_t = FS_t.compute_first_order_greek("vega")
# rho_t = FS_t.compute_first_order_greek("rho")
# theta_t = FS_t.compute_first_order_greek("theta")
#
# # Calcul du PnL expliqué
# PnL_explained = delta_t * dS + vega_t * dV + rho_t * dr + theta_t * dT
#
# # Affichage des résultats
# print(f"PnL expliqué : {PnL_explained:.4f}")
#
# PnL_inexpliqué = (price_t1 - price_t) - PnL_explained
# print(f"PnL inexpliqué : {PnL_inexpliqué:.4f}")
#
# import torch
# import matplotlib.pyplot as plt
# import scipy.stats as stats
#
# # Vérification du GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Utilisation de {device}")
#
# # Paramètres
# num_simulations = 1000
# PnL_inexpliqué_list = torch.zeros(num_simulations, device=device)
#
# j=0
# for i in range(num_simulations):
#     j+=1
#     # Simulation d'une trajectoire
#     S_traj, v_traj = simulator.simulate()
#
#     # Choix de t
#     t = 100
#     S_t, v_t = S_traj[0, t], v_traj[0, t]
#     S_t1, v_t1 = S_traj[0, t + 1], v_traj[0, t + 1]
#     r_t, r_t1 = 0.02, 0.02  # On garde r constant
#
#     # Création des objets ForwardStart
#     fs_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
#                         kappa=params["kappa"], v0=v_t, theta=params["theta"],
#                         sigma=params["sigma"], rho=params["rho"])
#
#     fs_t1 = ForwardStart(S0=S_t1, k=k, T0=T0, T1=T1, T2=T2, r=r_t1,
#                          kappa=params["kappa"], v0=v_t1, theta=params["theta"],
#                          sigma=params["sigma"], rho=params["rho"])
#
#     # Calcul des prix
#     price_t = torch.tensor(fs_t.heston_price().item(), device=device)
#     price_t1 = torch.tensor(fs_t1.heston_price().item(), device=device)
#
#     # Calcul du PnL total
#     PnL_total = price_t1 - price_t
#
#     # Calcul des Grecs
#     delta_t = torch.tensor(fs_t.compute_first_order_greek("delta"), device=device)
#     vega_t = torch.tensor(fs_t.compute_first_order_greek("vega"), device=device)
#     rho_t = torch.tensor(fs_t.compute_first_order_greek("rho"), device=device)
#     theta_t = torch.tensor(fs_t.compute_first_order_greek("theta"), device=device)
#     # vanna_t = torch.tensor(fs_t.compute_first_order_greek("vanna"), device=device)
#     # volga_t = torch.tensor(fs_t.compute_first_order_greek("volga"), device=device)
#
#     # Calcul du PnL expliqué
#     dS = torch.tensor(S_t1 - S_t, device=device)
#     dV = torch.tensor(v_t1 - v_t, device=device)
#     dr = torch.tensor(r_t1 - r_t, device=device)
#     dT = torch.tensor(-params["dt"], device=device)
#
#
#     PnL_explained = delta_t * dS + vega_t * dV + rho_t * dr + theta_t * dT
#
#     # Calcul du PnL inexpliqué
#     PnL_inexpliqué_list[i] = PnL_total - PnL_explained
#     print(j)
#
# # 🔹 Analyse statistique
# mean_pnl_inexpliqué = torch.mean(PnL_inexpliqué_list).item()
# std_pnl_inexpliqué = torch.std(PnL_inexpliqué_list).item()
#
# # Test de normalité basé sur skewness et kurtosis (sans numpy)
# skewness = torch.mean((PnL_inexpliqué_list - mean_pnl_inexpliqué) ** 3) / std_pnl_inexpliqué ** 3
# kurtosis = torch.mean((PnL_inexpliqué_list - mean_pnl_inexpliqué) ** 4) / std_pnl_inexpliqué ** 4
#
# # 🔥 Affichage des résultats
# print(f"Moyenne du PnL inexpliqué : {mean_pnl_inexpliqué:.6f}")
# print(f"Écart-type du PnL inexpliqué : {std_pnl_inexpliqué:.6f}")
# print(f"Skewness : {skewness.item():.6f}")
# print(f"Kurtosis : {kurtosis.item():.6f}")
#
# # 📊 Histogramme du PnL inexpliqué
# plt.figure(figsize=(10, 5))
# plt.hist(PnL_inexpliqué_list.cpu().tolist(), bins=50, alpha=0.7, color='blue', edgecolor='black')
# plt.axvline(mean_pnl_inexpliqué, color='red', linestyle='dashed', linewidth=2, label="Moyenne")
# plt.xlabel("PnL inexpliqué")
# plt.ylabel("Fréquence")
# plt.title("Distribution du PnL inexpliqué sur 1000 simulations")
# plt.legend()
# plt.grid()
# plt.show()
#
# # 📈 Q-Q Plot
# plt.figure(figsize=(8, 6))
# stats.probplot(PnL_inexpliqué_list.cpu().numpy(), dist="norm", sparams=(mean_pnl_inexpliqué, std_pnl_inexpliqué), plot=plt)
# plt.title("Q-Q Plot du PnL inexpliqué")
# plt.grid()
# plt.show()
