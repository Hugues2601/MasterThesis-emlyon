import torch
import matplotlib.pyplot as plt

from HestonModel.ForwardStart import ForwardStart


# Définition de la classe HestonSimulator
class HestonSimulator:
    def __init__(self, S0, r, kappa, theta, sigma, rho, v0, T, dt, n_paths=1, device="cpu"):
        self.S0 = S0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.T = T
        self.dt = dt
        self.n_paths = n_paths
        self.device = device
        self.n_steps = int(T / dt)

    def simulate(self):
        S = torch.zeros(self.n_paths, self.n_steps, device=self.device)
        v = torch.zeros(self.n_paths, self.n_steps, device=self.device)
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        dW_S = torch.randn(self.n_paths, self.n_steps - 1, device=self.device) * torch.sqrt(
            torch.tensor(self.dt, device=self.device))
        dW_v = self.rho * dW_S + torch.sqrt(torch.tensor(1 - self.rho ** 2)) * torch.randn(self.n_paths, self.n_steps - 1,
                                                                             device=self.device) * torch.sqrt(
            torch.tensor(self.dt, device=self.device))

        for t in range(1, self.n_steps):
            v[:, t] = torch.maximum(
                v[:, t - 1] + self.kappa * (self.theta - v[:, t - 1]) * self.dt + self.sigma * torch.sqrt(
                    torch.maximum(v[:, t - 1], torch.tensor(0.0, device=self.device))) * dW_v[:, t - 1],
                torch.tensor(0.0, device=self.device))
            S[:, t] = S[:, t - 1] * torch.exp((self.r - 0.5 * v[:, t - 1]) * self.dt + torch.sqrt(
                torch.maximum(v[:, t - 1], torch.tensor(0.0, device=self.device))) * dW_S[:, t - 1])

        return S.cpu().numpy(), v.cpu().numpy()


# Paramètres de simulation
# Définition des paramètres de simulation
params = {
    "S0": 400,      # Prix initial du sous-jacent
    "r": 0.02,      # Taux sans risque
    "kappa": 3.0,   # Vitesse de réversion
    "theta": 0.04,  # Niveau de variance de long-terme
    "sigma": 0.5,   # Volatilité de la variance (vol of vol)
    "rho": -0.7,    # Corrélation entre Wt^S et Wt^v
    "v0": 0.04,     # Variance initiale
    "T": 1.0,       # Horizon de simulation (1 an)
    "dt": 1/252,    # Pas de temps journalier
    "n_paths": 1,   # Une seule trajectoire pour simplifier
    "device": "cpu" # Exécution sur CPU pour affichage
}

# Instanciation du simulateur Heston
simulator = HestonSimulator(**params)

# Simulation d'une trajectoire
S_traj, v_traj = simulator.simulate()


t = 100
S_t = S_traj[0, t]      # Prix du sous-jacent à t
v_t = v_traj[0, t]      # Variance à t
r_t = 0.02      # On suppose que le taux sans risque reste constant

# Valeurs à t+1
S_t1 = S_traj[0, t+1]
v_t1 = v_traj[0, t+1]
r_t1 = 0.02

# Définition des paramètres du forward start
k = 1.0   # Ratio de strike par rapport au sous-jacent
T0 = 0.0  # Temps actuel
T1 = 1.0  # Début de l'option
T2 = 2.0  # Expiration de l'option

# Calcul du prix de l'option Forward Start à t
fs_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                    kappa=params["kappa"], v0=v_t, theta=params["theta"],
                    sigma=params["sigma"], rho=params["rho"])
price_t = fs_t.heston_price().item()

# Calcul du prix de l'option Forward Start à t+1
fs_t1 = ForwardStart(S0=S_t1, k=k, T0=T0, T1=T1, T2=T2, r=r_t1,
                     kappa=params["kappa"], v0=v_t1, theta=params["theta"],
                     sigma=params["sigma"], rho=params["rho"])
price_t1 = fs_t1.heston_price().item()

print(f"Prix de l'option Forward Start à t={t}: {price_t:.4f}")
print(f"Prix de l'option Forward Start à t+1={t+1}: {price_t1:.4f}")

# Variations des facteurs de risque
dS = S_t1 - S_t    # Variation du sous-jacent
dV = v_t1 - v_t    # Variation de la variance (on suppose sigma = sqrt(variance))
dr = r_t1 - r_t    # Variation du taux sans risque (souvent 0)
dT = -params["dt"] # Le temps avance donc T diminue

# Création des objets ForwardStart à t et t+1
FS_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                    kappa=params["kappa"], v0=v_t, theta=params["theta"],
                    sigma=params["sigma"], rho=params["rho"])

FS_t1 = ForwardStart(S0=S_t1, k=k, T0=T0, T1=T1, T2=T2, r=r_t1,
                     kappa=params["kappa"], v0=v_t1, theta=params["theta"],
                     sigma=params["sigma"], rho=params["rho"])

# Calcul des Grecs à t
delta_t = FS_t.compute_first_order_greek("delta")
vega_t = FS_t.compute_first_order_greek("vega")
rho_t = FS_t.compute_first_order_greek("rho")
theta_t = FS_t.compute_first_order_greek("theta")

# Calcul du PnL expliqué
PnL_explained = delta_t * dS + vega_t * dV + rho_t * dr + theta_t * dT

# Affichage des résultats
print(f"PnL expliqué : {PnL_explained:.4f}")

PnL_inexpliqué = (price_t1 - price_t) - PnL_explained
print(f"PnL inexpliqué : {PnL_inexpliqué:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Paramètres
num_simulations = 1000  # Nombre d'itérations pour analyser la distribution du PnL inexpliqué
PnL_inexpliqué_list = []

for _ in range(num_simulations):
    # Simulation d'une trajectoire
    S_traj, v_traj = simulator.simulate()

    # Choix de t
    t = 100
    S_t, v_t = S_traj[0, t], v_traj[0, t]
    S_t1, v_t1 = S_traj[0, t + 1], v_traj[0, t + 1]
    r_t, r_t1 = 0.02, 0.02  # On garde r constant

    # Création des objets ForwardStart
    fs_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                        kappa=params["kappa"], v0=v_t, theta=params["theta"],
                        sigma=params["sigma"], rho=params["rho"])

    fs_t1 = ForwardStart(S0=S_t1, k=k, T0=T0, T1=T1, T2=T2, r=r_t1,
                         kappa=params["kappa"], v0=v_t1, theta=params["theta"],
                         sigma=params["sigma"], rho=params["rho"])

    # Calcul des prix
    price_t = fs_t.heston_price().item()
    price_t1 = fs_t1.heston_price().item()

    # Calcul du PnL total
    PnL_total = price_t1 - price_t

    # Calcul des Grecs
    delta_t = fs_t.compute_first_order_greek("delta")
    vega_t = fs_t.compute_first_order_greek("vega")
    rho_t = fs_t.compute_first_order_greek("rho")
    theta_t = fs_t.compute_first_order_greek("theta")

    # Calcul du PnL expliqué
    dS = S_t1 - S_t
    dV = v_t1 - v_t
    dr = r_t1 - r_t
    dT = -params["dt"]

    PnL_explained = delta_t * dS + vega_t * dV + rho_t * dr + theta_t * dT

    # Calcul du PnL inexpliqué
    PnL_inexpliqué = PnL_total - PnL_explained
    PnL_inexpliqué_list.append(PnL_inexpliqué)

# Conversion en array NumPy pour analyse
PnL_inexpliqué_list = np.array(PnL_inexpliqué_list)

# 🔹 Analyse statistique
mean_pnl_inexpliqué = np.mean(PnL_inexpliqué_list)
std_pnl_inexpliqué = np.std(PnL_inexpliqué_list)
shapiro_test = shapiro(PnL_inexpliqué_list[:500])  # Test de normalité sur 500 valeurs max

# 🔥 Affichage des résultats
print(f"Moyenne du PnL inexpliqué : {mean_pnl_inexpliqué:.6f}")
print(f"Écart-type du PnL inexpliqué : {std_pnl_inexpliqué:.6f}")
print(f"Test de normalité de Shapiro-Wilk (p-value) : {shapiro_test.pvalue:.6f}")

# 📊 Histogramme du PnL inexpliqué
plt.figure(figsize=(10, 5))
plt.hist(PnL_inexpliqué_list, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(mean_pnl_inexpliqué, color='red', linestyle='dashed', linewidth=2, label="Moyenne")
plt.xlabel("PnL inexpliqué")
plt.ylabel("Fréquence")
plt.title("Distribution du PnL inexpliqué sur 1000 simulations")
plt.legend()
plt.grid()
plt.show()
