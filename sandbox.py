import numpy as np
import pandas as pd

# Paramètres du modèle Heston
kappa = 2.4130
theta = 0.0414
xi = 0.3085
rho = -0.8910
v0 = 0.0297
S0 = 5667.5
r = 0.03927
dt = 1 / 252
n_paths = 100_000

# Bruits
Z1 = np.random.randn(n_paths)
Z2 = np.random.randn(n_paths)
Z_v = Z1
Z_s = rho * Z1 + np.sqrt(1 - rho**2) * Z2

# Moments conditionnels de v_{t+dt}
m = theta + (v0 - theta) * np.exp(-kappa * dt)
s2 = (v0 * xi**2 * np.exp(-kappa * dt) * (1 - np.exp(-kappa * dt))) / kappa + \
     theta * xi**2 * (1 - np.exp(-kappa * dt))**2 / (2 * kappa)
psi = s2 / m**2

v_next = np.zeros(n_paths)

# Cas 1 : psi <= 1.5
mask1 = psi <= 1.5
if np.any(mask1):
    b2 = 2 / psi - 1 + np.sqrt(2 / psi * (2 / psi - 1))
    a = m / (1 + b2)
    v_next[mask1] = a * (np.sqrt(b2) + Z_v[mask1])**2

# Cas 2 : psi > 1.5
mask2 = ~mask1
if np.any(mask2):
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m
    u = np.random.rand(mask2.sum())
    v_temp = np.zeros_like(u)
    v_temp[u > p] = -np.log((1 - u[u > p]) / (1 - p)) / beta
    v_next[mask2] = v_temp

# Simulation de S_{t+dt}
S_next = S0 * np.exp((r - 0.5 * v0) * dt + np.sqrt(v0 * dt) * Z_s)

# Résumé
print("Min v :", np.min(v_next))
print("Max v :", np.max(v_next))
print('Min S')

# Tu peux aussi exporter les résultats si besoin
results = pd.DataFrame({"S": S_next, "v": v_next})
results.to_csv("heston_qe_simulation.csv", index=False)
S_list=results["S"].tolist()
print(len(S_list))
print(results.head())



