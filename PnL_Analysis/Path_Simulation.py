import matplotlib.pyplot as plt
from HestonModel.ForwardStart import ForwardStart
import numpy as np
import scipy.stats as stats
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
        S = torch.zeros(n_paths, self.n_steps, device=self.device)
        v = torch.zeros(n_paths, self.n_steps, device=self.device)
        dt_tensor = torch.full((n_paths, self.n_steps), self.dt, device=self.device)

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        dW_v = torch.randn(n_paths, self.n_steps - 1, device=self.device) * torch.sqrt(
            torch.tensor(self.dt, device=self.device))
        dZ = torch.randn(n_paths, self.n_steps - 1, device=self.device) * torch.sqrt(
            torch.tensor(self.dt, device=self.device))
        dW_S = self.rho * dW_v + torch.sqrt(torch.tensor(1 - self.rho ** 2, device=self.device)) * dZ

        # Valeurs dynamiques initiales
        kappa = torch.full((n_paths,), self.kappa, device=self.device)
        theta = torch.full((n_paths,), self.theta, device=self.device)
        sigma = torch.full((n_paths,), self.sigma, device=self.device)

        for i in range(1, self.n_steps):
            # AR(1) update avec clamping
            kappa = kappa + 0.1 * (self.kappa - kappa) + 0.05 * torch.randn(n_paths, device=self.device)
            theta = theta + 0.05 * (self.theta - theta) + 0.005 * torch.randn(n_paths, device=self.device)
            sigma = sigma + 0.1 * (self.sigma - sigma) + 0.02 * torch.randn(n_paths, device=self.device)

            kappa = torch.clamp(kappa, min=1e-4, max=10)
            theta = torch.clamp(theta, min=1e-4, max=1)
            sigma = torch.clamp(sigma, min=1e-4, max=5)

            # Racine de v (clamp pour éviter NaN)
            v_prev_clamped = torch.clamp(v[:, i - 1], min=1e-8)
            sqrt_v = torch.sqrt(v_prev_clamped)

            # Update S
            S[:, i] = S[:, i - 1] + S[:, i - 1] * (self.r * self.dt + sqrt_v * dW_S[:, i - 1])

            # Update v
            v[:, i] = v[:, i - 1] + kappa * (theta - v[:, i - 1]) * self.dt + sigma * sqrt_v * dW_v[:, i - 1]

            # Clamp v après update
            v[:, i] = torch.clamp(v[:, i], min=1e-8)

        return S, v, dt_tensor


    def plot_paths(self, S, v, num_paths=1000):
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


def pnl_analysis(S0, k, r, kappa, v0, theta, sigma, rho, T0, T1, T2):
    simulator = HestonSimulator(S0=S0, r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0, T=4.0, dt=1/252)
    S_paths, v_paths, dt_path = simulator.simulate(n_paths=15_000)
    simulator.plot_paths(S_paths, v_paths)

    forward_start = ForwardStart(S0=S0, k=k, T0=T0, T1=T1, T2=T2,
                                 r=r, kappa=kappa, v0=v0, theta=theta,
                                 sigma=sigma, rho=rho)

    prices_t, prices_t1, pnl_tot = forward_start.compute_heston_prices(S_paths, v_paths, t=100)



    explained_pnl = forward_start.compute_explained_pnl(S_paths, v_paths, t=100, dt=1/252, dt_path=dt_path)

    pnl_inex = pnl_tot - explained_pnl



    # Convertir en NumPy
    pnl_inex_np = pnl_inex.detach().cpu().numpy()
    pn_tot_np = pnl_tot.detach().cpu().numpy()
    print("Ratio PnL inexpliqué / PnL total :", pnl_inex_np.mean() / pn_tot_np.mean())

    # Statistiques descriptives
    mean = pnl_inex_np.mean()
    std_dev = pnl_inex_np.std()
    skewness = stats.skew(pnl_inex_np)
    kurtosis = stats.kurtosis(pnl_inex_np)

    print(f"Mean: {mean:.4f}, Std Dev: {std_dev:.4f}")
    print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

    # Tracé de l'histogramme et de la densité estimée
    plt.figure(figsize=(8, 5))
    plt.hist(pnl_inex_np, bins=30, density=True, alpha=0.6, color='b', label="Histogram")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2, label="Normal Fit")
    plt.title("Histogram and Normal Fit")
    plt.legend()
    plt.show()

    # Test de normalité de Shapiro-Wilk
    shapiro_test = stats.shapiro(pnl_inex_np)
    print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")

    # Test d'Anderson-Darling
    anderson_test = stats.anderson(pnl_inex_np, dist='norm')
    print(f"Anderson-Darling Test: Statistic={anderson_test.statistic:.4f}, Critical Values={anderson_test.critical_values}, Significance Levels={anderson_test.significance_level}")

    # Test de Kolmogorov-Smirnov
    ks_test = stats.kstest(pnl_inex_np, 'norm', args=(mean, std_dev))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_test.statistic:.4f}, p-value={ks_test.pvalue:.4f}")

    # Q-Q Plot pour visualiser la normalité
    plt.figure(figsize=(6, 6))
    stats.probplot(pnl_inex_np, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()