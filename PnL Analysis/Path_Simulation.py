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
        def __init__(self, S0, r, kappa, theta, sigma, rho, v0, T, dt, n_paths=1, device="cuda"):
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
            dW_v = self.rho * dW_S + torch.sqrt(torch.tensor(1 - self.rho ** 2)) * torch.randn(self.n_paths,
                                                                                               self.n_steps - 1,
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

    def price_forward_start(self, S_t, v_t, r_t, k, T0, T1, T2):
        fs = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                          kappa=self.params["kappa"], v0=v_t, theta=self.params["theta"],
                          sigma=self.params["sigma"], rho=self.params["rho"])
        return fs.heston_price().item()

    def compute_pnl_analysis(self, num_simulations=1000):
        PnL_inexpliqué_list = torch.zeros(num_simulations, device=self.device)
        k, T0, T1, T2 = 1.0, 0.0, 1.0, 2.0
        r_t = 0.02

        for i in range(num_simulations):
            S_traj, v_traj = self.simulator.simulate()
            t = 100
            S_t, v_t = S_traj[0, t], v_traj[0, t]
            S_t1, v_t1 = S_traj[0, t + 1], v_traj[0, t + 1]

            price_t = torch.tensor(self.price_forward_start(S_t, v_t, r_t, k, T0, T1, T2), device=self.device)
            price_t1 = torch.tensor(self.price_forward_start(S_t1, v_t1, r_t, k, T0, T1, T2), device=self.device)

            dS, dV, dr, dT = S_t1 - S_t, v_t1 - v_t, 0.0, -self.params["dt"]
            FS_t = ForwardStart(S0=S_t, k=k, T0=T0, T1=T1, T2=T2, r=r_t,
                                kappa=self.params["kappa"], v0=v_t, theta=self.params["theta"],
                                sigma=self.params["sigma"], rho=self.params["rho"])

            delta_t = torch.tensor(FS_t.compute_first_order_greek("delta"), device=self.device)
            vega_t = torch.tensor(FS_t.compute_first_order_greek("vega"), device=self.device)
            rho_t = torch.tensor(FS_t.compute_first_order_greek("rho"), device=self.device)
            theta_t = torch.tensor(FS_t.compute_first_order_greek("theta"), device=self.device)

            PnL_explained = delta_t * dS + vega_t * dV + rho_t * dr + theta_t * dT
            PnL_total = price_t1 - price_t
            PnL_inexpliqué_list[i] = PnL_total - PnL_explained

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


# Utilisation de la classe
params = {
    "S0": 400, "r": 0.02, "kappa": 3.0, "theta": 0.04, "sigma": 0.5, "rho": -0.7, "v0": 0.04, "T": 1.0,
    "dt": 1 / 252, "n_paths": 1, "device": "cuda"
}
analysis = ForwardStartAnalysis(params)
PnL_inexpliqué_list = analysis.compute_pnl_analysis(num_simulations=1000)
analysis.analyze_results(PnL_inexpliqué_list)
