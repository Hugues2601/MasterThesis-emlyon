import torch
from HestonModel.ForwardStart import ForwardStart
import matplotlib.pyplot as plt
from DisplayFactory.DisplayManager import DisplayManager

class DisplayGreeks(DisplayManager):
    def __init__(self, S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho):
        super().__init__(S0, k, T0, T1, T2, r, kappa, v0, theta, sigma, rho)

    def _dataProcessing(self, greek_name):
        k_values = torch.linspace(0.01, 2, 500)
        greek_list = []
        for k in k_values:
            FS = ForwardStart(self.S0, k, self.T0, self.T1, self.T2, self.r, self.kappa, self.v0, self.theta, self.sigma, self.rho)
            greek = FS.compute_first_order_greek(greek_name)
            greek_list.append(greek)
        return greek_list

    def display(self):
        k_values = torch.linspace(0.01, 2, 500)
        greeks = ["delta", "vega", "theta", "rho"]

        for greek in greeks:
            greek_list = self._dataProcessing(greek)

            plt.plot(k_values, greek_list, label=greek)
            plt.title(greek)
            plt.xlabel("k")
            plt.ylabel(f"{greek} value")
            plt.grid(True)

            # Highlighting zones
            plt.axvspan(0, 1, color='green', alpha=0.2, label="ITM Zone")  # ITM: k in [0, 1)
            plt.axvline(1, color='blue', linestyle='--', label="ATM")  # ATM: k = 1
            plt.axvspan(1, 2, color='red', alpha=0.2, label="OTM Zone")  # OTM: k in (1, 2]

            # Add legend for zones
            plt.legend(loc="upper right")

            plt.show()

