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

            itm_index = 0  # k = 0.01
            atm_index = len(k_values) // 2  # k ≈ 1
            otm_index = -1  # k = 2

            itm_value = greek_list[itm_index]
            atm_value = greek_list[atm_index]
            otm_value = greek_list[otm_index]

            vertical_offset = 0.1 * (max(greek_list) - min(greek_list))
            horizontal_offset = 0.05

            plt.annotate("ITM", xy=(k_values[itm_index], itm_value),
                         xytext=(k_values[itm_index] + horizontal_offset, itm_value + vertical_offset),
                         arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)

            plt.annotate("ATM", xy=(k_values[atm_index], atm_value),
                         xytext=(k_values[atm_index] + horizontal_offset, atm_value + vertical_offset),
                         arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)

            plt.annotate("OTM", xy=(k_values[otm_index], otm_value),
                         xytext=(k_values[otm_index] - horizontal_offset, otm_value + vertical_offset),
                         arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)

            plt.gca().invert_xaxis()

            plt.show()
