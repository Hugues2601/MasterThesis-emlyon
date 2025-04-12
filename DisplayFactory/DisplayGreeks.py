import torch
from HestonModel.ForwardStart import ForwardStart
import matplotlib.pyplot as plt
from DisplayFactory.DisplayManager import DisplayManager

class DisplayGreeks(DisplayManager):
    def __init__(self, S0, r, kappa, v0, theta, sigma, rho):
        self.S0 = S0
        self.r = r
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    def _dataProcessing(self, greek_name, T0, T1, T2):
        k_values = torch.linspace(0.01, 2, 500)
        greek_list = []
        for k in k_values:
            FS = ForwardStart(self.S0, k, T0, T1, T2, self.r,
                              self.kappa, self.v0, self.theta, self.sigma, self.rho)
            greek = FS.compute_greek(greek_name)
            greek_list.append(greek)
        return k_values, greek_list

    def display(self):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        greeks = ["delta", "vega", "theta", "vanna", "volga"]
        maturities = [
            (0.0, 0.5, 1.0),   # noir
            (0.0, 0.75, 1.5),  # rouge
            (0.0, 1.0, 2.0)    # bleu
        ]
        colors = ['black', '#d08770', '#8fbcbb']
        labels = ['T1=0.5, T2=1.0', 'T1=0.75, T2=1.5', 'T1=1.0, T2=2.0']

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
        axes = axes.flatten()

        for idx, greek in enumerate(greeks):
            ax = axes[idx]
            for (T0, T1, T2), color in zip(maturities, colors):
                k_vals, greek_vals = self._dataProcessing(greek, T0, T1, T2)
                ax.plot(k_vals, greek_vals, color=color)
            ax.set_title(greek)
            ax.set_xlabel("k")
            ax.set_ylabel(f"{greek} value")
            ax.grid(True)

        # Légende dans la dernière case (en bas à droite)
        legend_ax = axes[-1]
        legend_ax.axis('off')  # On désactive les axes
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=label)
                           for color, label in zip(colors, labels)]
        legend_ax.legend(
            handles=legend_elements,
            loc='center',
            frameon=True,
            fontsize=16,  # Agrandit le texte
            handlelength=4,  # Allonge les lignes de couleur
            handletextpad=3  # Espace entre ligne et texte
        )

        plt.tight_layout()
        plt.show()


# calibrated_params = {
#         'kappa': 2.41300630569458,
#         'v0': 0.029727613553404808,
#         'theta': 0.04138144478201866,
#         'sigma': 0.3084869682788849,
#         'rho': -0.8905978202819824
#     }
# calibrated_params = {
#     'kappa': 2.41300630569458,
#     'v0': 0.029727613553404808,
#     'theta': 0.04138144478201866,
#     'sigma': 0.3084869682788849,
#     'rho': -0.8905978202819824
# }
#
# display = DisplayGreeks(
#     S0=5667.65, r=0.03927,
#     kappa=calibrated_params['kappa'],
#     v0=calibrated_params["v0"],
#     theta=calibrated_params["theta"],
#     sigma=calibrated_params["sigma"],
#     rho=calibrated_params["rho"]
# )
# display.display()
#
#
# price = ForwardStart(S0=5667.65, k=1.0, T0=0.0, T1=.75, T2=1.5, r=0.03927, kappa=calibrated_params['kappa'], v0=calibrated_params["v0"], theta=calibrated_params["theta"], sigma=calibrated_params["sigma"], rho=calibrated_params["rho"]).heston_price()
# print(price.item())
#
#
#
#
#
# pastel_colors = [
#     "black",
#     "#8fbcbb",
#     "#a3be8c",
#     "#d08770",
#     "#e09ec7",
#     "#b48ead"
# ]