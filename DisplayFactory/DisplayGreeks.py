import torch
from HestonModel.ForwardStart import ForwardStart
import matplotlib.pyplot as plt

class DisplayGreeks:
    def __init__(self):
        pass

    def display_all_greeks(self):

        k_values = torch.linspace(0.01, 2, 500)
        deltas = []
        for k in k_values:
            FS = ForwardStart(100.0, k, 0.0, 1.0, 3.0, 0.05, 2, 0.04, 0.04, 0.2, -0.7)
            delta = FS.compute_first_order_greek("vega")
            deltas.append(delta)
        plt.plot(k_values, deltas)
        plt.show()