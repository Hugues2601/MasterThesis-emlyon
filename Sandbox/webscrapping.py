import torch
import matplotlib.pyplot as plt
from BlackScholes.VanillaBlackScholes import VanillaBlackScholes
from DataRetriever import get_yfinance_data
from HestonModel.Vanilla import VanillaHestonPrice

# Configurer Torch pour l'optimisation
class CONFIG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fonction pour calculer la volatilité implicit

df, lastPrice, timetomaturity, impliedVolatility, strike, spot_price = get_yfinance_data("AMZN")

print(impliedVolatility)
