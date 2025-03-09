from BlackScholes.VanillaBlackScholes import implied_vol
from DisplayFactory.DisplayManager import DisplayManager
from DataRetriever import get_yfinance_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


class DisplayVolSurface:
    def __init__(self, ticker):
        self.ticker = ticker

    def _dataProcessing(self):
        calls_list, lastPrice, timetomaturity, impliedVolatility, strike, spot_price, risk_free_rate  = get_yfinance_data(self.ticker)
        vol = implied_vol(strike, timetomaturity, lastPrice)
        return timetomaturity, vol, strike

    def display_comparison(self, other_iv):
        # Processer les données pour la première surface
        timetomaturity, impliedVolatility, strike = self._dataProcessing()
        df1 = pd.DataFrame({
            'strike': strike,
            'timetomaturity': timetomaturity,
            'impliedVolatility': impliedVolatility
        })

        df1_cleaned = df1.groupby(['strike', 'timetomaturity'], as_index=False).mean()
        strike_clean = df1_cleaned['strike'].values
        timetomaturity_clean = df1_cleaned['timetomaturity'].values
        impliedVolatility_clean = df1_cleaned['impliedVolatility'].values

        # Convertir other_iv en DataFrame pour le nettoyage
        df2 = pd.DataFrame({
            'strike': strike_clean,
            'timetomaturity': timetomaturity_clean,
            'impliedVolatility': other_iv  # Les IV fournies
        })

        df2_cleaned = df2.groupby(['strike', 'timetomaturity'], as_index=False).mean()
        impliedVolatility2_clean = df2_cleaned['impliedVolatility'].values

        # Créer une grille régulière pour les deux surfaces
        unique_strikes = np.unique(strike_clean)
        unique_times = np.unique(timetomaturity_clean)
        X, Y = np.meshgrid(
            np.linspace(unique_strikes.min(), unique_strikes.max(), 100),
            np.linspace(unique_times.min(), unique_times.max(), 100)
        )

        # Interpoler les données de volatilité pour les deux surfaces
        Z1 = griddata(
            (strike_clean, timetomaturity_clean),
            impliedVolatility_clean,
            (X, Y),
            method='linear'
        )
        Z2 = griddata(
            (strike_clean, timetomaturity_clean),
            impliedVolatility2_clean,
            (X, Y),
            method='linear'
        )

        # Tracer les deux surfaces
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Première surface
        surf1 = ax.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none', alpha=0.7)

        # Deuxième surface
        surf2 = ax.plot_surface(X, Y, Z2, cmap='plasma', edgecolor='none', alpha=0.7)

        # Ajouter des étiquettes et une barre de couleur
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity (Years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Volatility Surface Comparison')
        fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label='Surface 1 (Self)')
        fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, label='Surface 2 (Other)')

        plt.legend(['Surface 1 (Self)', 'Surface 2 (Other)'])
        plt.show()
