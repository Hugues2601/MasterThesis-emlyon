from DisplayFactory.DisplayManager import DisplayManager
from DataRetriever import get_yfinance_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import griddata


class DisplayVolSurface:
    def __init__(self):
        pass

    def _dataProcessing(self):
        calls_list, lastPrice, timetomaturity, impliedVolatility, strike, spot_price  = get_yfinance_data("XOM")
        return timetomaturity, impliedVolatility, strike

    def display(self):
        # Créer un DataFrame pour regrouper et nettoyer les données
        timetomaturity, impliedVolatility, strike = self._dataProcessing()
        df = pd.DataFrame({
            'strike': strike,
            'timetomaturity': timetomaturity,
            'impliedVolatility': impliedVolatility
        })

        # Retirer les doublons en prenant la moyenne des volatilités
        df_cleaned = df.groupby(['strike', 'timetomaturity'], as_index=False).mean()

        # Extraire les colonnes nettoyées
        strike_clean = df_cleaned['strike'].values
        timetomaturity_clean = df_cleaned['timetomaturity'].values
        impliedVolatility_clean = df_cleaned['impliedVolatility'].values

        # Créer une grille régulière pour la surface
        unique_strikes = np.unique(strike_clean)
        unique_times = np.unique(timetomaturity_clean)
        X, Y = np.meshgrid(unique_strikes, unique_times)

        # Interpoler les données de volatilité sur la grille
        Z = griddata(
            (strike_clean, timetomaturity_clean),  # Points existants
            impliedVolatility_clean,  # Volatilités correspondantes
            (X, Y),  # Points de la nouvelle grille
            method='linear'  # Méthode d'interpolation
        )

        # Tracer la surface de volatilité
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

        # Ajouter des étiquettes et une barre de couleur
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity (Years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Volatility Surface')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        plt.show()


DisplayVolSurface().display()