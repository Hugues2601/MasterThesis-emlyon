import yfinance as yf
import matplotlib.pyplot as plt

# Télécharger les données du VIX
vix = yf.download("^VIX", start="2006-01-01", end="2023-01-01")

# Créer le graphique
plt.figure(figsize=(10, 5))
plt.plot(vix.index, vix["Close"], color="black")

# Améliorations des polices
plt.xlabel("Year", fontsize=16)
plt.ylabel("VIX Value", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True)
plt.tight_layout()
plt.show()
