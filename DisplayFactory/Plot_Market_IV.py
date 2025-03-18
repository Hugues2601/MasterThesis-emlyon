def plot_implied_volatility(strike, implied_volatility, timetomaturity):
    """
    Plots implied volatility against strike price for different maturities.

    Parameters:
    - strike: List or array of strike prices.
    - implied_volatility: List or array of implied volatilities.
    - timetomaturity: List or array of time to maturity (same length as strike and implied_volatility).

    Returns:
    - A plot of implied volatility vs. strike price, grouped by maturity.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))

    # Convert to numpy array for easier processing
    strike = np.array(strike)
    implied_volatility = np.array(implied_volatility)
    timetomaturity = np.array(timetomaturity)

    # Get unique maturities and plot each one
    unique_maturities = np.unique(timetomaturity)
    for maturity in sorted(unique_maturities):
        mask = timetomaturity == maturity
        plt.plot(strike[mask], implied_volatility[mask], label=f"T={maturity:.2f}")

    # Labels and title
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title("Market Implied Volatility Smile")
    plt.legend(title="Maturity (Years)", loc="upper right")
    plt.grid(True)

    plt.show()
