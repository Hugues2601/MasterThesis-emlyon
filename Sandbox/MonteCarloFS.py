import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def MC_heston_forward_start():
    S0 = 100  # Initial stock price
    r = 0.04  # Risk-free rate
    T1 = 1  # Forward start date
    T2 = 3  # Maturity
    K_ratio = 0.3  # Strike as a ratio of S(T1)
    n = 252  # Number of time steps
    dt = T2 / n
    M = 10000  # Number of simulations
    S = np.zeros((M, n + 1))
    S[:, 0] = S0

    # Heston parameters
    Kappa = 4.14
    v0 = 0.01
    Theta = 0.01
    sigma = 0.07
    rho = -0.89
    vt = np.zeros((M, n + 1))
    vt[:, 0] = v0

    # Simulate paths
    for i in range(1, n + 1):
        Z = np.random.normal(0, 1, M)
        Y = np.random.normal(0, 1, M)
        Z2 = rho * Z + np.sqrt(1 - rho ** 2) * Y
        S[:, i] = S[:, i - 1] + S[:, i - 1] * (r * dt + np.sqrt(vt[:, i - 1]) * np.sqrt(dt) * Z)
        vt[:, i] = vt[:, i - 1] + (
                    Kappa * (Theta - vt[:, i - 1]) * dt + sigma * np.sqrt(vt[:, i - 1]) * np.sqrt(dt) * Z2)
        # Ensure variance remains positive
        vt[:, i] = np.maximum(vt[:, i], 0)

    # Calculate Payoff for forward start option
    t1_index = int(T1 / dt)
    ST1 = S[:, t1_index]  # Price at T1
    K = K_ratio * ST1  # Strike defined as a ratio of S(T1)
    Payoff = np.maximum(S[:, -1] - K, 0)  # Payoff of call option at T2
    Call_price = np.mean(Payoff) * np.exp(-r * T2)

    # Plot stock price paths
    plt.figure(figsize=(10, 6))
    for i in range(1000):
        plt.plot(S[i])
    plt.title("Monte Carlo Simulation for Forward Start Option")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.show()

    # Plot stochastic volatility paths
    plt.figure(figsize=(10, 6))
    for i in range(1000):
        plt.plot(vt[i])
    plt.title("Stochastic Volatility Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Volatility")
    plt.show()

    print(f"Forward Start Call Price: {Call_price}")


# Call the function
MC_heston_forward_start()


