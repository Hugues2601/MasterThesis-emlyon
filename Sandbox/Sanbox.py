# %%
"""
Created on Jan 2 2019
The Heston model and pricing of forward start options
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum
import scipy.optimize as optimize

# set i= imaginary number
i = complex(0.0, 1.0)


# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


def CallPutOptionPriceCOSMthd_FrwdStart(cf, CP, r, T1, T2, K, N, L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)

    tau = T2 - T1
    # reshape K to a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    # Adjust strike
    K = K + 1.0

    # assigning i=sqrt(-1)
    i = complex(0.0, 1.0)
    x0 = np.log(1.0 / K)

    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    # sumation from k = 0 to k=N-1
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a);

    # Determine coefficients for Put Prices
    H_k = CallPutCoefficients(CP, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * T2) * K * np.real(mat.dot(temp))
    return value


# Determine coefficients for Put and Call Prices
def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k


def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)), 2.0))
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi
                                                                       * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                                         (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k
                                                                                           * np.pi * (c - a) / (
                                                                                                       b - a)) * np.exp(
        c)
    chi = chi * (expr1 + expr2)

    value = {"chi": chi, "psi": psi}
    return value


# Forward start Black-Scholes option price
def BS_Call_Option_Price_FrwdStart(K, sigma, T1, T2, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])
    K = K + 1.0
    tau = T2 - T1
    d1 = (np.log(1.0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    value = np.exp(-r * T1) * st.norm.cdf(d1) - st.norm.cdf(d2) * K * np.exp(-r * T2)
    return value


# Implied volatility for the forward start call option
def ImpliedVolatility_FrwdStart(marketPrice, K, T1, T2, r):
    # To determine initial volatility we interpolate define a grid for sigma
    # and interpolate on the inverse
    sigmaGrid = np.linspace(0, 2, 200)
    optPriceGrid = BS_Call_Option_Price_FrwdStart(K, sigmaGrid, T1, T2, r)
    sigmaInitial = np.interp(marketPrice, optPriceGrid, sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))

    # Use determined input for the local-search (final tuning)
    func = lambda sigma: np.power(BS_Call_Option_Price_FrwdStart(K, sigma, T1, T2, r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol


def ChFHestonModelForwardStart(r, T1, T2, kappa, gamma, vbar, v0, rho):
    i = complex(0.0, 1.0)
    tau = T2 - T1
    D1 = lambda u: np.sqrt(np.power(kappa - gamma * rho * i * u, 2) + (u * u + i * u) * gamma * gamma)
    g = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))
    C = lambda u: (1.0 - np.exp(-D1(u) * tau)) / (gamma * gamma * (1.0 - g(u) * np.exp(-D1(u) * tau))) \
                  * (kappa - gamma * rho * i * u - D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A = lambda u: r * i * u * tau + kappa * vbar * tau / gamma / gamma * (kappa - gamma * rho * i * u - D1(u)) \
                  - 2 * kappa * vbar / gamma / gamma * np.log((1.0 - g(u) * np.exp(-D1(u) * tau)) / (1.0 - g(u)))
    c_bar = lambda t1, t2: gamma * gamma / (4.0 * kappa) * (1.0 - np.exp(-kappa * (t2 - t1)))
    delta = 4.0 * kappa * vbar / gamma / gamma
    kappa_bar = lambda t1, t2: 4.0 * kappa * v0 * np.exp(-kappa * (t2 - t1)) / (
                gamma * gamma * (1.0 - np.exp(-kappa * (t2 - t1))))
    term1 = lambda u: A(u) + C(u) * c_bar(0.0, T1) * kappa_bar(0.0, T1) / (1.0 - 2.0 * C(u) * c_bar(0.0, T1))
    term2 = lambda u: np.power(1.0 / (1.0 - 2.0 * C(u) * c_bar(0.0, T1)), 0.5 * delta)
    cf = lambda u: np.exp(term1(u)) * term2(u)
    return cf


def mainCalculation():
    CP = OptionType.CALL  # Type d'option : CALL ou PUT
    r = 0.04  # Taux d'intérêt

    # Paramètres des maturités
    T1 = 2.0  # Temps jusqu'au début de la période forward
    T2 = 3.0  # Temps jusqu'à l'échéance finale

    # Paramètres de strike
    K = np.array([1])  # Strike (peut être un tableau si plusieurs strikes)
    N = 500  # Nombre de termes dans l'expansion COS
    L = 10   # Taille du domaine de troncature

    # Paramètres du modèle de Heston
    kappa = 2.1
    gamma = 0.1
    vbar = 0.03
    rho = -0.2
    v0 = 0.05

    # Calcul de la fonction caractéristique pour le modèle Heston
    cf = ChFHestonModelForwardStart(r, T1, T2, kappa, gamma, vbar, v0, rho)

    # Calcul du prix de l'option forward start avec la méthode COS
    valCOS = CallPutOptionPriceCOSMthd_FrwdStart(cf, CP, r, T1, T2, K, N, L)

    print(f"Prix de l'option forward start pour K={K.flatten()} : {valCOS.flatten()}")


mainCalculation()
