from HestonModel.ForwardStart import ForwardStart
import numpy as np
import matplotlib.pyplot as plt

def plot_sensitivity(param_to_vary: str):
    assert param_to_vary in ["kappa", "v0", "theta", "sigma", "rho"], "Paramètre inconnu"

    # Paramètres calibrés par défaut
    calibrated_params = {
        'kappa': 2.41300630569458,
        'v0': 0.029727613553404808,
        'theta': 0.04138144478201866,
        'sigma': 0.3084869682788849,
        'rho': -0.8905978202819824
    }

    # Plages à tester pour chaque paramètre
    param_ranges = {
        'kappa': np.linspace(0.1, 8, 100),
        'v0': np.linspace(0.01, 0.4, 100),
        'theta': np.linspace(0.01, 0.4, 100),
        'sigma': np.linspace(0.05, 0.9, 100),
        'rho': np.linspace(-0.9, 0.9, 100)
    }

    price_list = []

    for value in param_ranges[param_to_vary]:
        params = calibrated_params.copy()
        params[param_to_vary] = value

        price = ForwardStart(
            S0=5667.5,
            k=1.0,
            r=0.03927,
            T0=0.0,
            T1=0.5,
            T2=1,
            kappa=params["kappa"],
            v0=params["v0"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"]
        ).heston_price()

        price_list.append(price.item())

    plt.plot(param_ranges[param_to_vary], price_list)
    plt.title(f"Sensibilité du prix à {param_to_vary}")
    plt.xlabel(param_to_vary)
    plt.ylabel("Prix Forward Start")
    plt.grid(True)
    plt.show()

    return price_list



# plot_sensitivity("kappa")
# plot_sensitivity("v0")
# plot_sensitivity("theta")
# plot_sensitivity("sigma")
# plot_sensitivity("rho")








from HestonModel.ForwardStart import ForwardStart
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_all_sensitivities():
    calibrated_params = {
        'kappa': 2.41300630569458,
        'v0': 0.029727613553404808,
        'theta': 0.04138144478201866,
        'sigma': 0.3084869682788849,
        'rho': -0.8905978202819824
    }

    param_ranges = {
        'kappa': np.linspace(0.1, 8, 100),
        'v0': np.linspace(0.01, 0.4, 100),
        'theta': np.linspace(0.01, 0.4, 100),
        'sigma': np.linspace(0.05, 0.9, 100),
        'rho': np.linspace(-0.9, 0.9, 100)
    }

    greek_labels = {
        'kappa': r'$\kappa$',
        'v0': r'$v_0$',
        'theta': r'$\theta$',
        'sigma': r'$\sigma$',
        'rho': r'$\rho$'
    }

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    axes = []
    # Trois premiers en haut (pleine largeur)
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

    # Deux du bas : les placer au centre (colonnes 1 et 2 uniquement)
    ax_bottom1 = fig.add_subplot(gs[1, 1])
    ax_bottom2 = fig.add_subplot(gs[1, 2])
    axes.extend([ax_bottom1, ax_bottom2])

    param_names = list(param_ranges.keys())

    for i, param in enumerate(param_names):
        price_list = []
        for value in param_ranges[param]:
            params = calibrated_params.copy()
            params[param] = value

            price = ForwardStart(
                S0=5667.5,
                k=1.0,
                r=0.03927,
                T0=0.0,
                T1=1.0,
                T2=2.0,
                kappa=params["kappa"],
                v0=params["v0"],
                theta=params["theta"],
                sigma=params["sigma"],
                rho=params["rho"]
            ).heston_price()

            price_list.append(price.item())

        axes[i].plot(param_ranges[param], price_list, color='black')
        axes[i].set_title(f"Sensitivity to {greek_labels[param]}", fontsize=13)
        axes[i].set_xlabel(greek_labels[param], fontsize=12)
        axes[i].set_ylabel("Forward Start Price")
        axes[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()


# plot_all_sensitivities()


    calibrated_params = {
        'kappa': 2.41300630569458,
        'v0': 0.029727613553404808,
        'theta': 0.04138144478201866,
        'sigma': 0.3084869682788849,
        'rho': -0.8905978202819824
    }

maturity_list =
price_list = []
price = ForwardStart(
    S0=5667.65,
    k=1.0,
    r=0.03927,
    T0=0.0,
    T1=1.0,
    T2=2.0,
    kappa=params["kappa"],
    v0=params["v0"],
    theta=params["theta"],
    sigma=params["sigma"],
    rho=params["rho"]
).heston_price()



