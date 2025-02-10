import scipy.stats as stats
import numpy as np
import scipy.special as sp
from scipy.integrate import quad


def gamma_function(alpha):
    """
    Computes the Gamma function \u0393(\u03b1) using numerical integration.
    """
    integrand = lambda t: np.exp(-t) * t ** (alpha - 1)
    result, _ = quad(integrand, 0, np.inf)
    return result


def compute_Km(m):
    """
    Computes the constant K_m for the t-distribution.
    """
    return sp.gamma((m + 1) / 2) / (np.sqrt(m * np.pi) * sp.gamma(m / 2))


def t_distribution_probability(z, m):
    """
    Computes the probability F(z) for the t-distribution.
    """
    Km = compute_Km(m)
    integrand = lambda u: (1 + u ** 2 / m) ** (-(m + 1) / 2)
    result, _ = quad(integrand, -np.inf, z)
    return Km * result


def main():
    """
    Main function to prompt user input and compute t-distribution probabilities.
    """
    print("T-Distribution Probability Calculator")

    for _ in range(3):  # Prompt for three different z values
        m = int(input("Enter degrees of freedom (7, 11, or 15): "))
        if m not in [7, 11, 15]:
            print("Invalid degrees of freedom. Choose 7, 11, or 15.")
            continue
        z = float(input("Enter z value: "))
        probability = t_distribution_probability(z, m)
        print(f"F({z}) for {m} degrees of freedom = {probability:.6f}")


if __name__ == "__main__":
    main()
