import math


def gamma_function(alpha):
    """
    Computes the Gamma function Γ(α) using numerical integration.
    """

    def integrand(t):
        return math.exp(-t) * t ** (alpha - 1)

    def simpsons_rule(f, a, b, n=1000):
        """Numerical integration using Simpson's rule."""
        h = (b - a) / n
        integral = f(a) + f(b)
        for i in range(1, n, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n - 1, 2):
            integral += 2 * f(a + i * h)
        return (h / 3) * integral

    return simpsons_rule(integrand, 0, float('inf'))


def compute_Km(m):
    """
    Computes the constant K_m for the t-distribution.
    """
    return gamma_function((m + 1) / 2) / (math.sqrt(m * math.pi) * gamma_function(m / 2))


def t_distribution_probability(z, m):
    """
    Computes the probability F(z) for the t-distribution.
    """
    Km = compute_Km(m)

    def integrand(u):
        return (1 + u ** 2 / m) ** (-(m + 1) / 2)

    return Km * simpsons_rule(integrand, -float('inf'), z)


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
