import math

def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    This function implements the Secant method to find the root of an equation. The equation should be written in the form
    fcn = 0 such that when the correct value of x is selected, the function evaluates to zero (or very close to it).

    :param fcn: The function for which we want to find the root.
    :param x0: Initial guess for the root.
    :param x1: Another initial guess for the root (should be different from x0).
    :param maxiter: Maximum number of iterations before stopping.
    :param xtol: Convergence tolerance; stops if |x_new - x_previous| < xtol.
    :return: Tuple containing (final estimate of the root, number of iterations).
    """
    x_diff = abs(xtol) + 1  # Ensure loop starts
    iter_count = 0
    f0 = fcn(x0)
    f1 = fcn(x1)

    while iter_count < maxiter and abs(x_diff) > abs(xtol):
        f1 = fcn(x1)
        if f1 - f0 == 0:  # Prevent division by zero
            return (x1, iter_count)

        x_new = x1 - f1 * ((x1 - x0) / (f1 - f0))
        x_diff = x_new - x1
        x0, x1 = x1, x_new
        f0 = f1
        iter_count += 1

    return (x1, iter_count)

def normal_cdf_integral(c, mu, sigma):
    """
    Computes the probability P(x < c | mu, sigma) using an approximation of the normal CDF.
    """
    return 0.5 * (1 + math.erf((c - mu) / (sigma * math.sqrt(2))))

def find_c_for_probability(target_p, mu, sigma):
    """
    Uses the Secant method to find c such that P(x < c | mu, sigma) = target_p.
    """
    def func(c):
        return normal_cdf_integral(c, mu, sigma) - target_p

    c_root, _ = Secant(func, mu - 3 * sigma, mu + 3 * sigma)  # Initial guesses
    return c_root

def main():
    print("Choose an option:")
    print("1: Specify c to compute P")
    print("2: Specify P to compute c")
    choice = int(input("Enter 1 or 2: "))

    mu = float(input("Enter mean (mu): "))
    sigma = float(input("Enter standard deviation (sigma): "))

    if choice == 1:
        c = float(input("Enter c: "))
        P = normal_cdf_integral(c, mu, sigma)
        print(f"P(x < {c} | {mu}, {sigma}) = {P:.6f}")
    elif choice == 2:
        P = float(input("Enter probability P: "))
        c = find_c_for_probability(P, mu, sigma)
        print(f"Value of c such that P(x < c | {mu}, {sigma}) = {P:.6f} is c = {c:.6f}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
