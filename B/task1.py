import numpy as np
import matplotlib.pyplot as plt

# Constants
beta = 1  # Inverse temperature (1/kT)
mu = 2.0  # Chemical potential
M = 10    # Number of energy levels
epsilon = np.array([n for n in range(1, M + 1)])  # Energy levels ε, 2ε, 3ε, ..., Mε

# Compute grand partition function
Z_grand = np.prod(1 + np.exp(-beta * (epsilon - mu)))

# Compute Fermi-Dirac occupation probability
fermi_dirac = 1 / (1 + np.exp(beta * (epsilon - mu)))

# Plot the Fermi-Dirac distribution
plt.figure(figsize=(8, 5))
plt.plot(epsilon, fermi_dirac, 'o-', label=r"$f(\epsilon) = \frac{1}{e^{\beta (\epsilon - \mu)} + 1}$")
plt.axvline(mu, linestyle="--", color="red", label=r"$\mu$ (Chemical Potential)")
plt.xlabel("Energy Level (ε)")
plt.ylabel("Occupation Probability")
plt.title("Fermi-Dirac Distribution")
plt.legend()
plt.grid()
plt.show()

# Print the grand partition function
print(f"Grand Partition Function: Z = {Z_grand:.5f}")