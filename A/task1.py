import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Constants
m = 1       # Mass
omega = 1   # Frequency
hbar = 1    # Planck's constant (set to 1 in natural units)

# Define energy levels
E_min, E_max = 0.1, 10.0  # Energy range
num_points = 100          # Number of energy points
E_values = np.linspace(E_min, E_max, num_points)

# Monte Carlo integration for phase space volume Ω(E)
def phase_space_volume(E, num_samples=100000):
    count = 0
    for _ in range(num_samples):
        px, py = np.random.uniform(-np.sqrt(2*m*E), np.sqrt(2*m*E), 2)
        x, y = np.random.uniform(-np.sqrt(2*E/m/omega**2), np.sqrt(2*E/m/omega**2), 2)
        H = (px**2 + py**2) / (2*m) + 0.5 * m * omega**2 * (x**2 + y**2)
        if H <= E:
            count += 1
    return count / num_samples

# Compute phase space volume for each energy level
Omega_E = np.array([phase_space_volume(E) for E in E_values])

# Compute density of states as g(E) = dΩ/dE
g_E = np.gradient(Omega_E, E_values)

# Normalize density of states
g_E /= np.max(g_E)

# Compute partition function Z(β)
def partition_function(beta):
    integrand = lambda E: g_E[np.searchsorted(E_values, E)] * np.exp(-beta * E)
    return spi.quad(integrand, E_min, E_max)[0]

beta_values = np.linspace(0.1, 5, 50)
Z_values = np.array([partition_function(beta) for beta in beta_values])

# Plot results
plt.figure(figsize=(12, 5))

# Density of States Plot
plt.subplot(1, 2, 1)
plt.plot(E_values, g_E, label=r"$g(E)$")
plt.xlabel("Energy (E)")
plt.ylabel("Density of States g(E)")
plt.title("Density of States for 2D Harmonic Oscillator")
plt.legend()

# Partition Function Plot
plt.subplot(1, 2, 2)
plt.plot(beta_values, Z_values, label=r"$Z(\beta)$", color="red")
plt.xlabel("Inverse Temperature (β)")
plt.ylabel("Partition Function Z(β)")
plt.title("Partition Function via Density of States")
plt.legend()

plt.tight_layout()
plt.show()
