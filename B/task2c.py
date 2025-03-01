from numpy import linspace, exp
from matplotlib import pyplot as plt

beta_eps = linspace(0, 5, 100)
n0 = 1/(1+exp(-beta_eps))
n1 = 1/(1+exp(beta_eps))
plt.plot(beta_eps, n0, label=r'$\left<n_0\right>_{\mathrm{C}}/N$')
plt.plot(beta_eps, n1, label=r'$\left<n_\epsilon\right>_{\mathrm{C}}/N$')
plt.xlabel(r'$\beta\epsilon$')
plt.legend()
plt.show()