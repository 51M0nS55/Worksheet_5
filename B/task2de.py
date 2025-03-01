from numpy import linspace, exp
from matplotlib import pyplot as plt

n = 100
beta_eps = linspace(0, 1, 100)
n1 = 1/(exp(beta_eps)-1) - (1+n)/(exp(beta_eps*(1+n))-1)
n1[0] = n/2
n0 = n - n1
plt.plot(beta_eps, n0, label=r'$\left<n_0\right>$')
plt.plot(beta_eps, n1, label=r'$\left<n_\epsilon\right>$')
plt.xlabel(r'$\beta\epsilon$')
plt.title(f'$N={n}$')
plt.legend()
plt.show()