#!/usr/bin/env python

from numpy import linspace, array, exp, sum, geomspace
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

def quantities(levels, beta, N):
	mu_list = []
	n0_list = []
	dn0_dT_list = []
	cv_list = []
	for b in beta:
		mu = fsolve(lambda m: sum(1/(exp(b*(levels-m))-1))-N, levels[0]-1/b/N)[0]
		e = levels-mu
		eb = exp(b*e)
		n = 1/(eb-1)
		d = eb*n**2
		dn_dT = d*b**2*(e-sum(d*e)/sum(d))
		mu_list.append(mu)
		n0_list.append(n[0])
		dn0_dT_list.append(dn_dT[0])
		cv_list.append(sum(dn_dT*levels))
	return array(mu_list), array(n0_list), array(dn0_dT_list), array(cv_list)

def plot(levels, t_vals, n=1e5):
	beta = 1/t_vals
	mu, n0, dn0_dT, cv = quantities(levels, beta, n)
	fig, (ax_mu, ax_n0, ax_n0_log, ax_dn0_dT, ax_cv) = plt.subplots(5, sharex=True)
	ax_mu.plot(t_vals, -mu)
	ax_mu.set_ylabel(r'$-\mu$')
	ax_mu.set_yscale('log')
	ax_n0.plot(t_vals, n0)
	ax_n0.set_ylabel(r'$\left<n_0\right>$')
	ax_n0_log.plot(t_vals, n0)
	ax_n0_log.set_ylabel(r'$\left<n_0\right>$')
	ax_n0_log.set_yscale('log')
	ax_dn0_dT.plot(t_vals, -dn0_dT)
	ax_dn0_dT.set_ylabel(r'$-\frac{\mathrm{d}\left<n_0\right>}{\mathrm{d}T}$')
	ax_cv.plot(t_vals, cv)
	ax_cv.set_ylabel(r'$C_V$')
	ax_n0.set_xscale('log')
	plt.xlabel(r'T')
	plt.show()

plot(linspace(0, 1, 2), geomspace(1e1, 1e8, 100))
plot(linspace(0, 2000, 200)**(1/20), geomspace(1e-1, 1e6, 100))