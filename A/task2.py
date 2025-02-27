#!/usr/bin/env python

from numpy import cos, sin, sqrt, pi, dot, roll, linspace, hstack
from numpy.random import rand, seed
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from matplotlib import animation

seed(1108)

# Constants
m1, m2, l1, l2, g = 0.5, 0.7, 0.3, 0.4, 9.8

def equations_of_motion(t, y):
    tht1, tht2, p1, p2 = y
    denom = m2 * (m1 + m2 - m2 * cos(tht1 - tht2) ** 2) * l1 ** 2 * l2 ** 2
    ddenom = 2 * m2 ** 2 * l1 ** 2 * l2 ** 2 * cos(tht1 - tht2) * sin(tht1 - tht2)
    num = m2 * l2 ** 2 * p1 ** 2 / 2 + (m1 + m2) * l1 ** 2 * p2 ** 2 / 2 - m2 * l1 * l2 * p1 * p2 * cos(tht1 - tht2)
    dnum = m2 * l1 * l2 * p1 * p2 * sin(tht1 - tht2)
    
    return [
        (m2 * l2 ** 2 * p1 - m2 * l1 * l2 * cos(tht1 - tht2) * p2) / denom,
        ((m1 + m2) * l1 ** 2 * p2 - m2 * l1 * l2 * cos(tht1 - tht2) * p1) / denom,
        -(dnum * denom - num * ddenom) / denom ** 2 - (m1 + m2) * g * l1 * sin(tht1),
        -(dnum * denom - num * ddenom) / denom ** 2 - m2 * g * l2 * sin(tht2),
    ]

# Solve equations of motion
solution = solve_ivp(
    equations_of_motion, (0, 5),
    y0=[1, 0.1, 0, 0],
    vectorized=True,
)

# Plot phase space trajectory
plt.plot(solution.y[1], solution.y[3])
plt.xlabel(r'$\theta_2$')
plt.ylabel(r'$p_2$')
plt.show()

plt.plot(solution.y[0], solution.y[1])
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.show()

# Animation function
def animate(sol):
    x1 = l1 * sin(sol.y[0])
    y1 = -l1 * cos(sol.y[0])
    x2 = x1 + l2 * sin(sol.y[1])
    y2 = y1 - l2 * cos(sol.y[1])
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2, markersize=8)
    
    def update(frame):
        line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        return line,
    
    anim = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=100, blit=True)
    plt.show()

animate(solution)

# Phase space volume estimation
def random_disk_point():
    r = sqrt(rand())
    theta = 2 * pi * rand()
    return r * cos(theta), r * sin(theta)

def calculate_area(points):
    vertices = ConvexHull(points).points
    x, y = vertices[:, 0], vertices[:, 1]
    return abs(dot(x, roll(y, 1)) - dot(y, roll(x, 1))) / 2

n_samples = 20
initial_conditions = []
time_eval = linspace(0, 5, 100)
for _ in range(n_samples):
    dtht2, dp2 = random_disk_point()
    dtht2 *= 0.01
    dp2 *= 0.01
    initial_conditions.append([1, 0.1 + dtht2, 0, dp2])

solutions = [solve_ivp(equations_of_motion, (0, 5), y0, vectorized=True, t_eval=time_eval) for y0 in initial_conditions]
areas = [calculate_area([[sol.y[1][i], sol.y[3][i]] for sol in solutions]) for i in range(len(time_eval))]

plt.plot(time_eval, areas)
plt.xlabel(r'$t$')
plt.ylabel('Phase Space Volume')
plt.show()
