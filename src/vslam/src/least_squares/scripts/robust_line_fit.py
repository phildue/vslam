# Copyright 2022 Philipp.Duernay
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt
p0 = np.array([0, 0])
p1 = np.array([1, 1])

N_POINTS = 100
HUBER_C = 1.345
N_ITERATIONS = 20
"""
Create data
"""
ps = np.zeros((N_POINTS, 2))
for i in range(N_POINTS):
    if np.random.uniform() > 0.95:
        a = np.array([np.random.normal(0, 200.0), np.random.normal(0, 200.0)])
        ps[i] = p0 + np.random.uniform(-100, 100) * p1 + a
    else:
        a = np.array([np.random.normal(0, 2.0), np.random.normal(0, 2.0)])
        ps[i] = p0 + np.random.uniform(-100, 100) * p1 + a

"""
For linear least squares, normal Equations are given by:

X.T * X * mc = X.T*y

Hence the least squares solution is:

mc = (X.T * X)^-1 ((X.T*y)

In order to avoid inverting X.T*X we can also use cholesky decomposition.

For weighted linear least squares normal equations are given by:

X.T * W * X * mc = X.T * W * y

where W is diagonal with sample wise information
"""


X = np.ones((N_POINTS, 2))
X[:, 0] = ps[:, 0]
y = ps[:, 1]

mc = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
mc_robust = mc.copy()

"""
Iteratively fit model and weight samples based on residual
"""
for i in range(N_ITERATIONS):
    r = np.zeros_like(y)
    r = y - mc_robust.dot(X.transpose())
    r -= r.mean()
    r /= r.std()

    # Construct weight matrix based on huber loss
    w = np.zeros_like(r)
    rc = HUBER_C * r
    rinv = 1.0/r
    w[np.abs(r) < HUBER_C] = 1.0
    idx_larger = np.logical_and(np.abs(r) >= HUBER_C, rc > 0)
    w[idx_larger] = rinv[idx_larger]
    idx_lesser = np.logical_and(np.abs(r) >= HUBER_C, rc <= 0)
    w[idx_lesser] = -rinv[idx_lesser]
    W = np.diag(w.flatten())
    mc_robust = np.linalg.solve(X.T.dot(W).dot(X), X.T.dot(W).dot(y))

plt.plot(ps[:, 0], ps[:, 1], 'x')
x = np.linspace(np.min(ps[:, 0]), np.max(ps[:, 0]), 100)
y = mc[0] * np.linspace(np.min(ps[:, 0]), np.max(ps[:, 0]), 100) + mc[1]
y_rob = mc_robust[0] * np.linspace(np.min(ps[:, 0]), np.max(ps[:, 0]), 100) + mc_robust[1]
plt.plot(x, y, '-')
plt.plot(x, y_rob, '-')
plt.legend(['Measurements', 'Fit', 'Robust Fit'])
plt.xlabel('x')
plt.xlabel('y')
plt.show()
