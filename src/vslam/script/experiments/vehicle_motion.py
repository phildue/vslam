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
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sophus.sophuspy as sp

"""
Vehicle Model
"""


class DifferentialDrive:
    def __init__(self, wheel_distance):
        self._wheel_distance = wheel_distance

    def forward(self, v, state, dt):
        vl, vr = v
        d = self._wheel_distance
        r = d/2 * (vl + vr)
        omega = (vr - vl)/d
        x, y, theta = state
        icc = np.array([x-r*np.sin(theta), y + r*np.cos(theta)])
        R = np.array([
                [np.cos(omega*dt), -np.sin(omega*dt), 0],
                [np.sin(omega*dt), np.cos(omega*dt), 0],
                [0, 0, 1],
        ])
        p = np.array([x-icc[0], y-icc[1], theta]).transpose()

        t = np.array([icc[0], icc[1], omega*dt]).transpose()
        # print(f"R={R}\np={p},t={t}")
        return R.dot(p) + t


"""
State Estimation
"""


class EKFConstantVelocitySE3:

    def __init__(self, covariance_process):
        self._pos = sp.SE3()
        self._vel = np.zeros((6,))
        self._P = np.identity(12)
        self._Q = covariance_process
        self._t = 0

    def _J_f_x(self, pose):
        """Compute Jacobian of f(x) = pose * exp(twist * dt)."""
        J_f_x = np.zeros((12, 12))
        t = pose.translation()
        R = pose.rotationMatrix()
        adj = np.zeros((6, 6))
        adj[:3, :3] = R
        adj[:3, 3:] = np.cross(t, np.identity(t.shape[0]) * -1)
        adj[3:, 3:] = R
        J_f_x[6:, 6:] = np.linalg.inv(adj)
        return J_f_x

    def _J_h_x(self, dt):
        """Compute Jacobian of measurement function h(x) = twist*dt."""
        return np.hstack([np.zeros((6, 6)), np.identity((6))*dt])

    def predict(self, t):
        x, P = self._predict(t)
        return x[:6], P[:6, :6], x[6:], P[6:, 6:]

    def _predict(self, t):
        dt = t - self._t
        pos = self._pos * sp.SE3.exp(self._vel*dt)

        J_f_x = self._J_f_x(self._pos)
        P = J_f_x @ self._P @ J_f_x.transpose() + self._Q.transpose()
        return np.hstack([pos.log(), self._vel]), P

    def update(self, z, z_cov, t):

        dt = t - self._t
        x, self._P = self._predict(t)
        self._pos = sp.SE3.exp(x[:6])
        self._vel = x[6:]

        # expectation
        h = self._vel*dt
        J_h_x = self._J_h_x(dt)
        E = J_h_x @ self._P @ J_h_x.transpose()
        print(f"E={E}")

        # innovation
        y = z-h
        Z = E + z_cov
        print(f"Z={Z}")

        # print(f"P={P_}")

        # Kalman gain
        K = self._P @ J_h_x.transpose() @ np.linalg.inv(Z)
        print(f"K={K}")

        # Correction
        dx = K @ y
        print(f"dx={dx}")

        self._vel = self._vel + dx[6:]
        # self._pos = self._pos + SE2Tangent(dx[:3])
        print(f"vel={self._vel}")

        self._P = self._P - K @ Z @ K.transpose()
        self._t = t
        print(f"P={self._P}")


if __name__ == '__main__':
    state = np.zeros((3,))
    v = np.array([1, 2])
    dt = 0.1
    sigma_measurement = 0.01
    cov_measurement = (sigma_measurement**2) * np.identity(6)
    cov_process = 0.0001*np.identity(12)
    kalman = EKFConstantVelocitySE3(cov_process)

    vehicle = DifferentialDrive(0.5)
    n_steps = 100
    trajectory = np.zeros((n_steps, 3))
    trajectory_pred = np.zeros((n_steps, 3))
    velocity = np.zeros((n_steps, 6))
    velocity_noisy = np.zeros((n_steps, 6))
    velocity_pred = np.zeros((n_steps, 6))

    uncertainty = np.zeros((n_steps, 1))
    pose_gt_prev = None
    for ti in range(n_steps):
        stateR = R.from_euler('xyz', [0, 0, state[2]]).as_matrix()
        pose_gt_prev = sp.SE3(stateR, np.array([state[0], state[1], 0]))
        state = vehicle.forward(v, state, dt)
        stateR = R.from_euler('xyz', [0, 0, state[2]]).as_matrix()
        pose_gt = sp.SE3(stateR, np.array([state[0], state[1], 0]))
        delta_pose = pose_gt_prev.inverse() * pose_gt

        delta_pose_noisy = delta_pose.log()
        delta_pose_noisy[0:2] += sigma_measurement * np.random.rand(2)

        kalman.update(delta_pose_noisy, cov_measurement, ti*dt)

        xp, Pp, xv, Pv = kalman.predict(ti*dt)

        trajectory[ti, :2] = state[:2]
        trajectory_pred[ti] = sp.SE3.exp(xp).translation()
        velocity_pred[ti] = xv
        velocity_noisy[ti] = delta_pose_noisy/dt
        velocity[ti] = delta_pose.log()/dt
        uncertainty[ti] = np.linalg.det(Pp) + np.linalg.det(Pv)

    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.plot(trajectory_pred[:, 0], trajectory_pred[:, 1], '-.')

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['GT', 'Estimated*', 'Integrated'])

    plt.figure()
    plt.ylabel('v [..]')
    plt.plot(np.linalg.norm(velocity[:, :2], axis=1))
    plt.plot(np.linalg.norm(velocity_noisy[:, :2], axis=1))
    plt.plot(np.linalg.norm(velocity_pred[:, :2], axis=1))
    plt.legend(['GT', 'Noisy', 'Estimated'])

    plt.figure()
    plt.ylabel('va [$^\\circ$]')
    plt.plot(velocity[:, 2])
    plt.plot(velocity_pred[:, 2])
    plt.legend(['GT', 'Estimated'])

    plt.figure()
    plt.ylabel('$\\Sigma$')
    plt.plot(uncertainty)

    plt.show()
