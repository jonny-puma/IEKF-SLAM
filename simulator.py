import sys
import numpy as np
import slam
import ekf
import matplotlib
from rot import rotz
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


# initial conditions
theta0 = np.pi/4
x0 = np.array([0,0])

# landmarks
p = np.array([ 1, 1,
              -1, 2,
              -2, 1])

y0 = np.hstack((theta0, x0, p))

# solve initial value problem
sol = solve_ivp(slam.dynamics, [0,40], y0, max_step = 0.1)
t = len(sol.t)

if not sol.success:
    print(f"Ivp solver failed: {sol.message}", file=sys.stderr)
    sys.exit(1)

# measurement noise
W = np.diag((1e-3, 1e-3))

# create noisy measurements
ekf_z = np.zeros((len(p), t))
for i in range(t):
    for j in range(0,len(p),2):
        w_w = np.random.multivariate_normal((0,0), W)
        ekf_z[j:j+2,i] = rotz(sol.y[0,i]).T@(sol.y[3+j:5+j,i] - sol.y[1:3,i]) + w_w

# ekf initial guess
ekf_x0 = np.hstack((y0[0:3], ekf_z[:,0]))
ekf_P0 = np.diag((0,0,0,3,3,3,3,3,3))

# inputs
ekf_u = np.vstack((np.full(t, 0.1), np.full(t, 0.1)))
ekf_u[:,0] = np.zeros(2)

# run EKF
ekf_y, ekf_P = ekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0)

plt.figure()
plt.plot(sol.y[1,:], sol.y[2,:])
plt.plot(ekf_y[1,:], ekf_y[2,:], color="r")
for i in range(0, len(p), 2):
    plt.plot(p[i], p[i+1], "xk")
for i in range(0, len(p), 2):
    plt.plot(ekf_y[3+i,:], ekf_y[4+i,:], ".r", alpha=0.01)
    plt.plot(ekf_y[3+i,-1], ekf_y[4+i,-1], "xg")
plt.axis("equal")
plt.title("System trajectory")
plt.show()
