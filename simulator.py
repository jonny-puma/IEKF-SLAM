import sys
import numpy as np
import slam
import ekf
import iekf
import matplotlib
from plottools import confidence_ellipse, nees
from rot import rotz
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


# initial conditions
theta0 = 0
x0 = np.array([0,0])

# landmarks
p = np.array([ 9, 7,
              -0, 14,
              -9, 7])

y0 = np.hstack((theta0, x0, p))


# solve initial value problem
t_end = 400
dt = 0.75
u = (9*np.pi/180, 1)
sys = slam.Unicycle2D(u, dt)
sol = solve_ivp(sys.dynamics, [0,t_end], y0, t_eval=np.arange(0, t_end, dt))
t = len(sol.t)

if not sol.success:
    print(f"Ivp solver failed: {sol.message}", file=sys.stderr)
    sys.exit(1)

# noise covariance
ekf_V = np.diag((1e-3, 1e-3, 1e-3, 0, 0, 0, 0, 0, 0))

# measurement noise cov = 12% of distance
w_gain = 0.12
ekf_W = np.eye(len(p))*w_gain**2

# create noisy measurements
ekf_z = np.zeros((len(p), t))
for i in range(t):
    for j in range(0,len(p),2):
        dz = rotz(sol.y[0,i]).T@(sol.y[3+j:5+j,i] - sol.y[1:3,i])
        W = (w_gain*np.eye(2)*dz)**2
        w_w = np.random.multivariate_normal((0,0), W)
        ekf_z[j:j+2,i] = dz + w_w

# ekf initial guess
ekf_x0 = np.hstack((y0[0:3], ekf_z[:,0]))
ekf_P0 = np.diag((0,0,0,1,1,1,1,1,1))

# inputs
w_u = np.random.normal(0,0.1*0.02,(2,t))
ekf_u = np.vstack((np.full(t, u[0]), np.full(t, u[1]))) + w_u
ekf_u[:,0] = np.zeros(2)

# Run EKF
ekf_y, ekf_P = ekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0, ekf_V, ekf_W)

# Run IEKF
iekf_y, iekf_P = iekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0, ekf_V, ekf_W)

# Plot EKF
plt.figure()
ax = plt.gca()
plt.plot(sol.y[1,:], sol.y[2,:])
plt.plot(ekf_y[1,:], ekf_y[2,:], color="r", alpha=0.7)
for i in range(0, len(p), 2):
    plt.plot(p[i], p[i+1], "xk")
for i in range(0, len(p), 2):
    # plt.plot(ekf_y[3+i,:], ekf_y[4+i,:], ".r", alpha=0.01)
    plt.plot(ekf_y[3+i,-1], ekf_y[4+i,-1], "xr")
    confidence_ellipse(ekf_y[3+i,-1], ekf_y[4+i,-1], ekf_P[3+i:5+i,3+i:5+i,-1], ax, edgecolor="r")
plt.axis("equal")
plt.title("EKF System Trajectory")

# Plot IEKF
plt.figure()
ax = plt.gca()
plt.plot(sol.y[1,:], sol.y[2,:])
plt.plot(iekf_y[1,:], iekf_y[2,:], color="r", alpha=0.7)
for i in range(0, len(p), 2):
    plt.plot(p[i], p[i+1], "xk")
for i in range(0, len(p), 2):
    # plt.plot(iekf_y[3+i,:], iekf_y[4+i,:], ".r", alpha=0.01)
    plt.plot(iekf_y[3+i,-1], iekf_y[4+i,-1], "xr")
    confidence_ellipse(iekf_y[3+i,-1], iekf_y[4+i,-1], iekf_P[3+i:5+i,3+i:5+i,-1], ax, edgecolor="r")
plt.axis("equal")
plt.title("IEKF System Trajectory")

# Calculate NEES
ekf_e = sol.y - ekf_y
iekf_e = sol.y - iekf_y
ekf_nees = nees(ekf_e[:3,1:], ekf_P[:3,:3,1:])
iekf_nees = nees(iekf_e[:3,1:], iekf_P[:3,:3,1:])

# Plot NEES
plt.figure()
plt.plot(sol.t[1:], ekf_nees)
plt.plot(sol.t[1:], iekf_nees)
plt.xlabel("time step")
plt.ylabel("NEES")
plt.title("Pose NEES")
plt.legend(("EKF", "IEKF"))

# plot heading error and uncertainty
plt.figure()

plt.subplot(1,2,1)
plt.plot(sol.t, ekf_e[0,:])
plt.plot(sol.t, 3*np.sqrt(ekf_P[0,0,:]), "r")
plt.plot(sol.t, -3*np.sqrt(ekf_P[0,0,:]), "r")
plt.title("EKF")
plt.ylabel("Error [rad]")
plt.xlabel("time step")
ax = plt.gca()

plt.subplot(1,2,2, sharey=ax)
plt.plot(sol.t, iekf_e[0,:])
plt.plot(sol.t, 3*np.sqrt(iekf_P[0,0,:]), "r")
plt.plot(sol.t, -3*np.sqrt(iekf_P[0,0,:]), "r")
plt.title("IEKF")
plt.xlabel("time step")
plt.legend(("Error", "99% certainty interval"), loc="upper right")

plt.suptitle("Heading error")
plt.show()
