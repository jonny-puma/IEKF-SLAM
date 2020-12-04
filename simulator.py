import sys
import numpy as np
import slam
import ekf
import iekf
import matplotlib
from plottools import confidence_ellipse, nees, rms
from rot import rotz
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

"""
    This script runs a simulation of the unicycle model,
    generates noisy samples, runs the EKF and IEKF, and plots the results.
"""

# set random seed
np.random.seed(99)

# initial conditions
theta0 = 0
x0 = np.array([0,0])

# generate landmarks
lp = 15
p = np.zeros(lp*2)
for i in range(0, lp):
    p[2*i] = np.cos(i*2*np.pi/lp)*8
    p[2*i+1] = np.sin(i*2*np.pi/lp)*8 + 6.125

y0 = np.hstack((theta0, x0, p))


# solve initial value problem
t_end = 400
dt = 0.75
u = (9*np.pi/180, 1)
# TODO: noise as init variable!
dynsys = slam.Unicycle2D(u, dt)
sol = solve_ivp(dynsys.dynamics, [0,t_end], y0, t_eval=np.arange(0, t_end, dt))
t = len(sol.t)

# check if solver successfull
if not sol.success:
    print(f"IVP solver failed: {sol.message}", file=sys.stderr)
    sys.exit(1)

# process covariance
V_gain = 1e-3
ekf_V = np.diag(np.hstack((np.ones(3)*V_gain, np.zeros(lp*2)))) 

# measurement noise variance 
w_gain = 0.1
ekf_W = 2*np.eye(lp*2)*w_gain

# create noisy measurements
W = np.eye(2)*w_gain
ekf_z = np.zeros((lp*2, t))
for i in range(t):
    for j in range(0,lp*2,2):
        dz = rotz(sol.y[0,i]).T@(sol.y[3+j:5+j,i] - sol.y[1:3,i])
        #W = abs(dz)*np.eye(2)*w_gain**2
        w_w = np.random.multivariate_normal((0,0), W)
        ekf_z[j:j+2,i] = dz + w_w

# ekf initial guess
ekf_x0 = np.hstack((y0[0:3], ekf_z[:,0]))
ekf_P0 = np.diag(np.hstack((np.zeros(3),np.ones(lp*2))))

# inputs
# w_u = np.random.normal(0,0.1*0.02,(2,t))
ekf_u = np.vstack((np.full(t, u[0]), np.full(t, u[1]))) # + w_u
ekf_u[:,0] = np.zeros(2)

# Run EKF
ekf_y, ekf_P = ekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0, ekf_V, ekf_W)

# Run IEKF
iekf_y, iekf_P = iekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0, ekf_V, ekf_W)

# Plot EKF trajectory
plt.figure()
ax = plt.gca()
plt.plot(sol.y[1,:], sol.y[2,:], "b")
plt.plot(ekf_y[1,:], ekf_y[2,:], color="r", linestyle="--", alpha=0.7)
for i in range(0, lp*2, 2):
    # true landmark positions
    plt.plot(p[i], p[i+1], ".b")
    # last landmark estimate
    plt.plot(ekf_y[3+i,-1], ekf_y[4+i,-1], "xr", markersize=6)
    # 3 sigma certainty ellipse of last landmark estimate
    confidence_ellipse(ekf_y[3+i,-1],
                       ekf_y[4+i,-1],
                       ekf_P[3+i:5+i,3+i:5+i,-1],
                       ax, edgecolor="r", n_std=3)
plt.legend(("System trajectory",
            "Estimated trajectory",
            "Landmark positions",
            "Last landmark estimates with\n"
             "99% uncertainty interval"))
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("EKF System Trajectory")

# Plot IEKF trajectory
plt.figure()
ax = plt.gca()
plt.plot(sol.y[1,:], sol.y[2,:], "b")
plt.plot(iekf_y[1,:], iekf_y[2,:], color="r", linestyle="--", alpha=0.7)
for i in range(0, lp*2, 2):
    # true landmark positions
    plt.plot(p[i], p[i+1], ".b")
    # last landmark estimate
    plt.plot(iekf_y[3+i,-1], iekf_y[4+i,-1], "xr", markersize=6)
    # 3 sigma certainty ellipse of last landmark estimate
    confidence_ellipse(iekf_y[3+i,-1],
                       iekf_y[4+i,-1],
                       iekf_P[3+i:5+i,3+i:5+i,-1],
                       ax, edgecolor="r", n_std=3)
plt.legend(("System trajectory",
            "Estimated trajectory",
            "Landmark positions",
            "Last landmark estimates with\n"
             "99% uncertainty interval"))
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
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
plt.axhline(1, color="k", linewidth=0.75)
plt.xlabel("time step")
plt.ylabel("NEES")
plt.title("Pose NEES")
plt.legend(("EKF", "IEKF"))

# Plot position RMS
ekf_rms = rms(ekf_e[1:3,1:])
iekf_rms = rms(iekf_e[1:3,1:])
plt.figure()
plt.plot(sol.t[1:], ekf_rms)
plt.plot(sol.t[1:], iekf_rms)
plt.axhline(0, color="k", linewidth=0.75)
plt.xlabel("time step")
plt.ylabel("RMS [m]")
plt.title("Position RMS")
plt.legend(("EKF", "IEKF"))

# plot heading error and 99% uncertainty interval
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
