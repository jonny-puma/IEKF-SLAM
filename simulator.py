import sys
import numpy as np
import slam
import ekf
import iekf
import plottools
from rot import rotz
from scipy.integrate import solve_ivp

"""
    This script runs a simulation of the unicycle model,
    generates noisy samples, runs the EKF and IEKF, and plots the results.
"""

# Number of monte carlo simulations
NRUNS = 100

# Sensor range in meters
sensor_range = 5

# Simulation length in seconds
t_end = 400

# Number of landmarks
lp = 20

# Initial conditions
theta0 = 0
x0 = np.array([0,0])

# Generate landmarks and stack with initial conditions
p = np.zeros(lp*2)
for i in range(0, lp):
    p[2*i] = np.cos(i*2*np.pi/lp)*8
    p[2*i+1] = np.sin(i*2*np.pi/lp)*8 + 6.125

y0 = np.hstack((theta0, x0, p))


# Solve initial value problem
dt = 0.75
u = (9*np.pi/180, 1)
dynsys = slam.Unicycle2D(u, dt)
sol = solve_ivp(dynsys.dynamics, [0,t_end], y0, t_eval=np.arange(0, t_end, dt))
t = len(sol.t)

# Check if solver successfull
if not sol.success:
    print(f"IVP solver failed: {sol.message}", file=sys.stderr)
    sys.exit(1)

# Process covariance
V_gain = 1e-3
ekf_V = np.eye(len(y0))*V_gain
#ekf_V = np.diag(np.hstack((np.ones(3)*V_gain, np.zeros(lp*2)))) 

# Measurement noise variance 
w_gain = 0.12
ekf_W = np.eye(lp*2)*w_gain**2

# Average of NRUNS simulations
dim = len(y0)
avg_ekf_y = np.zeros((dim, t))
avg_ekf_P = np.zeros((dim, dim, t))
avg_iekf_y = np.zeros((dim, t))
avg_iekf_P = np.zeros((dim, dim, t))

# Do NRUNS simulations
for _ in range(NRUNS):
    # Create noisy measurements
    ekf_z = np.zeros((lp*2, t))
    for i in range(t):
        for j in range(0,lp*2,2):
            dz = rotz(sol.y[0,i]).T@(sol.y[3+j:5+j,i] - sol.y[1:3,i])
            # Check if landmark within sensor range
            if np.linalg.norm(dz) <= sensor_range:
                W = abs(dz)*np.eye(2)*w_gain**2
                w_w = np.random.multivariate_normal((0,0), W)
                ekf_z[j:j+2,i] = dz + w_w
            else:
                # nan if out of sensor range
                ekf_z[j:j+2,i] = np.array((np.nan, np.nan))

    # Estimator initial guess
    p0 = np.ones(lp*2)*np.nan
    for i in range(0, 2*lp, 2):
        dp = np.linalg.norm(p[i:i+2] - y0[1:3]) 
        if dp < sensor_range:
            W0 = abs(dp)*np.eye(2)*w_gain**2
            p0[i:i+2] = p[i:i+2] + np.random.multivariate_normal((0,0), W0)
    ekf_x0 = np.hstack((y0[0:3], p0))
    ekf_P0 = np.diag(np.hstack((np.zeros(3),np.ones(lp*2))))

    # Noisy input
    w_u = np.random.normal(0,0.1*0.02,(2,t))
    ekf_u = np.vstack((np.full(t, u[0]), np.full(t, u[1]))) + w_u
    ekf_u[:,0] = np.zeros(2)

    # Run EKF and add results
    ekf_y, ekf_P = ekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0, ekf_V, ekf_W)
    avg_ekf_y += ekf_y
    avg_ekf_P += ekf_P

    # Run IEKF and add results
    iekf_y, iekf_P = iekf.estimate(sol.t, ekf_z, ekf_u, ekf_x0, ekf_P0, ekf_V, ekf_W)
    avg_iekf_y += iekf_y
    avg_iekf_P += iekf_P

# Divide by runs to get avarage
avg_ekf_y /= NRUNS
avg_ekf_P /= NRUNS
avg_iekf_y /= NRUNS
avg_iekf_P /= NRUNS

# Plot results
plottools.plot_simulation(sol, p, avg_ekf_y, avg_ekf_P, avg_iekf_y, avg_iekf_P)
