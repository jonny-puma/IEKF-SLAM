import numpy as np
from rot import rotz

# TODO: internal state

def dynamics(t, y):
    # states
    theta = y[0]
    x = y[1:3]
    ps = y[3:]

    # speed & steering
    u = 0.1
    v = np.array([u, 0])
    omega = 0.1 #np.sin(0.001*t)

    d_theta = omega
    d_x = rotz(theta) @ v
    d_p = np.zeros(len(ps))

    # Process noise
    w_v = np.random.multivariate_normal((0,0,0), np.eye(3)*1e-5)

    return np.hstack((d_theta, d_x, d_p)) + w_v

