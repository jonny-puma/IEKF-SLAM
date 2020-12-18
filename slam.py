import numpy as np
from rot import rotz

class Unicycle2D:
    """
        Simple 2D unicycle model.
        u = [omega, v]
    """
    def __init__(self, u, V):
        self.u = u
        self.V = V

    def dynamics(self, t, y):
        # states
        theta = y[0]
        x = y[1:3]

        # velocity & steering
        v = np.array([self.u[1], 0])
        omega = self.u[0]

        d_theta = omega
        d_x = rotz(theta) @ v
        d_X = np.hstack((d_theta, d_x))

        # Process noise
        w_v = np.random.multivariate_normal(np.zeros(len(y)), self.V)

        return d_X + d_X*w_v 

