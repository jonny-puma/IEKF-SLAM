import numpy as np
from rot import rotz

class Unicycle2D:
    """
        Simple 2D unicycle model.
        u = [omega, v]
    """
    def __init__(self, u, dt):
        self.u = u
        self.dt = dt

    def dynamics(self, t, y):
        # states
        theta = y[0]
        x = y[1:3]
        ps = y[3:]

        # speed & steering
        v = np.array([self.u[1], 0])
        omega = self.u[0]

        d_theta = omega
        d_x = rotz(theta) @ v
        d_p = np.zeros(len(ps))
        d_X = np.hstack((d_theta, d_x, d_p))

        # Process noise
        """
        V = np.diag(0.02*np.hstack((omega, v, np.zeros(len(ps)))))**2
        w_v = np.random.multivariate_normal(np.zeros(len(y)), V)
        """

        return d_X #+ w_v 

