import numpy as np


def rotz(theta):
    """
        Rotation matrix in the xy-plane.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def B(a):
    """
        B matrix from IEKF.
    """
    if a == 0:
        return np.zeros((2,2))
    else:
        return np.array([[np.sin(a)/a, -(1-np.cos(a))/a],
                         [(1-np.cos(a))/a, np.sin(a)/a]])
