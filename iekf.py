import pdb
import numpy as np
from rot import rotz, B

def estimate(t, y, u, x0, P0, V, W):
    # Dimensions
    lx = len(x0)
    lt = len(t)
    lp = lx - 3

    # Initialize estimates
    X_m = np.zeros((lx, lt))
    P_m = np.zeros((lx, lx, lt))
    X_m[:,0] = x0
    P_m[:,:,0] = P0

    # Identity matrix & skew symmetric matrix
    I = np.eye(lx)
    J = np.array([[0, -1],
                  [1, 0]])

    for i in range(1, lt):
        # timestep
        dt = t[i] - t[i-1]

        # velocity vector
        v = np.array([u[1,i], 0])

        # Linearized dynamics
        # A = I

        L = np.zeros((lx, lx))
        L[0,0] = 1
        L[1:3,1:3] = rotz(X_m[0,i-1])
        for j in range(1, lx, 2):
            L[j:j+2,0] = -J@X_m[j:j+2,i-1]

        # Prior estimate
        t_p = X_m[0,i-1] + u[0,i]*dt
        rotT = rotz(t_p).T
        x_p = X_m[1:3,i-1] + rotz(X_m[0,i-1])@v*dt
        p_p = X_m[3:,i-1]
        P_p = P_m[:,:,i-1] + L@V@L.T
        X_p = np.hstack((t_p, x_p, p_p))

        # Innovation error
        hx = np.zeros(lp)
        for j in range(0, lp, 2):
            hx[j:j+2] = rotT@(p_p[j:j+2] - x_p)
        z = y[:,i] - hx

        # Linearized observation
        H = np.zeros((lp, lx))
        for j in range(0, lp, 2):
            H[j:j+2,1:3] = -rotT
            H[j:j+2,3+j:5+j] = rotT

        # Kalman gain, tiny perturbation added to avoid inverting singular matrix
        K = P_p@H.T@np.linalg.inv(H@P_p@H.T + W) #+ np.random.normal(0,1e-5, (lp, lp)))

        # Posteriror uncertainty (symmetric P_m for numerical stability)
        P_m[:,:,i] = (I-K@H)@P_p@(I-K@H).T + K@W@K.T

        # Posterior state update
        dX = K@z
        dB = B(dX[0])
        dR = rotz(dX[0])
        X_m[0,i] = X_p[0] + dX[0]
        for j in range(1, lx, 2):
            X_m[j:j+2,i] = dR@X_p[j:j+2] + dB@dX[j:j+2]
        
    return X_m, P_m