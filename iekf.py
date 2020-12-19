import numpy as np
from rot import rotz, B


def estimate(t, y, u, x0, P0, V, W):
    """
        Run IEKF estimate on batch data.
        
        Parameters:
        -----------
        t: time samples
        y: landmark measurements, nan when measurement not available
        x0: initial state guess
        P0: initial covariance
        V: Process covariance
        W: Measurement covariance

        Returns:
        --------
        X_m: state estimates
        P_m: estimate covariances
    """

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
        # Timestep length
        dt = t[i] - t[i-1]

        # Velocity vector
        v = np.array([u[1,i], 0])

        # Linearized dynamics
        # A = I

        L = np.zeros((lx, lx))
        L[0,0] = 1
        L[1:3,1:3] = rotz(X_m[0,i-1])
        for j in range(1, lx, 2):
            if not np.isnan(X_m[j,i-1]):
                L[j:j+2,0] = -J@X_m[j:j+2,i-1]

        # Prior estimate
        t_p = X_m[0,i-1] + u[0,i]*dt
        rotT = rotz(t_p).T
        x_p = X_m[1:3,i-1] + rotz(X_m[0,i-1])@v*dt
        p_p = X_m[3:,i-1]
        P_p = P_m[:,:,i-1] + L@V@L.T

        # Innovation error
        hx = np.zeros(lp)
        for j in range(0, lp, 2):
            # First time observing landmark
            if np.isnan(p_p[j]) and not np.isnan(y[j,i]):
                p_p[j:j+2] = rotz(t_p)@y[j:j+2,i] + x_p
                P_p[j+3,j+3] = P_m[j+3,j+3,0]
                P_p[j+4,j+4] = P_m[j+4,j+4,0]
            hx[j:j+2] = rotT@(p_p[j:j+2] - x_p)
        z = np.nan_to_num(y[:,i] - hx)

        # Linearized observation
        H = np.zeros((lp, lx))
        for j in range(0, lp, 2):
            # Check if we have valid measurement
            if not np.isnan(y[j,i]):
                H[j:j+2,1:3] = -rotT
                H[j:j+2,3+j:5+j] = rotT

        # Kalman gain
        K = P_p@H.T@np.linalg.inv(H@P_p@H.T + W)

        # Posteriror uncertainty (symmetric P_m for numerical stability)
        P_m[:,:,i] = (I-K@H)@P_p@(I-K@H).T + K@W@K.T

        # Concatenated prior
        X_p = np.hstack((t_p, x_p, p_p))

        # Posterior state update
        dX = K@z
        dB = B(dX[0])
        dR = rotz(dX[0])
        X_m[0,i] = X_p[0] + dX[0]
        for j in range(1, lx, 2):
            X_m[j:j+2,i] = dR@X_p[j:j+2] + dB@dX[j:j+2]

    return X_m, P_m
