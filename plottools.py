import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def nees(X, P):
    """
        Average Normalised Estimation Error squared.
        Measures the consistency of a filter.
        NEES approx 1 is considered consistent.
    """
    l, t = X.shape
    n = np.zeros(t)
    for i in range(t):
        if np.count_nonzero(P[:,:,i]) == 0:
            n[i] = 0
        else:
            n[i] = X[:,i]@np.linalg.inv(P[:,:,i])@X[:,i].T
    return np.cumsum(n)/(np.arange(1,t+1)*l)

def rms(X):
    """
        Root Mean Squared error.
    """
    l, t = X.shape
    r = np.zeros(t)
    for i in range(t):
        r[i] = X[:,i]@X[:,i]
    return np.sqrt(r/l)

def plot_simulation(sol, p, ekf_y, ekf_P, iekf_y, iekf_P):
    t = len(sol.t)
    lp = len(p)
    # Plot EKF trajectory
    plt.figure()
    ax = plt.gca()
    plt.plot(sol.y[1,:], sol.y[2,:], "b")
    plt.plot(ekf_y[1,:], ekf_y[2,:], color="r", linestyle="--", alpha=0.7)
    for i in range(0, lp, 2):
        # true landmark positions
        plt.plot(p[i], p[i+1], ".b")
        # last landmark estimate
        plt.plot(ekf_y[3+i,-1], ekf_y[4+i,-1], "xr", markersize=6)
        # 3 sigma confidence ellipse of last landmark estimate
        confidence_ellipse(ekf_y[3+i,-1],
                           ekf_y[4+i,-1],
                           ekf_P[3+i:5+i,3+i:5+i,-1],
                           ax, edgecolor="r", n_std=3)
    plt.legend(("System trajectory",
                "Estimated trajectory",
                "Landmark positions",
                "Last landmark estimates with\n"
                 "99% confidence interval"))
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EKF System Trajectory")

    # Plot IEKF trajectory
    plt.figure()
    ax = plt.gca()
    plt.plot(sol.y[1,:], sol.y[2,:], "b")
    plt.plot(iekf_y[1,:], iekf_y[2,:], color="r", linestyle="--", alpha=0.7)
    for i in range(0, lp, 2):
        # true landmark positions
        plt.plot(p[i], p[i+1], ".b")
        # last landmark estimate
        plt.plot(iekf_y[3+i,-1], iekf_y[4+i,-1], "xr", markersize=6)
        # 3 sigma confidence ellipse of last landmark estimate
        confidence_ellipse(iekf_y[3+i,-1],
                           iekf_y[4+i,-1],
                           iekf_P[3+i:5+i,3+i:5+i,-1],
                           ax, edgecolor="r", n_std=3)
    plt.legend(("System trajectory",
                "Estimated trajectory",
                "Landmark positions",
                "Last landmark estimates with\n"
                 "99% confidence interval"))
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("IEKF System Trajectory")

    # Calculate NEES
    avg_ekf_e = sol.y - ekf_y
    avg_iekf_e = sol.y - iekf_y
    avg_ekf_nees = nees(avg_ekf_e[:3,1:], ekf_P[:3,:3,1:])
    avg_iekf_nees = nees(avg_iekf_e[:3,1:], iekf_P[:3,:3,1:])

    # Plot NEES
    plt.figure()
    plt.plot(sol.t[1:], avg_ekf_nees)
    plt.plot(sol.t[1:], avg_iekf_nees)
    plt.axhline(1, color="k", linewidth=0.75)
    plt.xlabel(" time [s]")
    plt.ylabel("NEES")
    plt.title("Pose NEES")
    plt.legend(("EKF", "IEKF"))

    # Plot position RMS
    avg_ekf_rms = rms(avg_ekf_e[1:3,1:])
    avg_iekf_rms = rms(avg_iekf_e[1:3,1:])
    plt.figure()
    plt.plot(sol.t[1:], avg_ekf_rms)
    plt.plot(sol.t[1:], avg_iekf_rms)
    plt.axhline(0, color="k", linewidth=0.75)
    plt.xlabel("time [s]")
    plt.ylabel("RMS [m]")
    plt.title("Position RMS")
    plt.legend(("EKF", "IEKF"))

    # plot heading error and 99% confidence interval
    plt.figure()

    plt.subplot(1,2,1)
    plt.plot(sol.t, avg_ekf_e[0,:])
    plt.plot(sol.t, 3*np.sqrt(ekf_P[0,0,:]), "r")
    plt.plot(sol.t, -3*np.sqrt(ekf_P[0,0,:]), "r")
    plt.title("EKF")
    plt.ylabel("Error [rad]")
    plt.xlabel("time [s]")
    ax = plt.gca()

    plt.subplot(1,2,2, sharey=ax)
    plt.plot(sol.t, avg_iekf_e[0,:])
    plt.plot(sol.t, 3*np.sqrt(iekf_P[0,0,:]), "r")
    plt.plot(sol.t, -3*np.sqrt(iekf_P[0,0,:]), "r")
    plt.title("IEKF")
    plt.xlabel("time [s]")
    plt.legend(("Error", "99% confidence interval"), loc="upper right")

    plt.suptitle("Heading error")

    plt.show()
