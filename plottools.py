import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2


# Plot settings
matplotlib.rcParams['figure.figsize'] = [12, 8]
matplotlib.rcParams['savefig.dpi'] = 200

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
        Normalised Estimation Error squared.
        Measures the consistency of a filter.
        NEES approx 1 is considered consistent.
    """
    l, t = X.shape
    n = np.zeros(t)
    for i in range(t):
        if np.count_nonzero(P[:,:,i]) == 0:
            n[i] = 0
        else:
            n[i] = X[:,i]@np.linalg.inv(P[:,:,i])@X[:,i]
    return n/l

def rms(X):
    """
        Root Mean Squared error.
    """
    l, t = X.shape
    r = np.zeros(t)
    for i in range(t):
        r[i] = X[:,i]@X[:,i]
    return np.sqrt(r/l)

def plot_simulation(sol, p, ekf_y, ekf_P, ekf_nees, iekf_y, iekf_P, iekf_nees, n, save=False):
    """
        Plot the result of a simulation and compare the
        estimates from two filters.
        
        Parameters:
        ----------
        sol: solution from ivp solver
        p: landmarks
        (i)ekf_y: state estimates from filter
        (i)ekf_P: covariance matrix from filter
        n: number of monte carlo runs
        save: bool, save to file if true
    """

    t = len(sol.t)
    lp = len(p)
    # Plot EKF trajectory
    plt.figure()
    ax = plt.gca()
    plt.plot(sol.y[1,:], sol.y[2,:], "b")
    plt.plot(ekf_y[1,:], ekf_y[2,:], color="r", linestyle="--", alpha=0.7)
    for i in range(0, lp, 2):
        # True landmark positions
        plt.plot(p[i], p[i+1], ".b")
        # Last landmark estimate
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

    if save:
        plt.savefig("data/plots/ekf_trajectory")

    # Plot IEKF trajectory
    plt.figure()
    ax = plt.gca()
    plt.plot(sol.y[1,:], sol.y[2,:], "b")
    plt.plot(iekf_y[1,:], iekf_y[2,:], color="r", linestyle="--", alpha=0.7)
    for i in range(0, lp, 2):
        # True landmark positions
        plt.plot(p[i], p[i+1], ".b")
        # Last landmark estimate
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

    if save:
        plt.savefig("data/plots/iekf_trajectory")

    # Pose error
    ekf_e = sol.y - ekf_y[:3]
    iekf_e = sol.y - iekf_y[:3]

    # Calculate NEES
    # ekf_nees = nees(ekf_e[:,1:], ekf_P[:3,:3,1:])
    # iekf_nees = nees(iekf_e[:,1:], iekf_P[:3,:3,1:])

    # NEES confidence interval
    dof = 3*n
    nees_c1 = chi2.ppf(0.01, dof)/dof
    nees_c2 = chi2.ppf(0.99, dof)/dof


    # Plot NEES
    plt.figure()
    plt.plot(sol.t[1:], ekf_nees)
    plt.plot(sol.t[1:], iekf_nees)
    plt.axhline(nees_c1, color="r", linestyle="--")
    plt.axhline(nees_c2, color="r", linestyle="--")
    plt.xlabel(" time [s]")
    plt.ylabel("NEES")
    plt.title("Pose NEES")
    plt.legend(("EKF", "IEKF", "99% confidence interval"))

    if save:
        plt.savefig("data/plots/pose_nees")

    cum_ekf_nees = np.cumsum(ekf_nees)/sol.t[1:]
    cum_iekf_nees = np.cumsum(iekf_nees)/sol.t[1:]

    # Plot normalized cumulative NEES
    plt.figure()
    plt.plot(sol.t[1:], cum_ekf_nees)
    plt.plot(sol.t[1:], cum_iekf_nees)
    plt.axhline(nees_c1, color="r", linestyle="--")
    plt.axhline(nees_c2, color="r", linestyle="--")
    plt.xlabel(" time [s]")
    plt.ylabel("NEES")
    plt.title("Normalized Cumulative Pose NEES")
    plt.legend(("EKF", "IEKF", "99% confidence interval"))

    if save:
        plt.savefig("data/plots/cum_pose_nees")

    # Plot position RMS
    ekf_rms = rms(ekf_e[1:,1:])
    iekf_rms = rms(iekf_e[1:,1:])
    plt.figure()
    plt.plot(sol.t[1:], ekf_rms)
    plt.plot(sol.t[1:], iekf_rms)
    plt.axhline(0, color="k", linewidth=0.75)
    plt.xlabel("time [s]")
    plt.ylabel("RMS [m]")
    plt.ylim(bottom=0)
    plt.title("Position RMS")
    plt.legend(("EKF", "IEKF"))

    if save:
        plt.savefig("data/plots/position_rms")

    # Plot heading error and 99% confidence interval
    plt.figure()

    plt.subplot(1,2,1)
    plt.plot(sol.t, ekf_e[0,:])
    plt.plot(sol.t, 3*np.sqrt(ekf_P[0,0,:]), "r")
    plt.plot(sol.t, -3*np.sqrt(ekf_P[0,0,:]), "r")
    plt.title("EKF")
    plt.ylabel("Error [rad]")
    plt.xlabel("time [s]")
    ax = plt.gca()

    plt.subplot(1,2,2, sharey=ax)
    plt.plot(sol.t, iekf_e[0,:])
    plt.plot(sol.t, 3*np.sqrt(iekf_P[0,0,:]), "r")
    plt.plot(sol.t, -3*np.sqrt(iekf_P[0,0,:]), "r")
    plt.title("IEKF")
    plt.xlabel("time [s]")
    plt.legend(("Error", "99% confidence interval"), loc="upper right")

    plt.suptitle("Heading error")

    if save:
        plt.savefig("data/plots/heading_error")

    plt.show()
