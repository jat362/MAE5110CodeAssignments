import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from A1_rimless_wheel import RimlessWheelParams
from A1_analysis import (compute_roa_grid, return_map, find_fixed_point,
                       floquet_multiplier)

os.makedirs("figures", exist_ok=True)

# roa map (state-space forward-rolling vs stopped)
def plot_roa(p, fname, n_theta=81, n_thetadot=81, thetadot_range=(-6, 6)):
    thetas, thetadots, grid = compute_roa_grid(
        p, thetadot_range=thetadot_range, n_theta=n_theta, n_thetadot=n_thetadot)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = matplotlib.colors.ListedColormap(["#cfcfcf", "#b24fd9"])
    im = ax.pcolormesh(thetas, thetadots, grid, cmap=cmap, shading="auto",
                        vmin=-0.5, vmax=2.5)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title(f"Region of Attraction (N={p.N}, gamma={p.gamma:.3f} rad)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Stopped\n",
                             "Rolling Forward"])
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return thetas, thetadots, grid

# return map (1D Poincare return map with identity line + fixed point)
def plot_return_map(p, fname, v_range=(0.05, 6.0), n=300):
    vs = np.linspace(v_range[0], v_range[1], n)
    v_next = np.array([return_map(v, p) for v in vs])
    valid = np.array([x is not None for x in v_next])
    vs_valid = vs[valid]
    v_next_valid = np.array([x for x in v_next[valid]])

    roots = find_fixed_point(p, bracket=v_range)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(vs_valid, v_next_valid, color="#4f4fd9", lw=2, label="return map")
    ax.plot(v_range, v_range, "k--", lw=1, label="identity")
    for v in roots:
        ax.plot(v, v, "ro", ms=8, zorder=5,
                 label=f"fixed point v*={v:.3f}")
    ax.set_xlabel(r"$\dot\theta_k$ (post-impact, rad/s)")
    ax.set_ylabel(r"$\dot\theta_{k+1}$ (post-impact, rad/s)")
    ax.set_title(f"Return Map (N={p.N}, gamma={p.gamma:.3f})")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return roots

# how roa size & floquet multiplier vary with slope gamma
def sweep_gamma(gammas, N=8, l=1.0, g=9.81):
    fixed_points, floquets, roa_fracs = [], [], []
    for gamma in gammas:
        p = RimlessWheelParams(g=g, l=l, N=N, gamma=gamma)
        roots = find_fixed_point(p)
        if not roots:
            fixed_points.append(np.nan)
            floquets.append(np.nan)
        else:
            v = roots[0]
            fixed_points.append(v)
            floquets.append(floquet_multiplier(v, p))
        _, _, grid = compute_roa_grid(p, thetadot_range=(-6, 6),
                                       n_theta=41, n_thetadot=41)
        roa_fracs.append(np.mean(grid == 1))
    return np.array(fixed_points), np.array(floquets), np.array(roa_fracs)

# how roa size & floquet multiplier vary with spoke amount (N)
def sweep_N(Ns, gamma=0.17, l=1.0, g=9.81):
    fixed_points, floquets, roa_fracs = [], [], []
    for N in Ns:
        p = RimlessWheelParams(g=g, l=l, N=N, gamma=gamma)
        roots = find_fixed_point(p)
        if not roots:
            fixed_points.append(np.nan)
            floquets.append(np.nan)
        else:
            v = roots[0]
            fixed_points.append(v)
            floquets.append(floquet_multiplier(v, p))
        _, _, grid = compute_roa_grid(p, thetadot_range=(-6, 6),
                                       n_theta=41, n_thetadot=41)
        roa_fracs.append(np.mean(grid == 1))
    return np.array(fixed_points), np.array(floquets), np.array(roa_fracs)


if __name__ == "__main__":
    base_p = RimlessWheelParams(g=9.81, l=1.0, N=8, gamma=0.17)

    print("Plotting RoA map...")
    plot_roa(base_p, "figures/roa_map.png")

    print("Plotting return map...")
    roots = plot_return_map(base_p, "figures/return_map.png")
    print("  fixed point(s):", roots)
    for v in roots:
        print("  Floquet multiplier:", floquet_multiplier(v, base_p))

    print("Sweeping gamma...")
    gammas = np.linspace(0.02, 0.35, 12)
    fps, fms, roas = sweep_gamma(gammas, N=8)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(gammas, fms, "o-")
    axs[0].axhline(1, color="r", ls="--", lw=1, label="stability boundary")
    axs[0].set_xlabel("slope gamma (rad)")
    axs[0].set_ylabel("Floquet multiplier")
    axs[0].set_title("Local convergence vs slope")
    axs[0].legend()
    axs[1].plot(gammas, roas, "o-", color="darkorange")
    axs[1].set_xlabel("slope gamma (rad)")
    axs[1].set_ylabel("fraction of grid in forward-rolling basin")
    axs[1].set_title("RoA size vs slope")
    fig.tight_layout()
    fig.savefig("figures/sweep_gamma.png", dpi=150)
    plt.close(fig)

    print("Sweeping N...")
    Ns = np.arange(6, 13)
    fps2, fms2, roas2 = sweep_N(Ns, gamma=0.17)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(Ns, fms2, "o-")
    axs[0].axhline(1, color="r", ls="--", lw=1, label="stability boundary")
    axs[0].set_xlabel("number of spokes N")
    axs[0].set_ylabel("Floquet multiplier")
    axs[0].set_title("Local convergence vs spoke count")
    axs[0].legend()
    axs[1].plot(Ns, roas2, "o-", color="darkorange")
    axs[1].set_xlabel("number of spokes N")
    axs[1].set_ylabel("fraction of grid in forward-rolling basin")
    axs[1].set_title("RoA size vs spoke count")
    fig.tight_layout()
    fig.savefig("figures/sweep_N.png", dpi=150)
    plt.close(fig)

    print("Done. See figures")