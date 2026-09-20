import numpy as np
import matplotlib.pyplot as plt

import inverted_pendulum_walker as model
from poincare_section import simulate_step


def build_lookup_table(thetadot_grid, alpha_grid, params, dt=5e-4, max_time=1.0):
    n_theta = len(thetadot_grid)
    n_alpha = len(alpha_grid)
    next_thetadot = np.full((n_theta, n_alpha), np.nan)
    reached_roa = np.zeros((n_theta, n_alpha), dtype=bool)

    for i, td_k in enumerate(thetadot_grid):
        for j, alpha in enumerate(alpha_grid):
            result = simulate_step(td_k, alpha, params, dt=dt, max_time=max_time)
            reached_roa[i, j] = result["reached_roa"]
            if not result["reached_roa"] and result["thetadot_next"] is not None:
                next_thetadot[i, j] = result["thetadot_next"]

    return next_thetadot, reached_roa
    """
    Returns:
      next_thetadot : (n_thetadot, n_alpha) array, thetadot_{k+1} for
                       each (state, action) pair, or np.nan if that
                       action lands directly in the RoA this step.
      reached_roa    : (n_thetadot, n_alpha) boolean array
    """

def backward_induction(thetadot_grid, next_thetadot, reached_roa, max_steps=15):
    n_theta, n_alpha = next_thetadot.shape
    steps_to_stand = np.full(n_theta, np.inf)
    best_alpha_idx = np.full(n_theta, -1, dtype=int)

    # snap next_thetadot to nearest grid index ahead of time
    next_idx = np.full((n_theta, n_alpha), -1, dtype=int)
    valid = ~np.isnan(next_thetadot)
    next_idx[valid] = np.array([
        np.argmin(np.abs(thetadot_grid - v)) for v in next_thetadot[valid]
    ])

    # step 1: states with SOME action landing directly in the RoA
    for i in range(n_theta):
        if reached_roa[i].any():
            steps_to_stand[i] = 1
            best_alpha_idx[i] = int(np.argmax(reached_roa[i]))

    # iterative backward induction: propagate steps_to_stand+1 through
    # the (state,action)->next_state graph until nothing changes
    for _ in range(max_steps):
        changed = False
        for i in range(n_theta):
            if steps_to_stand[i] != np.inf:
                continue  # already solved with fewer or equal steps
            best = np.inf
            best_j = -1
            for j in range(n_alpha):
                nxt = next_idx[i, j]
                if nxt < 0:
                    continue
                candidate = steps_to_stand[nxt] + 1
                if candidate < best:
                    best = candidate
                    best_j = j
            if best < np.inf:
                steps_to_stand[i] = best
                best_alpha_idx[i] = best_j
                changed = True
        if not changed:
            break

    return steps_to_stand, best_alpha_idx
    """
    Returns:
      steps_to_stand : (n_thetadot,) int array, np.inf if unreachable
                        within max_steps
      best_alpha_idx : (n_thetadot,) int array, index into alpha_grid
                        of the best action at that state (-1 if none)
    """

def grid_resolution_test(params, n_values, alpha_grid_size=15,
                          thetadot_bound=None, dt=5e-4):
    g, l = params["gravity"], params["length"]
    if thetadot_bound is None:
        thetadot_bound = np.sqrt(2 * g / l)

    alpha_grid = np.linspace(np.pi / 8, np.pi / 7, alpha_grid_size)

    results = {}
    for n in n_values:
        thetadot_grid = np.linspace(0.0, thetadot_bound, n)
        next_thetadot, reached_roa = build_lookup_table(thetadot_grid, alpha_grid, params, dt=dt)
        steps, _ = backward_induction(thetadot_grid, next_thetadot, reached_roa)
        results[n] = (thetadot_grid, steps)
        n_reachable = np.sum(np.isfinite(steps))
        print(f"n={n}: {n_reachable}/{n} states reach standing within max_steps")

    # compare each resolution against the finest one, interpolated
    finest_n = max(n_values)
    finest_grid, finest_steps = results[finest_n]
    finite_mask = np.isfinite(finest_steps)
    finest_interp_grid = finest_grid[finite_mask]
    finest_interp_vals = finest_steps[finite_mask]

    print(f"\nComparing against finest resolution (n={finest_n}):")
    for n in n_values:
        if n == finest_n:
            continue
        grid, steps = results[n]
        finite = np.isfinite(steps)
        if not finite.any() or len(finest_interp_grid) < 2:
            print(f"n={n}: not enough data to compare")
            continue
        interp_at_grid = np.interp(grid[finite], finest_interp_grid, finest_interp_vals)
        diffs = np.abs(steps[finite] - interp_at_grid)
        print(f"n={n}: max |step difference| vs finest = {diffs.max():.2f}, "
              f"mean = {diffs.mean():.2f}")

    return results


def plot_steps_to_stand(thetadot_grid, steps_to_stand, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    finite = np.isfinite(steps_to_stand)
    ax.scatter(thetadot_grid[finite], steps_to_stand[finite], c="#2b8cbe", s=30,
               label="reaches standing")
    if (~finite).any():
        ax.scatter(thetadot_grid[~finite], np.zeros(np.sum(~finite)) - 1, c="#c0392b",
                   marker="x", s=40, label="unreachable within max_steps")
    ax.set_xlabel(r"$\dot\theta_k$ at $\theta=0$ crossing (rad/s)")
    ax.set_ylabel("steps to reach standing RoA")
    ax.set_title("Footsteps needed to reach standstill, by initial velocity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    params = model.generate_params()
    roa_data = np.load("figures/roa_controller_grid.npz")
    params["roa_thetas"] = roa_data["thetas"]
    params["roa_thetadots"] = roa_data["thetadots"]
    params["roa_converged"] = roa_data["converged"]

    # grid resolution test 
    print("=== Grid resolution test ===")
    grid_resolution_test(params, n_values=[8, 15, 25, 40])

    # final table at chosen resolution
    print("\n=== Building final lookup table ===")
    g, l = params["gravity"], params["length"]
    thetadot_bound = np.sqrt(2 * g / l)
    thetadot_grid = np.linspace(0.0, thetadot_bound, 25)  # pick based on resolution test above
    alpha_grid = np.linspace(np.pi / 8, np.pi / 7, 15)

    next_thetadot, reached_roa = build_lookup_table(thetadot_grid, alpha_grid, params)
    steps_to_stand, best_alpha_idx = backward_induction(thetadot_grid, next_thetadot, reached_roa)

    for td, s in zip(thetadot_grid, steps_to_stand):
        print(f"thetadot_k={td:.3f}: steps to stand = {s}")

    plot_steps_to_stand(thetadot_grid, steps_to_stand, "figures/steps_to_stand.png")
    print("Saved figures/steps_to_stand.png")

    np.savez("figures/lookup_table.npz",
             thetadot_grid=thetadot_grid, alpha_grid=alpha_grid,
             next_thetadot=next_thetadot, reached_roa=reached_roa,
             steps_to_stand=steps_to_stand, best_alpha_idx=best_alpha_idx)
    print("Saved figures/lookup_table.npz")