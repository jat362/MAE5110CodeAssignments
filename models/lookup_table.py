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


def verify_policy_against_continuous_dynamics(thetadot_grid, alpha_grid, best_alpha_idx,
                                               steps_to_stand, params, test_velocities,
                                               dt=1e-4, max_steps_cap=12):
    results = []
    for v0 in test_velocities:
        idx0 = int(np.argmin(np.abs(thetadot_grid - v0)))
        predicted_steps = steps_to_stand[idx0]

        thetadot_k = v0
        actual_steps = 0
        reached = False
        for _ in range(max_steps_cap):
            idx = int(np.argmin(np.abs(thetadot_grid - thetadot_k)))
            j = best_alpha_idx[idx]
            if j < 0:
                break
            alpha = alpha_grid[j]
            result = simulate_step(thetadot_k, alpha, params, dt=dt)
            actual_steps += 1
            if result["reached_roa"]:
                reached = True
                break
            if result["thetadot_next"] is None:
                break
            thetadot_k = result["thetadot_next"]

        results.append({
            "v0": v0, "predicted_steps": predicted_steps,
            "actual_steps": actual_steps if reached else np.inf,
            "reached": reached,
        })

    n = len(results)
    fail_rate = sum(not r["reached"] for r in results) / n
    broken_promise_rate = sum(
        r["reached"] and r["actual_steps"] > r["predicted_steps"] for r in results
    ) / n
    exact_match_rate = sum(
        r["reached"] and r["actual_steps"] == r["predicted_steps"] for r in results
    ) / n
    worst_actual = max((r["actual_steps"] for r in results if r["reached"]), default=np.inf)

    return {
        "fail_rate": fail_rate, "broken_promise_rate": broken_promise_rate,
        "exact_match_rate": exact_match_rate, "worst_actual": worst_actual,
        "details": results,
    }


def grid_resolution_test(
    params,
    n_values,
    alpha_grid_size=15,
    thetadot_bound=None,
    dt=5e-4,
    n_test_points=100,
):

    g = params["gravity"]
    l = params["length"]

    if thetadot_bound is None:
        thetadot_bound = np.sqrt(2 * g / l)

    alpha_grid = np.linspace(
        np.pi / 8,
        np.pi / 7,
        alpha_grid_size,
    )

    # Use fixed off-grid verification points distributed across
    # the full angular-velocity range
    dv_test = thetadot_bound / n_test_points

    test_velocities = (
        np.arange(n_test_points) + 0.5
    ) * dv_test

    print(
        f"\nTesting {n_test_points} fixed off-grid velocities "
        f"from {test_velocities[0]:.3f} to "
        f"{test_velocities[-1]:.3f} rad/s\n"
    )

    header = (
        f"{'n':>5}"
        f"{'fail':>12}"
        f"{'broken':>12}"
        f"{'exact':>12}"
        f"{'max error':>12}"
        f"{'worst steps':>14}"
    )

    print(header)
    print("-" * len(header))

    summary = {}

    for n in n_values:

        thetadot_grid = np.linspace(
            0.0,
            thetadot_bound,
            n,
        )

        next_thetadot, reached_roa = build_lookup_table(
            thetadot_grid,
            alpha_grid,
            params,
            dt=dt,
        )

        steps, best_alpha_idx = backward_induction(
            thetadot_grid,
            next_thetadot,
            reached_roa,
        )

        verification = verify_policy_against_continuous_dynamics(
            thetadot_grid,
            alpha_grid,
            best_alpha_idx,
            steps,
            params,
            test_velocities,
        )

        # Calculate the maximum difference between the predicted
        # and actual number of steps.
        step_errors = []

        for result in verification["details"]:

            if (
                result["reached"]
                and np.isfinite(result["predicted_steps"])
                and np.isfinite(result["actual_steps"])
            ):
                error = abs(
                    result["actual_steps"]
                    - result["predicted_steps"]
                )

                step_errors.append(error)

        if len(step_errors) > 0:
            max_abs_error = np.max(step_errors)
        else:
            max_abs_error = np.inf

        verification["max_abs_error"] = max_abs_error

        summary[n] = verification

        print(
            f"{n:>5}"
            f"{verification['fail_rate']:>12.1%}"
            f"{verification['broken_promise_rate']:>12.1%}"
            f"{verification['exact_match_rate']:>12.1%}"
            f"{max_abs_error:>12.1f}"
            f"{verification['worst_actual']:>14}"
        )

    return summary


def plot_steps_to_stand(thetadot_grid, steps_to_stand, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    finite = np.isfinite(steps_to_stand)
    ax.scatter(thetadot_grid[finite], steps_to_stand[finite], c="#2b8cbe", s=30,
               label="reaches standing")
    if (~finite).any():
        ax.scatter(thetadot_grid[~finite], np.zeros(np.sum(~finite)) - 1, c="#c0392b",
                   marker="x", s=40, label="unreachable within max_steps")
    ax.set_xlabel(r"$\dot\theta$ at $\theta=0$ crossing (rad/s)")
    ax.set_ylabel("Steps to reach standing RoA")
    ax.set_title("Steps to Standstill")
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
    print("=== Grid resolution test (verified against real continuous dynamics) ===")
    resolution_results = grid_resolution_test(
        params,
        n_values=[10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
        n_test_points=100,
    )

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