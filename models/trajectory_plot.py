import numpy as np
import matplotlib.pyplot as plt

import inverted_pendulum_walker as model
from feedback_linearization import ankle_controller, roa_event_guard


def max_steps_backward_induction(thetadot_grid, next_thetadot, reached_roa, max_steps=200):
    n_theta, n_alpha = next_thetadot.shape
    max_steps_to_stand = np.full(n_theta, np.nan)  # nan = "not yet solved"
    worst_alpha_idx = np.full(n_theta, -1, dtype=int)

    next_idx = np.full((n_theta, n_alpha), -1, dtype=int)
    valid = ~np.isnan(next_thetadot)
    next_idx[valid] = np.array([
        np.argmin(np.abs(thetadot_grid - v)) for v in next_thetadot[valid]
    ])


    can_avoid_roa_entirely = np.zeros(n_theta, dtype=bool)
    has_only_roa_actions = np.array([reached_roa[i].all() for i in range(n_theta)])
    max_steps_to_stand[has_only_roa_actions] = 1
    worst_alpha_idx[has_only_roa_actions] = 0

    for _ in range(max_steps):
        changed = False
        for i in range(n_theta):
            if not np.isnan(max_steps_to_stand[i]) or can_avoid_roa_entirely[i]:
                continue
            all_solved = True
            worst = -np.inf
            worst_j = -1
            for j in range(n_alpha):
                if reached_roa[i, j]:
                    candidate = 1
                else:
                    nxt = next_idx[i, j]
                    if nxt < 0:
                        all_solved = False
                        continue
                    if np.isnan(max_steps_to_stand[nxt]):
                        all_solved = False
                        continue
                    candidate = max_steps_to_stand[nxt] + 1
                if candidate > worst:
                    worst = candidate
                    worst_j = j
            if all_solved:
                max_steps_to_stand[i] = worst
                worst_alpha_idx[i] = worst_j
                changed = True
        if not changed:
            break

    # anything still unsolved after the sweep is treated as able to
    # stall indefinitely (a cycle in the graph avoids the RoA forever)
    still_unsolved = np.isnan(max_steps_to_stand)
    max_steps_to_stand[still_unsolved] = np.inf

    return max_steps_to_stand, worst_alpha_idx


def simulate_step_with_trace(thetadot_k, alpha, params, dt=1e-4, max_time=2.0):
    step_params = dict(params)
    step_params["angle_of_attack"] = alpha
    step_params["ankle_torque"] = 0.0

    state = np.array([0.0, thetadot_k])
    t = 0.0
    n_impacts = 0
    ts, thetas, thetadots, taus = [0.0], [state[0]], [state[1]], [0.0]
    impact_times = []

    n_steps = int(max_time / dt)
    for _ in range(n_steps):
        if roa_event_guard(state, params):
            return {"thetadot_next": None, "reached_roa": True, "n_impacts": n_impacts,
                    "t": np.array(ts), "theta": np.array(thetas),
                    "thetadot": np.array(thetadots), "tau": np.array(taus),
                    "impact_times": impact_times}

        next_state = state + dt * model.dynamics(t, state, step_params)

        if model.event_guard(state, next_state, step_params):
            next_state = model.event_dynamics(next_state, step_params)
            n_impacts += 1
            impact_times.append(t + dt)
        else:
            if n_impacts >= 1:
                if state[0] < 0.0 <= next_state[0] or state[0] > 0.0 >= next_state[0]:
                    frac = abs(state[0]) / (abs(state[0]) + abs(next_state[0]) + 1e-15)
                    thetadot_cross = state[1] + frac * (next_state[1] - state[1])
                    ts.append(t + dt); thetas.append(0.0); thetadots.append(thetadot_cross); taus.append(0.0)
                    return {"thetadot_next": thetadot_cross, "reached_roa": False,
                            "n_impacts": n_impacts, "t": np.array(ts), "theta": np.array(thetas),
                            "thetadot": np.array(thetadots), "tau": np.array(taus),
                            "impact_times": impact_times}

        state = next_state
        t += dt
        ts.append(t); thetas.append(state[0]); thetadots.append(state[1]); taus.append(0.0)

    return {"thetadot_next": None, "reached_roa": False, "n_impacts": n_impacts,
            "t": np.array(ts), "theta": np.array(thetas),
            "thetadot": np.array(thetadots), "tau": np.array(taus),
            "impact_times": impact_times}


def trace_full_trajectory(thetadot_k0, thetadot_grid, alpha_grid, alpha_idx_table, params,
                           dt=1e-4, settle_time=1.5, settle_tol=1e-3, max_steps_cap=20):
    all_t, all_theta, all_thetadot, all_tau = [], [], [], []
    all_impact_times = []
    t_offset = 0.0
    thetadot_k = thetadot_k0
    n_impacts_total = 0
    reached_roa = False
    alphas_used = []

    for step_i in range(max_steps_cap):
        idx = int(np.argmin(np.abs(thetadot_grid - thetadot_k)))
        j = alpha_idx_table[idx]
        if j < 0:
            break
        alpha = alpha_grid[j]
        alphas_used.append(alpha)

        result = simulate_step_with_trace(thetadot_k, alpha, params, dt=dt)

        seg_t = result["t"] + t_offset
        all_t.append(seg_t)
        all_theta.append(result["theta"])
        all_thetadot.append(result["thetadot"])
        all_tau.append(result["tau"])
        all_impact_times.extend([it + t_offset for it in result["impact_times"]])
        n_impacts_total += result["n_impacts"]

        t_offset = seg_t[-1]

        if result["reached_roa"]:
            reached_roa = True
            break
        if result["thetadot_next"] is None:
            break  # ran out of time mid-step without resolving -- stop here
        thetadot_k = result["thetadot_next"]

    # settling tail: continuous closed-loop integration with the ankle controller
    if reached_roa:
        state = np.array([all_theta[-1][-1], all_thetadot[-1][-1]])
        step_params = dict(params)
        n_settle_steps = int(settle_time / dt)
        seg_t, seg_theta, seg_thetadot, seg_tau = [], [], [], []
        t = t_offset
        for _ in range(n_settle_steps):
            tau = ankle_controller(state, step_params)
            step_params["ankle_torque"] = tau
            state = state + dt * model.dynamics(t, state, step_params)
            t += dt
            seg_t.append(t); seg_theta.append(state[0]); seg_thetadot.append(state[1]); seg_tau.append(tau)
            if abs(state[0]) < settle_tol and abs(state[1]) < settle_tol:
                break
        all_t.append(np.array(seg_t))
        all_theta.append(np.array(seg_theta))
        all_thetadot.append(np.array(seg_thetadot))
        all_tau.append(np.array(seg_tau))

    return {
        "t": np.concatenate(all_t), "theta": np.concatenate(all_theta),
        "thetadot": np.concatenate(all_thetadot), "tau": np.concatenate(all_tau),
        "impact_times": all_impact_times, "n_impacts": n_impacts_total,
        "reached_roa": reached_roa, "alphas_used": alphas_used,
    }


def alpha_sequence_from_policy(thetadot_k0, thetadot_grid, alpha_grid, next_thetadot,
                                reached_roa, alpha_idx_table, max_steps=10):
    idx0 = int(np.argmin(np.abs(thetadot_grid - thetadot_k0)))
    idx = idx0
    alphas = []
    for _ in range(max_steps):
        j = alpha_idx_table[idx]
        if j < 0:
            break
        alphas.append(alpha_grid[j])
        if reached_roa[idx, j]:
            break
        nxt = next_thetadot[idx, j]
        if np.isnan(nxt):
            break
        idx = int(np.argmin(np.abs(thetadot_grid - nxt)))
    return alphas


def trace_fixed_sequence_trajectory(thetadot_k0, alpha_sequence, params, dt=1e-4,
                                     settle_time=1.5, settle_tol=1e-3):
    all_t, all_theta, all_thetadot, all_tau = [], [], [], []
    all_impact_times = []
    t_offset = 0.0
    thetadot_k = thetadot_k0
    n_impacts_total = 0
    reached_roa = False

    for alpha in alpha_sequence:
        result = simulate_step_with_trace(thetadot_k, alpha, params, dt=dt)

        seg_t = result["t"] + t_offset
        all_t.append(seg_t)
        all_theta.append(result["theta"])
        all_thetadot.append(result["thetadot"])
        all_tau.append(result["tau"])
        all_impact_times.extend([it + t_offset for it in result["impact_times"]])
        n_impacts_total += result["n_impacts"]

        t_offset = seg_t[-1]

        if result["reached_roa"]:
            reached_roa = True
            break
        if result["thetadot_next"] is None:
            break
        thetadot_k = result["thetadot_next"]

    if reached_roa:
        state = np.array([all_theta[-1][-1], all_thetadot[-1][-1]])
        step_params = dict(params)
        n_settle_steps = int(settle_time / dt)
        seg_t, seg_theta, seg_thetadot, seg_tau = [], [], [], []
        t = t_offset
        for _ in range(n_settle_steps):
            tau = ankle_controller(state, step_params)
            step_params["ankle_torque"] = tau
            state = state + dt * model.dynamics(t, state, step_params)
            t += dt
            seg_t.append(t); seg_theta.append(state[0]); seg_thetadot.append(state[1]); seg_tau.append(tau)
            if abs(state[0]) < settle_tol and abs(state[1]) < settle_tol:
                break
        all_t.append(np.array(seg_t))
        all_theta.append(np.array(seg_theta))
        all_thetadot.append(np.array(seg_thetadot))
        all_tau.append(np.array(seg_tau))

    return {
        "t": np.concatenate(all_t), "theta": np.concatenate(all_theta),
        "thetadot": np.concatenate(all_thetadot), "tau": np.concatenate(all_tau),
        "impact_times": all_impact_times, "n_impacts": n_impacts_total,
        "reached_roa": reached_roa,
    }

# overlay plots
def plot_trajectory_comparison(traj_a, label_a, traj_b, label_b, title, fname):
    color_a, color_b = "#682bbe", "#57e391"
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    axs[0].plot(traj_a["t"], traj_a["theta"], color=color_a, label=label_a)
    axs[0].plot(traj_b["t"], traj_b["theta"], color=color_b, label=label_b)
    for it in traj_a["impact_times"]:
        axs[0].axvline(it, color=color_a, lw=0.8, alpha=0.4, ls="--")
    for it in traj_b["impact_times"]:
        axs[0].axvline(it, color=color_b, lw=0.8, alpha=0.4, ls=":")
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel(r"$\theta$ (rad)")
    axs[0].legend(fontsize=8)

    axs[1].plot(traj_a["t"], traj_a["thetadot"], color=color_a, label=label_a)
    axs[1].plot(traj_b["t"], traj_b["thetadot"], color=color_b, label=label_b)
    for it in traj_a["impact_times"]:
        axs[1].axvline(it, color=color_a, lw=0.8, alpha=0.4, ls="--")
    for it in traj_b["impact_times"]:
        axs[1].axvline(it, color=color_b, lw=0.8, alpha=0.4, ls=":")
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel(r"$\dot\theta$ (rad/s)")

    axs[2].plot(traj_a["t"], traj_a["tau"], color=color_a, label=label_a)
    axs[2].plot(traj_b["t"], traj_b["tau"], color=color_b, label=label_b)
    axs[2].set_ylabel(r"$\tau$ (N m)")
    axs[2].set_xlabel("t (s)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    params = model.generate_params()

    roa_data = np.load("figures/roa_controller_grid.npz")
    params["roa_thetas"] = roa_data["thetas"]
    params["roa_thetadots"] = roa_data["thetadots"]
    params["roa_converged"] = roa_data["converged"]

    table = np.load("figures/lookup_table.npz")
    thetadot_grid = table["thetadot_grid"]
    alpha_grid = table["alpha_grid"]
    next_thetadot = table["next_thetadot"]
    reached_roa = table["reached_roa"]
    steps_to_stand = table["steps_to_stand"]
    best_alpha_idx = table["best_alpha_idx"]

    # Pick an initial condition needing >= 3 steps
    # under the fastest-path policy.
    candidates = np.where(steps_to_stand >= 3)[0]

    if len(candidates) == 0:
        raise RuntimeError(
            "No grid state needs >=3 steps -- "
            "widen thetadot_bound or rerun lookup_table.py"
        )

    idx0 = candidates[0]
    thetadot_k0 = thetadot_grid[idx0]

    print(
        f"Chosen initial condition: thetadot_k0 = "
        f"{thetadot_k0:.4f} rad/s "
        f"(fastest-path steps = {steps_to_stand[idx0]:.0f})"
    )
    # Fastest-path trajectory

    fast_traj = trace_full_trajectory(
        thetadot_k0,
        thetadot_grid,
        alpha_grid,
        best_alpha_idx,
        params,
    )


    # Worst-case trajectory

    max_steps_arr, worst_alpha_idx = max_steps_backward_induction(
        thetadot_grid,
        next_thetadot,
        reached_roa,
    )

    worst_steps_here = max_steps_arr[idx0]

    print(
        f"Maximum steps before RoA from this same "
        f"initial condition: {worst_steps_here}"
    )

    if np.isfinite(worst_steps_here):

        worst_traj = trace_full_trajectory(
            thetadot_k0,
            thetadot_grid,
            alpha_grid,
            worst_alpha_idx,
            params,
            max_steps_cap=int(worst_steps_here) + 3,
        )


        plot_trajectory_comparison(
            fast_traj,
            f"Fastest-path ({fast_traj['n_impacts']} steps)",
            worst_traj,
            f"Worst-case ({worst_traj['n_impacts']} steps)",
            (
                f"Comparison Trajectory Plot "
                f"(theta_k0={thetadot_k0:.3f} rad/s)"
            ),
            "figures/trajectory_comparison.png",
        )

    else:
        print(
            "This initial condition can avoid the RoA indefinitely "
            "under some policy (max_steps_to_stand = inf)."
        )