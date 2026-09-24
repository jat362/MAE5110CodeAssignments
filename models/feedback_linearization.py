import numpy as np
import matplotlib.pyplot as plt

import inverted_pendulum_walker as model


def ankle_controller(state, params):
    theta, thetadot = state
    g = params["gravity"]
    l = params["length"]
    m = params["mass"]

    kp = params.get("kp", 20.0)
    kd = params.get("kd", 2.0 * np.sqrt(kp))  # critical damping of linearized system

    # tau saturation limits
    tau_min = params.get("tau_min", -0.1 * m * g * l)
    tau_max = params.get("tau_max", 0.05 * m * g * l)

    tau_cancel_gravity = -m * g * l * np.sin(theta)
    tau_stabilize = m * l**2 * (-kp * theta - kd * thetadot)

    # combine and clip to actuator limits
    tau = tau_cancel_gravity + tau_stabilize
    return float(np.clip(tau, tau_min, tau_max))


# continuous dynamics, models on fixed stance leg
# uses rk4
def simulate_closed_loop(state0, params, sim_time=3.0, dt=1e-3):
    state = np.array(state0, dtype=float)
    n_steps = int(sim_time / dt)

    def closed_loop_dynamics(t, state, params):
        params = dict(params)  
        params["ankle_torque"] = ankle_controller(state, params)
        return model.dynamics(t, state, params)

    t = 0.0
    for _ in range(n_steps):
        k1 = closed_loop_dynamics(t, state, params)
        k2 = closed_loop_dynamics(t + dt / 2, state + dt / 2 * k1, params)
        k3 = closed_loop_dynamics(t + dt / 2, state + dt / 2 * k2, params)
        k4 = closed_loop_dynamics(t + dt, state + dt * k3, params)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    return state

# simulate the closed loop forward and check whether converged near origin by sim_time
def grid_search_roa(params, theta_range, thetadot_range,
                     n_theta=41, n_thetadot=41, sim_time=3.0, dt=2e-3,
                     converged_tol=1e-2):
    thetas = np.linspace(*theta_range, n_theta)
    thetadots = np.linspace(*thetadot_range, n_thetadot)
    converged = np.zeros((n_thetadot, n_theta), dtype=bool)

    for i, td0 in enumerate(thetadots):
        for j, th0 in enumerate(thetas):
            final_state = simulate_closed_loop([th0, td0], params, sim_time, dt)
            converged[i, j] = (
                abs(final_state[0]) < converged_tol
                and abs(final_state[1]) < converged_tol
            )

    return thetas, thetadots, converged


def fit_conservative_rectangle(thetas, thetadots, converged):
    theta0_idx = np.argmin(np.abs(thetas))
    thetadot0_idx = np.argmin(np.abs(thetadots))

    theta_max = 0.0
    for j in range(theta0_idx, len(thetas)):
        if not converged[thetadot0_idx, j]:
            break
        theta_max = abs(thetas[j])
    for j in range(theta0_idx, -1, -1):
        if not converged[thetadot0_idx, j]:
            break
        theta_max = min(theta_max, abs(thetas[j]))

    thetadot_max = 0.0
    for i in range(thetadot0_idx, len(thetadots)):
        if not converged[i, theta0_idx]:
            break
        thetadot_max = abs(thetadots[i])
    for i in range(thetadot0_idx, -1, -1):
        if not converged[i, theta0_idx]:
            break
        thetadot_max = min(thetadot_max, abs(thetadots[i]))

    # shrink until the full rectangle is verified converged 
    theta_mask = np.abs(thetas) <= theta_max
    thetadot_mask = np.abs(thetadots) <= thetadot_max
    while theta_max > 0 or thetadot_max > 0:
        box = converged[np.ix_(thetadot_mask, theta_mask)]
        if box.all():
            break
        theta_max *= 0.9
        thetadot_max *= 0.9
        theta_mask = np.abs(thetas) <= theta_max
        thetadot_mask = np.abs(thetadots) <= thetadot_max

    return theta_max, thetadot_max


def roa_event_guard(state, params):
    thetas = params["roa_thetas"]
    thetadots = params["roa_thetadots"]
    converged = params["roa_converged"]

    theta, thetadot = state
    j = int(np.argmin(np.abs(thetas - theta)))
    i = int(np.argmin(np.abs(thetadots - thetadot)))
    return bool(converged[i, j]) # true once state is within the controller's RoA


def plot_roa(thetas, thetadots, converged, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pcolormesh(thetas, thetadots, converged, cmap="Blues", shading="auto")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\theta_0$ (rad)")
    ax.set_ylabel(r"$\dot\theta_0$ (rad/s)")
    ax.set_title("Ankle controller Region of Attraction")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    params = model.generate_params()
    params["angle_of_attack"] = np.pi / 8 

    g, l, m = params["gravity"], params["length"], params["mass"]

    theta_bound = 0.20
    thetadot_bound = 1.5

    thetas, thetadots, converged = grid_search_roa(
        params,
        theta_range=(-theta_bound, theta_bound),
        thetadot_range=(-thetadot_bound, thetadot_bound),
        n_theta=41, n_thetadot=41,
    )

    n_converged = converged.sum()
    print(f"{n_converged} / {converged.size} grid points converged")

    plot_roa(thetas, thetadots, converged, "figures/roa_controller.png")
    print("Saved figures/roa_controller.png")

    # Saved grid as npz so assignment_2.py can load and populate params 
    # for roa_event_guard without recomputing the search every run
    np.savez("figures/roa_controller_grid.npz",
             thetas=thetas, thetadots=thetadots, converged=converged)
    print("Saved figures/roa_controller_grid.npz")