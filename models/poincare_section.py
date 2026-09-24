import numpy as np

import inverted_pendulum_walker as model
from feedback_linearization import roa_event_guard


def simulate_step(thetadot_k, alpha, params, dt=1e-4, max_time=2.0):
    step_params = dict(params)
    step_params["angle_of_attack"] = alpha
    step_params["ankle_torque"] = 0.0

    state = np.array([0.0, thetadot_k])
    t = 0.0
    n_impacts = 0
    crossed_theta_zero_since_impact = False

    n_steps = int(max_time / dt)
    for _ in range(n_steps):
        if roa_event_guard(state, step_params):
            return {"thetadot_next": None, "reached_roa": True, "n_impacts": n_impacts}

        next_state = state + dt * model.dynamics(t, state, step_params)

        if model.event_guard(state, next_state, step_params):
            next_state = model.event_dynamics(next_state, step_params)
            n_impacts += 1
            crossed_theta_zero_since_impact = False
        else:
            # detect a theta=0 crossing since the last impact (sign change)
            if crossed_theta_zero_since_impact is False and n_impacts >= 1:
                if state[0] < 0.0 <= next_state[0] or state[0] > 0.0 >= next_state[0]:
                    # linear interpolation for the crossing velocity
                    frac = abs(state[0]) / (abs(state[0]) + abs(next_state[0]) + 1e-15)
                    thetadot_cross = state[1] + frac * (next_state[1] - state[1])
                    return {"thetadot_next": thetadot_cross, "reached_roa": False,
                            "n_impacts": n_impacts}

        state = next_state
        t += dt

    # failure case: walker never completed a step, never crossed theta = 0,
    # never reached the RoA within the allowed simulation time
    return {"thetadot_next": None, "reached_roa": False, "n_impacts": n_impacts}