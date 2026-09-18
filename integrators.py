def explicit_euler(dynamics, t, state, params, dt):
    """Simple explicit Euler integrator."""
    return state + dt * dynamics(t, state, params)


def rk4(dynamics, t, state, params, dt):
    """Fourth-order Runge–Kutta integrator."""
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + dt/2, state + dt/2 * k1, params)
    k3 = dynamics(t + dt/2, state + dt/2 * k2, params)
    k4 = dynamics(t + dt, state + dt * k3, params)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
