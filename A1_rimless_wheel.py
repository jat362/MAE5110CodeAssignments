import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class RimlessWheelParams:
    g: float = 9.81
    l: float = 1.0
    N: int = 8
    gamma: float = 0.17  # slope angle: radians (~10 deg)

    #alpha: half angle between spokes
    @property
    def alpha(self):
        return np.pi / self.N

#simple pendulum
def dynamics(t, state, p: RimlessWheelParams):
    theta, thetadot = state
    thetaddot = (p.g / p.l) * np.sin(theta)
    return [thetadot, thetaddot]


def energy(theta, thetadot, p: RimlessWheelParams):
    return 0.5 * thetadot**2 + (p.g / p.l) * np.cos(theta)


def thetadot_from_energy(theta, E, p: RimlessWheelParams, sign=+1):
    v2 = 2.0 * (E - (p.g / p.l) * np.cos(theta))
    if v2 < 0:
        return None
    return sign * np.sqrt(v2)


# when leg hits ground
def make_events(p: RimlessWheelParams):
    def forward_guard(t, state, p):
        return state[0] - (p.gamma + p.alpha)
    forward_guard.terminal = True
    forward_guard.direction = 1  # only trigger when increasing through 0

    def backward_guard(t, state, p):
        return state[0] - (p.gamma - p.alpha)
    backward_guard.terminal = True
    backward_guard.direction = -1  # only trigger when decreasing through 0

    return [forward_guard, backward_guard]

# impact
def reset_map(state, p: RimlessWheelParams, which_guard: int):
    
    # which_guard: 0 = forward guard hit (theta = gamma+alpha, thetadot>0)
    # 1 = backward guard hit (theta = gamma-alpha, thetadot<0)
    
    theta, thetadot = state
    if which_guard == 0:
        theta_new = theta - 2 * p.alpha
    else:
        theta_new = theta + 2 * p.alpha
    thetadot_new = thetadot * np.cos(2 * p.alpha)
    return np.array([theta_new, thetadot_new])


def simulate(state0, t_final, p: RimlessWheelParams, max_impacts=500,
             max_step=1e-3, zeno_vel_tol=1e-5):
    
    events = make_events(p)
    t0 = 0.0
    x0 = np.array(state0, dtype=float)
    t_all = [np.array([t0])]
    x_all = [x0.reshape(1, 2)]
    impacts = []
    status = "no_impact"

    for _ in range(max_impacts):
        t_remaining = t_final - t0
        if t_remaining <= 0:
            status = "rolling" if impacts else "no_impact"
            break
        sol = solve_ivp(dynamics, [t0, t0 + t_remaining], x0, args=(p,),
                         events=events, max_step=max_step, rtol=1e-9,
                         atol=1e-11, dense_output=False)
        t_all.append(sol.t[1:])
        x_all.append(sol.y[:, 1:].T)

        if sol.status != 1:
            # reached t_final with no more impacts in this swing
            status = "rolling" if impacts else "no_impact"
            break

        which_guard = 0 if len(sol.t_events[0]) > 0 else 1
        t_impact = sol.t_events[which_guard][0]
        x_pre = sol.y_events[which_guard][0]
        x_post = reset_map(x_pre, p, which_guard)
        impacts.append((t_impact, x_pre.copy(), x_post.copy(), which_guard))

        if abs(x_pre[1]) < zeno_vel_tol:
            status = "stopped"
            break

        t0 = t_impact
        x0 = x_post
    else:
        status = "rolling"  # hit max_impacts while still going strong

    t_arr = np.concatenate(t_all)
    x_arr = np.concatenate(x_all, axis=0)
    return {"t": t_arr, "x": x_arr, "impacts": impacts, "status": status}