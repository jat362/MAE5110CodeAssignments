import numpy as np
from A1_rimless_wheel import RimlessWheelParams

def step_to_next_guard(theta, thetadot, p: RimlessWheelParams):
    g_over_l = p.g / p.l
    E = 0.5 * thetadot**2 + g_over_l * np.cos(theta)
    theta_fwd = p.gamma + p.alpha
    theta_bwd = p.gamma - p.alpha

# detect equilibirum 
    if theta > 0:
        side = 1
    elif theta < 0:
        side = -1
    else:
        side = 0  # theta exactly 0; side determined by thetadot below

    if thetadot > 0:
        vel_side = 1
    elif thetadot < 0:
        vel_side = -1
    else:
        vel_side = 0

    if side == 0 and vel_side == 0:
        return theta, 0.0, None, "stuck"  # exactly at equilibrium, at rest

    if side == 0:
        side = vel_side  # at theta=0 but moving: use velocity to pick side

    moving_away = (vel_side == side) or (vel_side == 0)

# velocity at the guard
    def vel_at(theta_target):
        return np.sqrt(max(0.0, 2 * (E - g_over_l * np.cos(theta_target))))

    if moving_away:
        target_side = side
    else:
        # moving toward theta=0: does it have enough energy to cross?
        target_side = -side if E >= g_over_l else side

    if target_side > 0:
        return theta_fwd, vel_at(theta_fwd), 0, "forward"
    else:
        return theta_bwd, -vel_at(theta_bwd), 1, "backward"

# instantaneous change when new leg hits ground
def apply_reset(theta, thetadot, which_guard, p: RimlessWheelParams):
    if which_guard == 0:
        theta_new = theta - 2 * p.alpha
    else:
        theta_new = theta + 2 * p.alpha
    thetadot_new = thetadot * np.cos(2 * p.alpha)
    return theta_new, thetadot_new


def run_to_steady_state(theta0, thetadot0, p: RimlessWheelParams,
                         max_steps=2000, zeno_tol=1e-4):
    theta, thetadot = theta0, thetadot0
    seq = []
    for i in range(max_steps):
        theta, thetadot, which_guard, status = step_to_next_guard(theta, thetadot, p)
        if status == "stuck":
            return {"label": "stuck", "thetadot_sequence": seq, "n_impacts": i}
        theta, thetadot = apply_reset(theta, thetadot, which_guard, p)
        seq.append(thetadot)
        if abs(thetadot) < zeno_tol:
            return {"label": "stopped", "thetadot_sequence": seq, "n_impacts": i + 1}
    # ran out of steps still impacting -> converged to a rolling limit cycle
    label = "rolling_forward" if seq[-1] > 0 else "rolling_backward"
    return {"label": label, "thetadot_sequence": seq, "n_impacts": max_steps}



# region of attraction

def compute_roa_grid(p: RimlessWheelParams, theta_range=None,
                      thetadot_range=(-8, 8), n_theta=121, n_thetadot=121):
   
    if theta_range is None:
        theta_range = (p.gamma - p.alpha, p.gamma + p.alpha)

    thetas = np.linspace(theta_range[0], theta_range[1], n_theta)
    thetadots = np.linspace(thetadot_range[0], thetadot_range[1], n_thetadot)

    label_map = {"stuck": 0, "stopped": 0, "rolling_forward": 1,
                 "rolling_backward": 2}
    grid = np.zeros((n_thetadot, n_theta), dtype=int)

    for i, td0 in enumerate(thetadots):
        for j, th0 in enumerate(thetas):
            result = run_to_steady_state(th0, td0, p)
            grid[i, j] = label_map[result["label"]]

    return thetas, thetadots, grid


# 1D Poincare return map
# thetadot_k -> thetadot_{k+1} at the forward guard
def return_map(thetadot_in, p: RimlessWheelParams):
    theta0 = p.gamma - p.alpha  # position right after landing
    theta_next, thetadot_next, which_guard, status = step_to_next_guard(
        theta0, thetadot_in, p)
    if status != "forward":
        return None
    _, thetadot_post = apply_reset(theta_next, thetadot_next, which_guard, p)
    return thetadot_post


def find_fixed_point(p: RimlessWheelParams, bracket=(0.1, 10.0), n=400):
    from scipy.optimize import brentq

    def f(v):
        r = return_map(v, p)
        if r is None:
            return np.nan
        return r - v

    vs = np.linspace(bracket[0], bracket[1], n)
    fs = np.array([f(v) for v in vs])
    valid = ~np.isnan(fs)
    vs, fs = vs[valid], fs[valid]

# fixed point
    roots = []
    for i in range(len(vs) - 1):
        if fs[i] == 0:
            roots.append(vs[i])
        elif fs[i] * fs[i + 1] < 0:
            roots.append(brentq(f, vs[i], vs[i + 1]))
    return roots 


def floquet_multiplier(v_star, p: RimlessWheelParams, eps=1e-4):
#|slope| < 1 => locally stable limit cycle.
    
    v_plus = return_map(v_star + eps, p)
    v_minus = return_map(v_star - eps, p)
    return (v_plus - v_minus) / (2 * eps)