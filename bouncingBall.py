import numpy as np
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

# bouncing ball model
def bouncing_ball_dynamics(t, state, params):
    """
    state = [y, v]
    y: height
    v: vertical velocity
    """
    y, v = state
    g = params["gravity"]

    dydt = v
    dvdt = -g

    return np.array([dydt, dvdt])


def bouncing_ball_energy(state, params):
    y, v = state
    m = params["mass"]
    g = params["gravity"]

    potential = m * g * max(y, 0)
    kinetic = 0.5 * m * v**2

    return potential, kinetic

# cleanup for future
# integrators
def explicit_euler(dynamics, t, state, params, dt):
    return state + dt * dynamics(t, state, params)


def rk4(dynamics, t, state, params, dt):
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + dt/2, state + dt/2 * k1, params)
    k3 = dynamics(t + dt/2, state + dt/2 * k2, params)
    k4 = dynamics(t + dt, state + dt * k3, params)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)



params = {
    "gravity": 9.81,
    "mass": 1.0,
    "restitution": 0.8
}

initial_state = np.array([1.0, 0.0])   # start 1 meter above ground

timestep = 1e-4
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

integrator = explicit_euler  


# simulation loop
for step, t in enumerate(time_traj[:-1]):

    # integrate normally
    state_next = integrator(
        bouncing_ball_dynamics, t, state_traj[:, step], params, timestep
    )

    y, v = state_next

    # bounce after integration
    if y <= 0 and v < 0:
        v = -params["restitution"] * v
        y = 0

    state_traj[:, step + 1] = np.array([y, v])


# energy
potential_energy = np.zeros(n_timesteps)
kinetic_energy = np.zeros(n_timesteps)

for i in range(n_timesteps):
    pe, ke = bouncing_ball_energy(state_traj[:, i], params)
    potential_energy[i] = pe
    kinetic_energy[i] = ke


# plots 
plt.figure(figsize=(8, 5))
plt.plot(time_traj, potential_energy, label="Potential Energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic Energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total Energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing Ball Energy vs Time")
plt.legend()
plt.tight_layout()
plt.show()
