import numpy as np
import matplotlib.pyplot as plt

from models import pendulum as model

#Numerical Integrator
# from integrators import explicit_euler, rk4
from integrators import rk4 as integrator



# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
#integrator = explicit_euler
# integrator = rk4

#original:
#for step, t in enumerate(time_traj[:-1]):
#    state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
#        t, state_traj[:, step], params
#    )

#rk4:
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator(
        model.dynamics, t, state_traj[:, step], params, timestep
)


#Stability Test
timesteps = [1e-5, 1e-4, 1e-3, 1e-2]
stability_results = {}

def simulate(model, integrator, initial_state, params, dt, sim_time):
    n_steps = int(sim_time / dt) + 1
    time = np.arange(n_steps) * dt
    state = np.zeros((2, n_steps))
    state[:, 0] = initial_state

    for step, t in enumerate(time[:-1]):
        state[:, step + 1] = integrator(
            model.dynamics, t, state[:, step], params, dt
        )

    return time, state


def compute_energy(model, state, params):
    return model.calculate_energy(state, params)


for dt in timesteps:
    # run simulation
    time_dt, state_dt = simulate(model, integrator, initial_state, params, dt, sim_time)

    # compute energy
    potential_dt, kinetic_dt = compute_energy(model, state_dt, params)
    total_dt = potential_dt + kinetic_dt

    # stability check: relative energy drift
    E0 = total_dt[0]
    rel_error = abs(np.max(total_dt) - E0) / abs(E0)

    stable = rel_error < 0.01   # 1% tolerance
    stability_results[dt] = stable

print("\nStability Results:")
for dt, stable in stability_results.items():
    print(f"dt = {dt:.0e} → {'stable' if stable else 'unstable'}")


# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

potential_energy, kinetic_energy = model.calculate_energy(state_traj, params)

#plots
plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()


# Time comparison

import timeit

euler_time = timeit.timeit(
    lambda: explicit_euler(model.dynamics, 0, state_traj[:, 0], params, timestep),
    number=1000
)

rk4_time = timeit.timeit(
    lambda: rk4(model.dynamics, 0, state_traj[:, 0], params, timestep),
    number=1000
)

print(f"Euler time: {euler_time:.6f}s")
print(f"RK4 time: {rk4_time:.6f}s")
