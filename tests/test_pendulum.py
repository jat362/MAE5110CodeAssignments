import numpy as np

from integrators import rk4
from models import pendulum


def test_energy_conservation():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    state = np.array([np.pi / 4, 0.0])

    kinetic_energy, potential_energy = pendulum.calculate_energy(state, params)
    initial_energy = kinetic_energy + potential_energy

    timestep = 0.001

    for step in range(100):
        time = step * timestep
        state = rk4(pendulum.dynamics, time, state, timestep, params)
        kinetic_energy, potential_energy = pendulum.calculate_energy(state, params)
        total_energy = kinetic_energy + potential_energy

        assert np.isclose(
            total_energy, initial_energy, rtol=1e-8, atol=1e-10
        )

def test_damping():
    params = pendulum.generate_params()
    params["mass"] = 2.0
    params["length"] = 0.5
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0
    state = np.array([np.pi / 4, 1.0])

    acceleration_without_damping = pendulum.dynamics(0.0, state, params)[1]

    params["damping_coeff"] = 0.2
    acceleration_with_damping = pendulum.dynamics(0.0, state, params)[1]
    assert np.isclose(
        acceleration_with_damping - acceleration_without_damping,
        -0.4,
    )

def test_torque():
    params = pendulum.generate_params()
    params["mass"] = 2.0
    params["length"] = 0.5
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0
    state = np.array([np.pi / 4, 1.0])

    acceleration_without_torque = pendulum.dynamics(0.0, state, params)[1]

    params["torque"] = 0.3
    acceleration_with_torque = pendulum.dynamics(0.0, state, params)[1]
    assert np.isclose(
        acceleration_with_torque - acceleration_without_torque,
        0.6,
    )