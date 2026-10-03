"""Model rimless-wheel motion, impacts, and standing."""

import numpy as np

CONTACT_ANGLE_TOLERANCE = 1e-10  # rad


def create_parameters():
    """Return default physical parameters."""
    return {
        "gravity": 9.81,  # m/s^2
        "spoke_length": 1.0,  # m
        "hub_mass": 1.0,  # kg
        "spoke_count": 8,
        "slope_angle": np.deg2rad(15.0),  # rad
    }


def calculate_state_derivative(time, state, parameters):
    """Calculate the state derivative between impacts."""
    angle, angular_velocity = state

    gravity = parameters["gravity"]
    spoke_length = parameters["spoke_length"]

    angular_acceleration = gravity / spoke_length * np.sin(angle)

    return np.array([angular_velocity, angular_acceleration])


def calculate_contact_angles(parameters):
    """Return the step-start (theta+) and step-end (theta-) angles."""
    half_spoke_angle = np.pi / parameters["spoke_count"]
    slope_angle = parameters["slope_angle"]

    step_start_angle = slope_angle - half_spoke_angle
    step_end_angle = slope_angle + half_spoke_angle
    return step_start_angle, step_end_angle


def detect_contact(state, parameters):
    """Detect outward motion through either contact boundary."""
    angle, angular_velocity = state
    step_start_angle, step_end_angle = calculate_contact_angles(parameters)
    return (angle >= step_end_angle and angular_velocity > 0) or (
        angle <= step_start_angle and angular_velocity < 0
    )


def reset_after_contact(state, parameters):
    """Reset a located contact state."""
    angle, angular_velocity = state
    spoke_spacing = 2 * np.pi / parameters["spoke_count"]
    direction = np.sign(angular_velocity)

    angle_after_contact = angle - direction * spoke_spacing
    angular_velocity_after_contact = angular_velocity * np.cos(spoke_spacing)
    return np.array([angle_after_contact, angular_velocity_after_contact])


def detect_standing(state, parameters, velocity_tolerance):
    """Detect near-rest two-foot support below both tipping barriers."""
    angle, angular_velocity = state
    step_start_angle, step_end_angle = calculate_contact_angles(parameters)
    contact_angles = np.array([step_start_angle, step_end_angle])
    at_contact = np.any(
        np.isclose(angle, contact_angles, rtol=0, atol=CONTACT_ANGLE_TOLERANCE)
    )
    tipping_energy = (
        parameters["gravity"]
        / parameters["spoke_length"]
        * np.min(2 * np.sin(contact_angles / 2) ** 2)
    )
    return (
        step_start_angle < 0 < step_end_angle
        and at_contact
        and abs(angular_velocity) <= velocity_tolerance
        and 0.5 * angular_velocity**2 < tipping_energy
    )


def calculate_energy(state, parameters, contact_height=0.0):
    """Return kinetic and potential energy."""
    angle, angular_velocity = state
    gravity = parameters["gravity"]
    spoke_length = parameters["spoke_length"]
    hub_mass = parameters["hub_mass"]

    hub_height = contact_height + spoke_length * np.cos(angle)
    kinetic_energy = 0.5 * hub_mass * spoke_length**2 * angular_velocity**2
    potential_energy = hub_mass * gravity * hub_height
    return kinetic_energy, potential_energy


def generate_params():
    """Expose the existing parameter factory through the course interface."""
    return create_parameters()


def generate_initial_condition():
    """Return [angle, angular velocity] for an upright, forward-moving wheel."""
    return np.array([0.0, 1.0])


def dynamics(time, state, params):
    """Expose the existing continuous dynamics through the course interface."""
    return calculate_state_derivative(time, state, params)
