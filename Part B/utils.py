"""Utility functions for particle filter assignment.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury

Adapted by Oliver Dale.

"""

import numpy as np
import bisect
from numpy.random import uniform


def gauss(v, mu=0, sigma=1):
    """Evaluate a Gaussian PDF."""
    
    return np.exp(-0.5 * ((v - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)


###############################################################################
# Particle filter functions

def resample(particles, weights):
    """Resample particles in proportion to their weights.

    Particles and weights should be arrays, and will be updated in place.
    """
    cum_weights = np.cumsum(weights)

    if cum_weights[-1] == 0.0:
        print('All weights are zero, giving up...')
        return False

    cum_weights = cum_weights/cum_weights[-1]

    new_particles = []
    for _ in particles:
        # Copy a particle into the list of new particles, choosing based
        # on weight
        m = bisect.bisect_left(cum_weights, uniform(0, 1))
        p = particles[m]
        new_particles.append(p)

    # Add random particles
    weights_sum = np.sum(weights)
    w = weights / weights_sum
    particle_range = np.sum(w**2)

    for i in range(0, 10):
        new_particles[i] = new_particles[i] + uniform(-particle_range, particle_range)
    
    # Replace old particles with new particles
    for m, p in enumerate(new_particles):
        particles[m] = p

    # Reset weights
    weights[:] = 1
    return [particles, weights]


def is_degenerate(weights):
    """Return true if the particles are degenerate and need resampling."""
    
    weights_sum = np.sum(weights)
    w = weights / weights_sum
    print("particle check: ", np.sum(w**2))
    return 1.0 / np.sum(w**2) < 0.5 * len(w)


def a_last_resort(particles,weights):
    """Resample particles randomly in an attempt to find the true position of the robot.

    Particles and weights should be arrays, and will be updated in place.
    """

    cum_weights = np.cumsum(weights)

    if cum_weights[-1] == 0.0:
        print('All weights are zero, giving up...')
        return False

    cum_weights = cum_weights/cum_weights[-1]

    new_particles = []
    for _ in particles:
        # Copy a particle into the list of new particles, choosing based
        # on weight
        m = bisect.bisect_left(cum_weights, uniform(0, 1))
        p = particles[m]
        new_particles.append(p)

    # Add random particles with a distribution proportional to the squared sum of the normalised weights. 
    weights_sum = np.sum(weights)
    w = weights / weights_sum
    particle_range = np.sum(w**2)

    for i in range(0, len(particles)):
        new_particles[i] = new_particles[i] + uniform(-particle_range, particle_range)
    
    # Replace old particles with new particles
    for m, p in enumerate(new_particles):
        particles[m] = p

    # Reset weights
    weights[:] = 1

    return [particles, weights]


###############################################################################
# Functions for working with angles (in radians)

def wrapto2pi(angle):
    """Convert angle into range [0, 2 * pi)."""
    return angle % (2 * np.pi)


def wraptopi(angle):
    """Convert angle into range [-pi, pi)."""
    return wrapto2pi(angle + np.pi) - np.pi


def angle_difference(angle1, angle2):
    """Return principal difference angle, angle2 - angle1  with return value in range [-pi, pi)."""
    return ((((angle2 - angle1) % (2 * np.pi)) + (3 * np.pi)) % (2 * np.pi)) - np.pi
