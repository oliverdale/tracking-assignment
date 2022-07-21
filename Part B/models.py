"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import arctan, cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference


def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

    M = particle_poses.shape[0]

    dt = dt/(10**9)

    for m in range(M):
        phi = particle_poses[m,2]

        # # Velocity motion model
        # if (abs(speed_command[1]) < 0.0001 and abs(speed_command[0]) > 0.0001):
        #     particle_poses[m, 0] += speed_command[0] * dt * cos(particle_poses[m, 2]) + 0.01 * randn()
        #     particle_poses[m, 1] += speed_command[0] * dt * sin(particle_poses[m, 2]) + 0.01 * randn()
        #     particle_poses[m, 2] = wraptopi(phi + 0.001 * randn()) 

        # elif (abs(speed_command[1]) > 0.0001 and abs(speed_command[0]) < 0.0001):
        #     particle_poses[m, 0] += (0) + 0.01 * randn()
        #     particle_poses[m, 1] += (0) + 0.01 * randn()
        #     particle_poses[m, 2] = wraptopi(phi + speed_command[1] * dt + 0.001 * randn())    

        # elif (abs(speed_command[1]) > 0.0001 and abs(speed_command[0]) > 0.0001):
        #     particle_poses[m, 0] += (-speed_command[0]/speed_command[1])*sin(phi) + (speed_command[0]/speed_command[1])*sin(phi + speed_command[1] * dt) + 0.01 * randn()
        #     particle_poses[m, 1] += (speed_command[0]/speed_command[1])*cos(phi) - (speed_command[0]/speed_command[1])*cos(phi + speed_command[1] * dt) + 0.01 * randn()
        #     particle_poses[m, 2] = wraptopi(phi + speed_command[1] * dt+ 0.001 * randn()) 

        lphi1 = arctan2((odom_pose[1] - odom_pose_prev[1]),(odom_pose[0] - odom_pose_prev[0])) - odom_pose_prev[2]
        d = sqrt((odom_pose[1] - odom_pose_prev[1])**2 + (odom_pose[0] - odom_pose_prev[0])**2)
        lphi2 = odom_pose[2] - odom_pose_prev[2] - lphi1

        phi1_std =0.001
        d_std = 0.02
        phi2_std = 0.001

        rand_lphi1 = np.random.normal(lphi1, phi1_std, 1)
        rand_d = np.random.normal(d, d_std, 1)
        rand_lphi2 = np.random.normal(lphi2, phi2_std, 1)

        particle_poses[m, 0] += rand_d * cos(particle_poses[m, 2] + rand_lphi1) 
        particle_poses[m, 1] += rand_d * sin(particle_poses[m, 2] + rand_lphi1) 
        particle_poses[m, 2] = wraptopi(phi + rand_lphi1 + rand_lphi2) 

    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    M = particle_poses.shape[0]

    particle_weights = np.zeros(M)
    range_error = np.zeros(M)
    bearing_error = np.zeros(M)
    range_weight = np.zeros(M)
    bearing_weight = np.zeros(M)
    

    for m in range(M):
        robot_range = sqrt((beacon_pose[0])**2 + (beacon_pose[1])**2)
        robot_bearing = arctan2(beacon_pose[1], beacon_pose[0])

        particle_range = sqrt((beacon_loc[0] - particle_poses[m,0])**2 + (beacon_loc[1] - particle_poses[m,1])**2)
        particle_bearing = angle_difference(particle_poses[m,2], arctan2(beacon_loc[1] - particle_poses[m,1], beacon_loc[0] - particle_poses[m,0]))

        range_error[m] = robot_range - particle_range
        bearing_error[m] = angle_difference(robot_bearing, particle_bearing)

        range_weight[m] = gauss(range_error[m], 0, 0.08)
        bearing_weight[m] = gauss(bearing_error[m], 0, 0.05)
        
        particle_weights[m] = range_weight[m] * bearing_weight[m]

    return particle_weights
