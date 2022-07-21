#!/usr/bin/env python3
"""Calculation of motion models."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

# Load data
filelocation = 'Part A/training1.csv'
data = np.loadtxt(filelocation, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

dt = np.empty(0)
dr = np.empty(0)
for i in range(len(raw_ir1)-1):
    dt = np.append(dt,time[i+1]-time[i])
    dr = np.append(dr, range_[i+1]-range_[i])


plt.figure()
plt.plot(time[1:len(time)],dr/dt)
plt.plot(time,velocity_command)

plt.show()
