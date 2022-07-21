# Kalman filter demonstration program to estimate
# position of a robot moving at constant speed.
# M.P. Hayes UCECE 2015
from re import A
from numpy.random import randn
from numpy import inf, zeros
import numpy as np
from matplotlib import pyplot as plt

filelocation = 'Part A/training1.csv'
data = np.loadtxt(filelocation, delimiter=',', skiprows=1)
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

#---------------------------------------------------------------------------------------------
# Motion Model
#---------------------------------------------------------------------------------------------
def motion_model(cur_displacement, prev_velocity, velocity_command, dt):
    current_velocity = prev_velocity + (velocity_command - prev_velocity)/25
    cur_displacement = cur_displacement + current_velocity*dt
    return [current_velocity, cur_displacement]

# Sensor ID's
SONAR1 = 0
SONAR2 = 1
IR1 = 2
IR2 = 3
IR3 = 4
IR4 = 5

HIGH_VARIANCE = 10000

#---------------------------------------------------------------------------------------------
# Upper and Lower Sensor Bounding
#---------------------------------------------------------------------------------------------
Lsonar1 = 0.02
Usonar1 = 4
Lsonar2 = 0.3
Usonar2 = 5
Lir1 = 0.15
Uir1 = 1.5
Lir2 = 4/100
Uir2 = 30/100
Lir3 = 10/100
Uir3 = 80/100
Lir4 = 1
Uir4 = 5

#---------------------------------------------------------------------------------------------
# Sensor Models
#---------------------------------------------------------------------------------------------
s1_a = 0.9947413628641788
s1_b = -0.01719304171556235
# s1_sigma = 0.9137995040288974

s2_a = 1.008419936122565
s2_b = -0.0164594854388818

ir1_a = 0.15653414365094925
ir1_b = -0.047378733831767064
# ir1_sigma = 0.16476446341561057

ir2_a = 0.16712226631872987
ir2_b = -0.07993089550276249
# ir2_sigma = 0.1635108904932071 

ir3_a = 0.2857914375056021
ir3_b = 0.11301135872787066

ir4_a = 1.4880409767554952
ir4_b = 1.2571846661912054

# Initial position
x = 0
# Speed
v = velocity_command[0]

# Number of steps and time-step interval
Nsteps = len(time)

# Process and sensor noise standard deviations
std_W = 0.14898582180965
# std_V_s1 = s1_sigma/s1_a #0.020808755342598604 #0.005372362188558495
# std_V_ir1 = ir1_sigma/ir1_a
# std_V_ir2 = ir2_sigma/ir2_a

# Process and measurement noise variances
var_W = std_W ** 2

# Start with a poor initial estimate of robot’s position
mean_X_posterior = 0
var_X_posterior = 1
estimated_position = np.empty(0)
variance = np.empty(0)

w_s1_array = np.empty(0)
w_ir1_array = np.empty(0)
w_ir2_array = np.empty(0)

ir1_xdata = np.arange(0.15,1.5,0.01)
ir2_range_data = (ir2_a/(raw_ir2-ir2_b))

def sr1_std_model(x):  
    return x**4.0*(-0.08697996)+x**3.0*(0.66150193)+x**2.0*(-1.33720511)+x*(0.98917387)+(-0.14152459)

def sr2_std_model(x):  
    return x**4.0*(0.01031636)+x**3.0*(-0.06710155)+x**2.0*(0.14751923)+x*(-0.12477506)+(0.03569059)    

def ir1_std_model(x):
    return x**4.0*(-0.21043084)+x**3.0*(0.54897043)+x**2.0*(-0.28120274)+x*(-0.16678677)+(0.24750224)   

def ir2_std_model(x):
    return x**4.0*(78.05468942)+x**3.0*(52.63646784)+x**2.0*(-54.65965193)+x*(12.66762483)+(-0.69525184)      

def ir3_std_model(x):
    return x**4.0*(5.33646456)+x**3.0*(-9.74500716)+x**2.0*(6.18398335)+x*(-1.59358615)+(0.20127576)

def ir4_std_model(x):
    return x**4.0*(0.01095191)+x**3.0*(-0.10947332)+x**2.0*(0.39962388)+x*(-0.60240662)+(0.42726962)    

def within_range(range, id):
    passed = 0

    if id == 0:
        if ((range > Lsonar1) and (range < Usonar1)):
            passed = 1
        # else:
            # print("Sonar 1 Failed")

    if id == 1:
        if ((range > Lsonar2) and (range < Usonar2)):
            passed = 1            

    elif id == 2:
        if ((range > Lir1) and (range < Uir1)):
            passed = 1
        # else:
            # print("IR 1 Failed")

    elif id == 3:
        if ((range > Lir2) and (range <  Uir2)):
            passed = 1
        # else:
            # print("IR 2 Failed")

    elif id == 4:
       if ((range > Lir3) and (range <  Uir3)):
            passed = 1   

    elif id == 5:
       if ((range > Lir4) and (range <  Uir4)):
            passed = 1             

    return passed

def check_outlier(prior, sensor_reading, var, scale):
    passed = 0
    if ( (abs(sensor_reading-prior)) < (scale*(np.sqrt(var))) ):
        passed = 1
    # else:
        # print("outlier")
    return passed

def calculate_sr1(sensor_reading,mean_X_prior):
    # ML estimate of position from measurement (using sensor model)
    x_infer_s1 = (sensor_reading-s1_b)/s1_a

    # TO DO: Add code to determine variance based off x_infer_s1.
    var_V_s1 = (sr1_std_model(x_infer_s1)/s1_a) ** 2
    
    if (within_range(x_infer_s1, SONAR1) and check_outlier(mean_X_prior, x_infer_s1, var_V_s1, 1)):
        return [x_infer_s1, var_V_s1]
    else:
        var_V_s1 = HIGH_VARIANCE
        # print("high")
        return [x_infer_s1, var_V_s1]

def calculate_sr2(sensor_reading,mean_X_prior):
    # ML estimate of position from measurement (using sensor model)
    x_infer_s2 = (sensor_reading-s2_b)/s2_a

    # TO DO: Add code to determine variance based off x_infer_s1.
    var_V_s2 = (sr2_std_model(x_infer_s2)/s2_a) ** 2
    
    if (within_range(x_infer_s2, SONAR2) and check_outlier(mean_X_prior, x_infer_s2, var_V_s2, 4.289)):
        return [x_infer_s2, var_V_s2]
    else:
        var_V_s2 = HIGH_VARIANCE
        # print("high")
        return [x_infer_s2, var_V_s2]        

def calculate_ir1(sensor_reading,mean_X_prior):
    # For IR sensor
    #x_infer_ir2 = (raw_ir2[n]-(ir2_a/mean_X_prior+ir2_b))/(-ir2_a/mean_X_prior**2.0) + mean_X_prior
    x_infer_ir1 = (ir1_a/(sensor_reading-ir1_b))
    ir1_sigma = ir1_std_model(x_infer_ir1)
    var_V_ir1 = ir1_sigma**2.0/(-ir1_a/mean_X_prior**2.0)**2.0  

    if ( within_range(x_infer_ir1, IR1) and check_outlier(mean_X_prior,x_infer_ir1,var_V_ir1,1) ):
        return [x_infer_ir1, var_V_ir1]
    else:
        var_V_ir1 = HIGH_VARIANCE
        return [x_infer_ir1, var_V_ir1]

def calculate_ir2(sensor_reading,mean_X_prior):
    # For IR sensor
    #x_infer_ir1 = (raw_ir1[n]-(ir1_a/mean_X_prior+ir1_b))/(-ir1_a/mean_X_prior**2.0)+mean_X_prior
    x_infer_ir2 = (ir2_a/(sensor_reading-ir2_b))
    ir2_sigma = ir2_std_model(x_infer_ir2)    
    var_V_ir2 = ir2_sigma**2.0/(-ir2_a/mean_X_prior**2.0)**2.0

    if ( within_range(x_infer_ir2, IR2) and check_outlier(mean_X_prior,x_infer_ir2,var_V_ir2,0.5) ):
        return [x_infer_ir2, var_V_ir2]
    else:
        var_V_ir2 = HIGH_VARIANCE
        return [x_infer_ir2, var_V_ir2]

def calculate_ir3(sensor_reading,mean_X_prior):
    # For IR sensor
    #x_infer_ir1 = (raw_ir1[n]-(ir1_a/mean_X_prior+ir1_b))/(-ir1_a/mean_X_prior**2.0)+mean_X_prior
    x_infer_ir3 = (ir3_a/(sensor_reading-ir3_b))
    ir3_sigma = ir3_std_model(x_infer_ir3)
    var_V_ir3 = ir3_sigma**2.0/(-ir3_a/mean_X_prior**2.0)**2.0

    if ( within_range(x_infer_ir3, IR3) and check_outlier(mean_X_prior,x_infer_ir3,var_V_ir3,0.5) ):
        return [x_infer_ir3, var_V_ir3]
    else:
        var_V_ir3 = HIGH_VARIANCE
        return [x_infer_ir3, var_V_ir3]        

def calculate_ir4(sensor_reading,mean_X_prior):
    # For IR sensor
    #x_infer_ir1 = (raw_ir1[n]-(ir1_a/mean_X_prior+ir1_b))/(-ir1_a/mean_X_prior**2.0)+mean_X_prior
    x_infer_ir4 = (ir4_a/(sensor_reading-ir4_b))
    ir4_sigma = ir4_std_model(x_infer_ir4)
    var_V_ir4 = ir4_sigma**2.0/(-ir4_a/mean_X_prior**2.0)**2.0

    if ( within_range(x_infer_ir4, IR4) and check_outlier(mean_X_prior,x_infer_ir4,var_V_ir4,0.5) ):
        return [x_infer_ir4, var_V_ir4]
    else:
        var_V_ir4 = HIGH_VARIANCE
        return [x_infer_ir4, var_V_ir4]        

w_1_array = np.empty(0)
w_2_array = np.empty(0)
w_3_array = np.empty(0)
K_array = np.empty(0)
current_velocity = velocity_command[0]
# Kalman filter
for n in range(1, Nsteps):
    count = 0

    x_blu = np.empty(0)
    var_X_blu = np.empty(0)

    x_infer = np.empty(0)
    var_V = np.empty(0)

    # Calculate mean and variance of prior estimate for position
    # (using motion model)
    [current_velocity, mean_X_prior] = motion_model(mean_X_posterior, current_velocity, velocity_command[n], (time[n]-time[n-1]))
    var_X_prior = var_X_posterior + var_W

    [x_infer_1, var_V_1] = calculate_ir1(raw_ir1[n],mean_X_prior)
    [x_infer_2, var_V_2] = calculate_sr1(sonar1[n],mean_X_prior)
    [x_infer_3, var_V_3] = calculate_ir4(raw_ir4[n],mean_X_prior)

    # var_V_3 = 10000
    # var_V_2 = 10000
    # var_V_1 = 10000

    w_1 = (1/var_V_1)/((1/var_V_1)+(1/var_V_2)+(1/var_V_3))
    w_2 = (1/var_V_2)/((1/var_V_1)+(1/var_V_2)+(1/var_V_3))
    w_3 = (1/var_V_3)/((1/var_V_1)+(1/var_V_2)+(1/var_V_3))

    w_1_array = np.append(w_1_array, w_1)
    w_2_array = np.append(w_2_array, w_2)
    w_3_array = np.append(w_3_array, w_3)

    x_blu = w_1*x_infer_1 + w_2*x_infer_2 + w_3*x_infer_3
    var_X_blu = 1/(1/var_V_1 + 1/var_V_2 + 1/var_V_3)

    K = var_X_prior / (var_X_blu + var_X_prior)
    # if (var_V_1 == HIGH_VARIANCE and var_V_2 == HIGH_VARIANCE and var_V_3 == HIGH_VARIANCE):
    #     K = 0
    # Calculate mean and variance of posterior estimate for position
    mean_X_posterior = mean_X_prior + K * (x_blu - mean_X_prior)
    var_X_posterior = (1 - K) * var_X_prior

    K_array = np.append(K_array,K)

    estimated_position = np.append(estimated_position, mean_X_posterior)
    variance = np.append(variance,var_X_posterior)

plt.figure()
plt.plot(time[1:], estimated_position, color='r')
plt.plot(time, range_)
plt.show()

plt.figure()
plt.plot(time[1:], variance)
plt.show()

plt.figure()
plt.scatter(estimated_position,w_1_array,s=1)
plt.scatter(estimated_position,w_2_array,s=1)
plt.scatter(estimated_position, w_3_array,s=1)
plt.legend(["w1", "w2", "w3"])
plt.show()

# plt.figure()
# plt.scatter(np.arange(len(w_s1_array)),w_s1_array)
# plt.scatter(np.arange(len(w_ir1_array)),w_ir1_array)
# plt.scatter(np.arange(len(w_ir2_array)),w_ir2_array)
# plt.show()

# plt.figure()
# plt.scatter(time, ir2_range_data, s=1)
# plt.show()

# plt.figure()
# plt.plot(K_array)
# plt.show()

# print(w_s1)
# print(w_ir1)
# print(w_ir2)

# rangeee = np.arange(0.1,0.8,0.01)
# plt.figure()
# plt.plot(rangeee, ir3_a/rangeee+ir3_b)
# plt.show()
