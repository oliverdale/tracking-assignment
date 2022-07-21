#!/usr/bin/env python3
"""Calculation of sensor models."""

from re import M
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy import stats as stats

# Load data
filelocation = 'Part A/calibration.csv'
data = np.loadtxt(filelocation, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

def linear_model (x, a, b):
    return a * x + b

def inv_linear_model (z, a, b):
    return (z-b)/a

def inv_hyperbolic_model (z, a, b):
    return a/(z-b)    

def hyperbolic_model (x, a, b):
    return a/x+b

class SensorFit:
    def __init__ (self, threshold, x_data, y_data, time_data, type="Sonar"):
        self.threshold = threshold
        self.x_data = x_data
        self.y_data = y_data
        self.raw_data = y_data
        self.range = x_data
        self.sensormodel = []
        self.type = type
        self.a = 0
        self.b = 0
        self.time = time_data
        self.mu = -1
        self.sigma = -1
        self.variance_model = np.empty(0)
        self.step = -1
    
    def update_model (self):
        model, _ = opt.curve_fit(linear_model, self.x_data, self.y_data)
        self.a, self.b = model
        self.sensormodel = linear_model(self.x_data, self.a, self.b)    

    def update_data (self, scale = 1):
        y_data_update = np.empty(0)
        x_data_update = np.empty(0)
        time_update = np.empty(0)

        for i in range(0, len(self.y_data)-1):
            if abs(self.y_data[i] - self.sensormodel[i]) < self.threshold/scale:
                y_data_update  = np.append(y_data_update, self.y_data[i])
                x_data_update = np.append(x_data_update, self.x_data[i])
                # time_update = np.append(time_update, self.time[i])
        self.y_data = y_data_update
        self.x_data = x_data_update
        # self.time = time_update

    def find_model (self, tol):
        N = 100
        i = 1
        error = 100
        self.update_model()
        while (error > tol):
            self.update_data(i)
            self.update_model()
            error = max(abs(self.y_data - self.sensormodel)) 
            i+=1
            if (i >= N):
                break

        self.find_var()

    def find_var(self):
        window_size = 50
        offset = int(window_size / 2)

        error = linear_model(self.range, self.a, self.b)-self.raw_data
        variance_array = np.empty(0)
        variance_range = np.empty(0)
        for i in range(len(error)-window_size):
            error_block = error[i:i + window_size]
            variance = np.std(error_block)
            variance_array = np.append(variance_array, variance)
            if (self.type=="IR"):
                variance_range = np.append(variance_range, 1/self.range[i+offset])
            else:
                variance_range = np.append(variance_range, self.range[i+offset])

        self.variance_model = np.polyfit(variance_range, variance_array, 4)
        z = self.variance_model
        print(z)
        plt.figure()
        plt.scatter(variance_range, variance_array, s=1, color='r')
        if (self.type == "IR"):
            plt.plot(1/self.range, (1/self.range)**4.0*z[0]+(1/self.range)**3.0*z[1]+(1/self.range)**2.0*z[2]+(1/self.range)*z[3]+z[4])
        else:
            plt.plot(self.range, (self.range)**4.0*z[0]+(self.range)**3.0*z[1]+(self.range)**2.0*z[2]+(self.range)*z[3]+z[4])
        plt.show()

    def plot_model (self):
        plt.figure(figsize=(12, 5))
        # plt.subplot(121)
        if (self.type == "IR"):
            plt.plot(1/self.x_data, self.sensormodel, '--', color='red')
            plt.scatter(1/self.range, self.raw_data, color='b',s=2)
            plt.scatter(1/self.x_data, self.y_data, color='g',s=2)      
        else:
            plt.plot(self.x_data, self.sensormodel, '--', color='red')
            plt.scatter(self.range, self.raw_data, color='b',s=2)
            plt.scatter(self.x_data, self.y_data, color='g',s=2)   
        plt.xlabel("Range x")    
        plt.ylabel("Measurement z")
        plt.show()

        # error = self.sensormodel-self.y_data

        # sorted_error = self.sensormodel-self.y_data
        # sorted_error.sort()

        # plt.subplot(122)
        # plt.hist(sorted_error, 40, density=True)
        # plt.plot(sorted_error, stats.norm.pdf(sorted_error, self.mu, self.sigma))
        # plt.show()

        # plt.figure()
        # plt.scatter(self.x_data,error,s=2)
        # plt.xlabel("Range x")
        # plt.ylabel("Measurement error v")
        # plt.show()        

#---------------------------------------------------------------------------------------------
# Upper and Lower Sensor Bounding
#---------------------------------------------------------------------------------------------
Lsonar1 = 0.02
Usonar1 = 4
Lsonar2 = 0.3
Usonar2 =5
Lir1 = 0.15
Uir1 = 1.5
Lir2 = 4/100
Uir2 = 30/100
Lir3 = 10/100
Uir3 = 80/100
Lir4 = 1
Uir4 = 5

sonar1_bound = np.empty(0)
range_s1 = np.empty(0)
time_s1 = np.empty(0)
sonar2_bound = np.empty(0)
range_s2 = np.empty(0)
time_s2 = np.empty(0)
ir1_bound = np.empty(0)
range_ir1 = np.empty(0)
time_ir1 = np.empty(0)
ir2_bound = np.empty(0)
range_ir2 = np.empty(0)
time_ir2 = np.empty(0)
ir3_bound = np.empty(0)
range_ir3 = np.empty(0)
time_ir3 = np.empty(0)
ir4_bound = np.empty(0)
range_ir4 = np.empty(0)
time_ir4 = np.empty(0)

for i in range(len(range_)):
    if range_[i] > Lsonar1 and range_[i] < Usonar1:
        sonar1_bound = np.append(sonar1_bound,sonar1[i])
        range_s1 = np.append(range_s1,range_[i])
        time_s1 = np.append(time_s1, time[i])
    if range_[i] > Lsonar2 and range_[i] < Usonar2:
        sonar2_bound = np.append(sonar2_bound,sonar2[i])
        range_s2 = np.append(range_s2,range_[i])
        time_s2 = np.append(time_s2, time[i])   
    if range_[i] > Lir1 and range_[i] < Uir1:
        ir1_bound = np.append(ir1_bound,raw_ir1[i])
        range_ir1 = np.append(range_ir1,range_[i])   
        time_ir1 = np.append(time_ir1, time[i])           
    if range_[i] > Lir2 and range_[i] < Uir2:
        ir2_bound = np.append(ir2_bound,raw_ir2[i])
        range_ir2 = np.append(range_ir2,range_[i]) 
        time_ir2 = np.append(time_ir2, time[i])     
    if range_[i] > Lir3 and range_[i] < Uir3:
        ir3_bound = np.append(ir3_bound,raw_ir3[i])
        range_ir3 = np.append(range_ir3,range_[i])  
        time_ir3 = np.append(time_ir3, time[i])                         
    if range_[i] > Lir4 and range_[i] < Uir4:
        ir4_bound = np.append(ir4_bound,raw_ir4[i])
        range_ir4 = np.append(range_ir4,range_[i])
        time_ir4 = np.append(time_ir4, time[i])          

#---------------------------------------------------------------------------------------------
# Fitting and Plotting Sensor Models
#---------------------------------------------------------------------------------------------

# Sonar 1
# sonar_model_1 = SensorFit(5, range_s1, sonar1_bound, time_s1, "Sonar")
# sonar_model_1.find_model(0.1)
# sonar_model_1.plot_model()

# # Sonar 2
# sonar_model_2 = SensorFit(5, range_s2, sonar2_bound, time_s2, "Sonar")
# sonar_model_2.find_model(0.1)
# sonar_model_2.plot_model()
# print(sonar_model_2.a)
# print(sonar_model_2.b)

# # IR 1
# ir_model_1 = SensorFit(2, 1/range_ir1, ir1_bound, time_ir1, "IR")
# ir_model_1.find_model(0.07)
# ir_model_1.plot_model()

# IR 2
ir_model_2 = SensorFit(0.6, 1/range_ir2, ir2_bound, time_ir2, "IR")
ir_model_2.find_model(0.25)
ir_model_2.plot_model()
print(ir_model_2.a)
print(ir_model_2.b)

# # IR 3
# ir_model_3 = SensorFit(1, 1/range_ir3, ir3_bound, time_ir3, "IR")
# ir_model_3.find_model(0.2)
# ir_model_3.plot_model()
# print(ir_model_3.a)
# print(ir_model_3.b)

# # IR 4
# ir_model_4 = SensorFit(2, 1/range_ir4, ir4_bound, time_ir4, "IR")
# ir_model_4.find_model(0.07)
# ir_model_4.plot_model()
# print(ir_model_4.a)
# print(ir_model_4.b)

#---------------------------------------------------------------------------------------------
# Speed Estimation
#---------------------------------------------------------------------------------------------

# Load data
filelocationt1 = 'Part A/training1.csv'
data = np.loadtxt(filelocationt1, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

errors = np.empty(0)

estimated_speed = np.gradient(range_)/np.gradient(time)

def motion_model(initial, velocity_command, time):
    modelled_displacement = np.empty(0)
    modelled_speed = np.empty(0)
    process_noise_array = np.empty(0)
    flag = 0

    initial_displacement = initial
    cur_displacement = initial_displacement
    current_velocity = velocity_command[0]

    command_accel = np.gradient(velocity_command)/np.gradient(time)
    tol = 0.005

    for i in range(len(time)-1):
        prev_velocity = current_velocity

        # if (command_accel[i] == 0 and not flag):
        #     current_velocity = velocity_command[i]
        # elif (command_accel[i] > 0 or flag == 1):
        #     flag = 1
        #     current_velocity = prev_velocity + 0.00286 * (time[i+1]-time[i])
        # elif (command_accel[i] < 0 or flag == 2):
        #     flag = 2
        #     current_velocity = prev_velocity - 0.00286 * (time[i+1]-time[i])
        # if ( current_velocity >= (velocity_command[i]-tol) and current_velocity <= (velocity_command[i]+tol) ):
        #     flag = 0

        current_velocity = prev_velocity + (velocity_command[i] - prev_velocity)/25
        # if (velocity_command[i] == 0):
        #     current_velocity = velocity_command[i]
        prev_displacement = cur_displacement
        cur_displacement = cur_displacement + current_velocity*(time[i+1]-time[i])
        process_noise = cur_displacement - prev_displacement - current_velocity*(time[i+1]-time[i])
        process_noise_array = np.append(process_noise_array, process_noise)
 
        modelled_displacement = np.append(modelled_displacement,cur_displacement)
        modelled_speed = np.append(modelled_speed,current_velocity)

    estimated_accel = np.gradient(estimated_speed)/np.gradient(time)
    plt.figure()
    plt.plot(time[0:len(time)-1],modelled_displacement)
    plt.plot(time, range_)
    plt.show()

    error = modelled_displacement-range_[0:len(range_)-1]
    mu = sum(error)/len(error)
    sigma = np.std(error)
    # print(mu)
    # print(sigma)

    # plt.figure()
    # plt.plot(time[0:len(time)-1],process_noise_array)
    # plt.show()

    plt.figure()
    plt.plot(time,estimated_speed)
    plt.plot(time[0:len(time)-1],modelled_speed)
    plt.plot(time,velocity_command)
    plt.legend(["estimated","modelled","commanded"])
    # plt.plot(time,command_accel)
    # plt.plot(time,estimated_accel)

    # # plt.scatter(time, )
    # # plt.plot(time,range_)
    # # plt.plot(range_, sonar1)
    plt.show()

    return sigma**2.0

var_motion_model = motion_model(range_[0],velocity_command,time)

# plt.subplot(121)
# plt.scatter(sonar_model_1.time,sonar_model_1.x_data)



# plt.figure()
# plt.plot(errors)
# plt.show()

#---------------------------------------------------------------------------------------------
# Sensor fusion
#---------------------------------------------------------------------------------------------
# filelocation = 'Part A/test.csv'
# data = np.loadtxt(filelocation, delimiter=',', skiprows=1)
# index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# # modelled_speed = motion_model(0,velocity_command,time)

# # # plt.figure()
# # # plt.plot(time[0:len(time)-1],modelled_speed)
# # # plt.plot(time,velocity_command)

# # # plt.show()

# x_hat_s1 = inv_linear_model(sonar1, sonar_model_1.a, sonar_model_1.b)

# var_x_hat_s1 = (sonar_model_1.sigma)**2.0/(sonar_model_1.a)**2.0
# x_hat_s2 = inv_linear_model(sonar2, sonar_model_2.a, sonar_model_2.b)
# x_hat_ir1 = inv_hyperbolic_model(raw_ir1, ir_model_1.a, ir_model_1.b)
# plt.figure()
# # plt.plot(time,x_hat_s1)
# plt.plot(time,x_hat_s2)
# plt.plot(time,x_hat_ir1)
# plt.show()

