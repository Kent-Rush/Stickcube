import serial
import time
import numpy as np

# arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)
# for ii in range(100):
# 	data = arduino.readline()

# input('letsgoo')

# n = 300
# datalog = np.zeros((n,3))
# for ii in range(n):
# 	data = arduino.readline()
# 	data_str = data[:-2].decode("utf-8")
# 	vector = np.asarray([float(x) for x in data_str.split()])
# 	datalog[ii,:] = vector[3:6]

# np.save('data.npy',datalog)

datalog = np.load('data.npy')
rate = data[90:250,2]