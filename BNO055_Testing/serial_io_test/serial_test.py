import serial
import time
import numpy as np

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)
while True:
	data = arduino.readline()
	data_str = data[:-2].decode("utf-8")
	vector = np.asarray([float(x) for x in data_str.split()])
	print(vector)