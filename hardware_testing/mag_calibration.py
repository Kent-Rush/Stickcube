import serial
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from live_plotter import plotter
from scipy.optimize import minimize

def crux(A):
    return np.array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1], A[0], 0]])


def get_data(serial_com):
    data = serial_com.readline()
    data_str = data[:-2].decode("utf-8")
    measurements = np.asarray([float(x) for x in data_str.split()])
    mag = measurements[0:3]
    rate = measurements[3:6]
    accel = measurements[6:9]
    return mag, rate, accel

def collect_data():
    arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)

    input('ready to read')

    mag_data = []
    for ii in range(1000):
        mag, rate, accel = get_data(arduino)
        mag_data.append(mag)
        if ii%100 == 0:
            print(ii)

    mag_data = np.vstack(mag_data)
    np.save('magdata2', mag_data)

def minim(x, data):

    shifted = data - x
    lengths = np.linalg.norm(shifted, axis = 1)
    return np.var(lengths)

if __name__ == "__main__":

    arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)

    

    for ii in range(100):
        mag, rate, accel = get_data(arduino)

    print('reading still data')
    mag_data = []
    rate_data = []
    accel_data = []
    for ii in range(300):
        mag, rate, accel = get_data(arduino)
        mag_data.append(mag)
        rate_data.append(rate)
        accel_data.append(accel)

    for m in mag_data:
        print(m)

    mag_data = np.vstack(mag_data)
    rate_data = np.vstack(rate_data)
    accel_data = np.vstack(accel_data)

    np.save('still_mag_data_025dt', mag_data)
    np.save('still_rate_data_025dt', rate_data)
    np.save('still_accel_data_025dt', accel_data)

    input('ready to read moving data')

    mag_data = []
    rate_data = []
    accel_data = []
    for ii in range(1000):
        mag, rate, accel = get_data(arduino)
        mag_data.append(mag)
        rate_data.append(rate)
        accel_data.append(accel)

    mag_data = np.vstack(mag_data)
    rate_data = np.vstack(rate_data)
    accel_data = np.vstack(accel_data)

    np.save('moving_mag_data_025dt', mag_data)
    np.save('moving_rate_data_025dt', rate_data)
    np.save('moving_accel_data_025dt', accel_data)


    # mag_data = np.load('magdata2.npy')

    # fig = plt.figure()
    # ax  = fig.add_subplot(111,projection = '3d')
    # ax.scatter(mag_data[:,0],mag_data[:,1],mag_data[:,2])

    # mean_len = np.mean(np.linalg.norm(mag_data, axis = 1))

    # clean_data = []
    # for m in mag_data:
    #     if abs(np.linalg.norm(m)-mean_len) < mean_len*.4:
    #         clean_data.append(m)

    # clean_data = np.vstack(clean_data)

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection = '3d')
    # ax.scatter(clean_data[:,0],clean_data[:,1],clean_data[:,2])

    # x0 = np.array([0,0,0])
    # offset = minimize(minim, x0, args=clean_data)

    # print(offset)
    # print(minim(x0, clean_data))

    # plt.show()




