import serial
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../../Aero_Funcs')
from Controls_Funcs import *
import time

def crux(A):
    return np.array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1], A[0], 0]])

def small_rot(n):
    return np.identity(3) + crux(n)

class plotter():

    def __init__(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.lines = []

        self.lines.append(self.ax.plot([0,1], [0,0], [0,0])[0])
        self.lines.append(self.ax.plot([0,0], [0,1], [0,0])[0])
        self.lines.append(self.ax.plot([0,0], [0,0], [0,1])[0])

    def ploot(self, A):
        for ii, line in enumerate(self.lines):

            pt = np.zeros(3)
            pt[ii] = 1
            pt = A@pt

            self.lines[ii].set_xdata([0,pt[0]])
            self.lines[ii].set_ydata([0,pt[1]])
            self.lines[ii].set_3d_properties([0,pt[2]])
        plt.draw()

def plotlive(R):
    ax.cla()
    ax.axis('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    for ii in range(3):

        pt = np.zeros(3)
        pt[ii] = 1
        pt = R@pt

        ax.plot([0,pt[0]], [0,pt[1]], [0,pt[2]])

    plt.pause(.025)    
    plt.draw()

def get_data(serial_com):
    data = serial_com.readline()
    data_str = data[:-2].decode("utf-8")
    measurements = np.asarray([float(x) for x in data_str.split()])
    mag = measurements[0:3]
    rate = measurements[3:6]
    accel = measurements[6:9]
    return mag, rate, accel
    

def get_Q(P_, Qa, Qb, Qd, R_mag, R_rate, R_accel, ca, cd, dt):
    Q = zeros((12,12))
    Q[0:3,0:3] = P_[0:3,0:3] + dt**2*(P_[3:6,3:6] + Qb + R_rate)# Eq 10.1.17
    Q[0:3,3:6] = -dt*(P_[0:3,3:6] + Qb)
    Q[3:6,3:6] = P_[3:6,3:6] + Qb
    Q[6:9,6:9] = ca**2*P_[6:9,6:9] + Qa
    Q[9:12,9:12] = cd**2*P_[9:12,9:12] + Qd

    return Q

def get_H(g_gyro, m_gyro, dt):
    
    H = np.vstack([np.hstack([ -crux(g_gyro), dt*crux(g_gyro), np.identity(3) ,  np.zeros((3,3)) ]),
                   np.hstack([ -crux(m_gyro), dt*crux(m_gyro), np.zeros((3,3)), -np.identity(3)  ])])
    return H


def quat_update(qr_i, qi_i, n):

    qi_dw, qr_dw = axis_angle2quat(n)
    qr, qi = quat_mult(qr_dw, qi_dw, qr_i, qi_i)

    return qr, qi

def axis_angle2quat(n):

    angle = np.linalg.norm(n)
    if angle < 1e-12:
        return np.array([0,0,0]), 1

    axis = n/norm(n)
    

    imag = axis*sin(angle/2)
    real = cos(angle/2)

    return imag, real

if __name__ == "__main__":
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    dt = 0.025
    # arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)

    still_mag = np.load('still_mag_data_025dt.npy')
    still_rate = np.load('still_rate_data_025dt.npy')
    still_accel = np.load('still_accel_data_025dt.npy')

    moving_mag = np.load('moving_mag_data_025dt.npy')
    moving_rate = np.load('moving_rate_data_025dt.npy')
    moving_accel = np.load('moving_accel_data_025dt.npy')

    mag_abs = np.mean(still_mag, axis = 0)
    g_abs = np.array([0,0,9.8])

    R_mag = np.cov(still_mag.T)
    R_rate = np.cov((still_rate/180*np.pi).T)
    R_accel = np.cov(still_accel.T)

    B_mag_nominal = np.linalg.norm(np.mean(still_mag, axis=0))

    
    ca = 0.1
    cd = 0.8
    Qa = np.identity(3)*1e-8
    Qb = np.identity(3)*1e-6
    Qd = np.identity(3)*1e-8

    Ra = R_accel + Qa + dt**2*(Qb + R_rate)
    Rm = R_mag + Qd + dt**2*B_mag_nominal**2*(Qb + R_rate)

    R = np.vstack([np.hstack([Ra             , np.zeros((3,3))]),
                   np.hstack([np.zeros((3,3)), Rm           ])])

    P = np.identity(12)
    

    qr = 1
    qi = np.array([0,0,0])
    w_bias = np.array([0,0,0])
    m_bias = np.array([0,0,0])
    lin_accel = np.array([0,0,0])


    iters = 600
    for mag, rate, accel in zip(moving_mag[:iters], moving_rate[:iters], moving_accel[:iters]):
    # while True:
        # mag, rate, accel = get_data(arduino)




        rate = rate/180*np.pi

        qr, qi = quat_update(qr, qi, rate*dt)

        DCM = quat2dcm(qr, qi)


        Q = get_Q(P, Qa, Qb, Qd, R_mag, R_rate, R_accel, ca, cd, dt)
        P = Q.copy()

        g_gyro = DCM@g_abs
        m_gyro = DCM@mag_abs

        H = get_H(g_gyro, m_gyro, dt)

        K = P@H.T@np.linalg.inv(H@P@H.T + R)

        g_accel = accel - ca*lin_accel

        z = np.hstack([g_accel - g_gyro,
                      (mag-m_bias)     - m_gyro])

        X = np.zeros(12)
        X = K@z

        print(X)

        P = (np.identity(12) - K@H)@P

        qr, qi = quat_update(qr, qi, -X[0:3]*dt)
        
        w_bias = w_bias - X[3:6]
        lin_accel = lin_accel -X[6:9]
        m_bias = m_bias - X[9:12]

        DCM = quat2dcm(qr, qi)

        plotlive(DCM)


        # print(['{:+.3f}'.format(ii) for ii in measurements])