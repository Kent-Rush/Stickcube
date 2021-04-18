import serial
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def get_att_pts(R):
    pts = []
    for ii in range(3):

        pt = np.zeros(3)
        pt[ii] = 1
        p  = R@pt

        pts.append([ [0,p[0]], [0,p[1]], [0,p[2]] ])

    return pts


def plotlive(Rs):
    ax.cla()
    ax.axis('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    c = ['r','g','b','k']
    for ii, R in enumerate(Rs):
        pts = get_att_pts(R)
        for pt in pts:
            ax.plot(*pt, c[ii])

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
    Q = np.zeros((12,12))
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

    axis = n/np.linalg.norm(n)
    

    imag = axis*np.sin(angle/2)
    real = np.cos(angle/2)

    return imag, real

def quat_mult(real1, imag1, real2, imag2):
    """
    Multiplies two quaternions
    q1 (x) q2
    """

    imag3 = real1*imag2 + real2*imag1 + crux(imag1)@imag2
    real3 = real2*real1 - np.dot(imag1, imag2)

    return real3, imag3

def TRIAD(a_b, b_b, a_i, b_i):


    x_b = a_b/np.linalg.norm(a_b)
    y_b = np.cross(a_b, b_b)/np.linalg.norm(np.cross(a_b, b_b))
    z_b = np.cross(x_b, y_b)

    x_i = a_i/np.linalg.norm(a_i)
    y_i = np.cross(a_i, b_i)/np.linalg.norm(np.cross(a_i, b_i))
    z_i = np.cross(x_i, y_i)

    b2t = np.vstack([x_b, y_b, z_b])
    i2t = np.vstack([x_i, y_i, z_i])

    i2b = b2t.T@i2t

    q_real, q_imag = dcm2quat(i2b.T)

    return q_real, q_imag

def quat2dcm(n, E):
    """
    generates the active rotation matrix from a quaternion.
    :param n: scalar part
    :param E: vector part

    """
    
    frame_rotation = (2*n**2 - 1)*np.identity(3) + 2*np.outer(E,E) - 2*n*crux(E) 

    return frame_rotation.T

def dcm2quat(C):

    """
    From "A Survey on the Computation of Quaternions from Rotation Matrices" by Soheil and Federico
    """

    qi_sign = C[2,1] - C[1,2]
    qj_sign = C[0,2] - C[2,0]
    qk_sign = C[1,0] - C[0,1]

    real = .25*np.sqrt((C[0,0] + C[1,1] + C[2,2] + 1)**2 +
                    (C[2,1] - C[1,2])**2 +
                    (C[0,2] - C[2,0])**2 +
                    (C[1,0] - C[0,1])**2)

    qi = .25*np.sqrt((C[2,1] - C[1,2])**2 +
                  (C[0,0] - C[1,1] - C[2,2] + 1)**2 +
                  (C[1,0] + C[0,1])**2 +
                  (C[2,0] + C[0,2])**2)

    qj = .25*np.sqrt((C[0,2] - C[2,0])**2 +
                  (C[1,0] + C[0,1])**2 +
                  (C[1,1] - C[0,0] - C[2,2] + 1)**2 +
                  (C[2,1] + C[1,2])**2)

    qk = .25*np.sqrt((C[1,0] - C[0,1])**2 +
                  (C[2,0] + C[0,2])**2 +
                  (C[2,1] + C[1,2])**2 +
                  (C[2,2] - C[0,0] - C[1,1] + 1)**2)

    if qi_sign < 0:
        qi = -qi

    if qj_sign < 0:
        qj = -qj

    if qk_sign <0:
        qk = -qk

    return real, np.hstack([qi, qj, qk])

if __name__ == "__main__":
    # plt.ion()
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

    
    ca = 0.2
    cd = 0.3
    Qa = np.identity(3)*1e-1
    Qb = np.identity(3)*1e-1
    Qd = np.identity(3)*1e-2

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

    qrc = 1
    qic = np.array([0,0,0])

    w1 = .1
    w2 = 1-w1


    triad_quat = []
    complementary_quat = []
    ekf_quat = []
    Ps = []
    Ks = []

    iters = 600
    for mag, rate, accel in zip(moving_mag, moving_rate, moving_accel):
    # while True:
        # mag, rate, accel = get_data(arduino)


        # print(w_bias,m_bias,lin_accel)

        rate = rate/180*np.pi

        qr, qi = quat_update(qr, qi, -(rate-w_bias)*dt)

        S = np.linalg.norm(np.hstack([qr, qi]))
        qr = qr/S
        qi = qi/S

        DCM = quat2dcm(qr, qi)

        Q = get_Q(P, Qa, Qb, Qd, R_mag, R_rate, R_accel, ca, cd, dt)
        P = Q.copy()

        g_gyro = DCM@g_abs
        m_gyro = DCM@mag_abs

        H = get_H(g_gyro, m_gyro, dt)

        K = P@H.T@np.linalg.inv(H@P@H.T + R)

        lin_accel = lin_accel*ca
        m_bias = m_bias*cd

        g_accel = accel - lin_accel

        z = np.hstack([g_accel - g_gyro,
                      (mag-m_bias)     - m_gyro])

        X = np.zeros(12)
        X = K@z

        P = (np.identity(12) - K@H)@P

        qr, qi = quat_update(qr, qi, X[0:3])
        S = np.linalg.norm(np.hstack([qr, qi]))
        qr = qr/S
        qi = qi/S
        
        w_bias = w_bias - X[3:6]
        lin_accel = lin_accel -X[6:9]
        m_bias = m_bias - X[9:12]

        DCM = quat2dcm(qr, qi)

        ekf_quat.append(np.hstack([qr, qi]))
        Ps.append(P.copy())
        Ks.append(K.copy())

        #Complementary time

        qrt, qit = TRIAD(mag, accel, mag_abs, g_abs)
        qit = -qit
        qrw, qiw = quat_update(qrc, qic, -rate*dt)

        S = np.linalg.norm(np.hstack([qrw, qiw]))
        qrw = qrw/S
        qiw = qiw/S


        q1 = np.hstack([qrt, qit])
        q2 = np.hstack([qrw, qiw])
        z  =np.sqrt((w1 - w2)**2 + 4*w1*w2*(np.dot(q1, q1)**2))

        q = (np.sqrt((w1*(w1 - w2 + z))/(z*(w1 + w2 + z)))*q1 + 
             np.sign(np.dot(q1,q2))*np.sqrt((w2*(w2 - w1 + z))/(z*(w1 + w2 + z)))*q2)

        qrc = q[0]
        qic = q[1:4]

        DCM2 = quat2dcm(qrt, qit)
        DCM3 = quat2dcm(qrw, qiw)
        DCM4 = quat2dcm(qrc, qic)

        complementary_quat.append(np.hstack([qrc, qic]))
        triad_quat.append(np.hstack([qrt, qit]))

        # plotlive([DCM4, DCM2, DCM3, DCM])


    t = np.asarray([xx*dt for xx in range(len(moving_mag))])
    triad_quat = np.vstack(triad_quat)
    complementary_quat = np.vstack(complementary_quat)
    ekf_quat = np.vstack(ekf_quat)

    for ii in range(len(moving_mag)):

        if triad_quat[ii,1] < 0:
            triad_quat[ii,:] = -triad_quat[ii,:]


        if complementary_quat[ii,1] < 0:
            complementary_quat[ii,:] = -complementary_quat[ii,:]


        if ekf_quat[ii,1] < 0:
            ekf_quat[ii,:] = -ekf_quat[ii,:]

    Ks = np.stack(Ks, axis = 2)
    Ps = np.stack(Ps, axis = 2)

    print(triad_quat)

    plt.figure()
    plt.plot(triad_quat, 'r')
    plt.plot(complementary_quat,'b')
    plt.plot(ekf_quat,'k')
    plt.title('qs')
    plt.legend(['Triad, Complementary, EKF'])
    
    plt.figure()
    plt.title('Ps')
    for ii in range(3):
        for jj in range(3):
            plt.semilogy(abs(Ps[ii,jj,:]))

    plt.figure()
    plt.title('Ks')
    for ii in range(3):
        for jj in range(3):
            plt.semilogy(abs(Ks[ii,jj,:]))

    plt.figure()
    Ts = np.asarray([np.trace(Ps[:,:,x]) for x in range(len(moving_mag))])
    plt.plot(Ts)

    np.save('Ps', Ps)

    plt.show()