from numpy import *
from numpy.linalg import *
from scipy.integrate import ode
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, diff

import Controls_Funcs as CF
import Aero_Plots as AP

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation

from Sensors import *

def measurements(x, u, r_body, mag_i):

    q_real = x[0]
    q_imag = x[1:4]
    w_b_i_b = x[4:7]

    dcm_body2eci = CF.quat2dcm(q_real, q_imag)
    accelerometer = CF.crux(w_b_i_b)@CF.crux(w_b_i_b)@r_body + CF.crux(u)@r_body + dcm_body2eci.T@array([0, 0, 9.81])
    gyro = w_b_i_b
    magnetometer = dcm_body2eci.T@mag_i

    return accelerometer, gyro, magnetometer

def TRIAD(a_b, b_b, a_i, b_i):


    x_b = a_b/norm(a_b)
    y_b = cross(a_b, b_b)/norm(cross(a_b, b_b))
    z_b = cross(x_b, y_b)

    x_i = a_i/norm(a_i)
    y_i = cross(a_i, b_i)/norm(cross(a_i, b_i))
    z_i = cross(x_i, y_i)

    b2t = vstack([x_b, y_b, z_b])
    i2t = vstack([x_i, y_i, z_i])

    i2b = b2t.T@i2t

    q_real, q_imag = CF.dcm2quat(i2b.T)

    return q_real, q_imag

def main():

    true_states = load('true_states.npy')
    d_omegas = load('d_omega.npy')

    magnetic_field_inertial = array([1,1,1])

    r_body = array([0,0,.3])

    bias_std        = 0.01
    ecrw_std        = 0.005
    white_noise_std = 0.01

    Gyro = RateGyro(bias_std, ecrw_std, white_noise_std)
    Accel = Accelerometer(bias_std, white_noise_std)
    Mag = Magnetometer(white_noise_std)

    AccelMeasurements = zeros((len(true_states), 3))
    GyroMeasurements  = zeros((len(true_states), 3))
    MagMeasurements  = zeros((len(true_states), 3))


    quat_ests = zeros((len(true_states), 4))
    for ii, state in enumerate(true_states):
        a_m0, w_m0, m_m0 = measurements(state, d_omegas[ii], r_body, magnetic_field_inertial)

        a_m = Accel.measure(a_m0)
        w_m = Gyro.measure(w_m0)
        m_m = Mag.measure(m_m0)

        gravity_body = a_m - CF.crux(w_m)@CF.crux(w_m)@r_body

        q_real, q_imag = TRIAD(gravity_body, m_m, array([0, 0, 9.81]), magnetic_field_inertial)
        quat_ests[ii] = hstack([q_real, q_imag])


    angular_error = zeros(len(true_states))
    for ii in range(len(quat_ests)):

        true_states[ii, 0:4] = true_states[ii, 0:4]/norm(true_states[ii, 0:4])

        err_real, err_imag = CF.quat_mult(quat_ests[ii,0], quat_ests[ii,1:4], true_states[ii,0], -true_states[ii,1:4]) 
        angular_error[ii] = arccos(err_real)*2


    plt.plot(angular_error)

    # #plotting poop
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-.3,.3)
    ax.set_ylim(-.3,.3)
    ax.set_zlim(-.3,.3)
    ax.plot([0,.3], [0,0], [0,0],'k')
    ax.plot([0, 0], [0,.3], [0,0],'k')
    ax.plot([0, 0], [0,0], [0,.3],'k')

    lines = []
    for true, est in zip(true_states, quat_ests):
        dcm_body2eci = CF.quat2dcm(true[0], true[1:4])
        xbdy = vstack([zeros(3), dcm_body2eci[:,0]*.1])
        ybdy = vstack([zeros(3), dcm_body2eci[:,1]*.1])
        zbdy = vstack([zeros(3), dcm_body2eci[:,2]*.1])



        px, = ax.plot(xbdy[:,0], xbdy[:,1], xbdy[:,2], 'g')
        py, = ax.plot(ybdy[:,0], ybdy[:,1], ybdy[:,2], 'g')
        pz, = ax.plot(zbdy[:,0], zbdy[:,1], zbdy[:,2], 'g')

        dcm_body2eci = CF.quat2dcm(est[0], est[1:4])
        xbdy = vstack([zeros(3), dcm_body2eci[:,0]*.1])
        ybdy = vstack([zeros(3), dcm_body2eci[:,1]*.1])
        zbdy = vstack([zeros(3), dcm_body2eci[:,2]*.1])



        ex, = ax.plot(xbdy[:,0], xbdy[:,1], xbdy[:,2], 'b')
        ey, = ax.plot(ybdy[:,0], ybdy[:,1], ybdy[:,2], 'b')
        ez, = ax.plot(zbdy[:,0], zbdy[:,1], zbdy[:,2], 'b')



        lines.append([px, py, pz, ex, ey, ez])

    anim = animation.ArtistAnimation(fig, lines, interval = 50, blit = True, repeat = True)
    plt.show()





if __name__ == '__main__':
    main()  

