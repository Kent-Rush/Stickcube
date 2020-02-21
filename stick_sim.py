from numpy import *
from numpy.linalg import *
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import sys
sys.path.insert(0, '../Aero_Funcs')

import Controls_Funcs as CF
import Aero_Plots as AP


def main():

    
    Cd = .5

    axis = array([1,0,0])
    angle = pi

    dcm = CF.axis_angle2dcm(axis = axis, angle = angle, degrees = False)
    q_real, q_imag = CF.dcm2quat(dcm)
    omega = array([2,2,0])


    pos = array([0,0,.3])
    # vel = array([0.1,0,0])
    

    mass = 1
    inertia = CF.inertia_cuboid(.1, .1, .1, mass)
    inertia = inertia - mass*CF.crux(-pos)@CF.crux(-pos)
    #print(inertia)

    state0 = hstack([q_real, q_imag, omega])

    solver = ode(propagate)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state0, 0)
    solver.set_f_params(inertia, mass, Cd, pos)

    times = arange(0, 400, 1, dtype = float)

    tspan = 100
    dt = .1
    newstates = []
    while solver.successful() and solver.t < tspan:
        solver.integrate(solver.t + dt)
        newstates.append(solver.y)
    newstates = vstack(newstates)

    radius = vstack([CF.quat2dcm(state[0], state[1:4])@pos for state in newstates])
    plt.figure()
    plt.plot(norm(radius, axis = 1))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(radius[:,0], radius[:,1], radius[:,2])
    ax.scatter(radius[-1, 0], radius[-1,1], radius[-1,2])
    AP.plot_earth(ax, radius = .3)
    ax.set_xlim(-.3,.3)
    ax.set_ylim(-.3,.3)
    ax.set_zlim(-.3,.3)
    #ax.axis('equal')
    
    plt.show()



def propagate(t, state, inertia, mass, Cd, r_body):

    g = 9.81

    q_real = state[0]
    q_imag = state[1:4]
    omega = state[4:7]

    a_gravity = -array([0,0,g])

    dcm_body2eci = CF.quat2dcm(q_real, q_imag)

    #pos = dcm_body2eci@r_body

    T = cross(r_body, dcm_body2eci.T@a_gravity*mass)
    Tdrag = -omega*Cd



    Tcontrol = -.1*q_imag + -.1*omega

    d_omega = inv(inertia)@(Tcontrol - cross(omega, inertia@omega))

    d_imag = .5*(q_real*identity(3) + CF.crux(q_imag))@omega
    d_real = -.5*dot(q_imag, omega)

    return hstack([d_real, d_imag, d_omega])



if __name__ == '__main__':
    main()

