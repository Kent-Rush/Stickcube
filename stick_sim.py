from numpy import *
from numpy.linalg import *
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.animation as animation

import sys
sys.path.insert(0, '../Aero_Funcs')

import Controls_Funcs as CF
import Aero_Plots as AP

set_printoptions(precision = 2)

def main():

    #initial Conditions:
    #coefficient of draf
    Cd = .1
    #mass of the cube
    mass = 1
    #positoin of the cube in body frame
    cube_pos_body = array([0,0,1])
    #matrix of the wheel inertias along the diagonal, taking advantage
    #of the fact that htey are essentially only spinning in 2d
    Iwheels = identity(2)
    #matrix definig the axes of rotations of each wheel
    #each column is the axis of rotation
    Awheels = vstack([array([1,0]), array([0,1])]).T

    #defining the initial attitude quaternion
    axis = array([0,1,0])
    angle = pi/8
    q_real = cos(angle/2)
    q_imag = axis*sin(angle/2)

    #initial angular velocity
    omega = array([.5,0,0])

    #initial wheel velocities
    omega_w1 = 0
    omega_w2 = 0

    #inertia matrix of the cube at the geometric center
    inertia = CF.inertia_cuboid(.1, .1, .1, mass)
    #parallel axis theorem
    inertia = inertia - mass*CF.crux(-cube_pos_body)@CF.crux(-cube_pos_body)
    #create total initial states
    state0 = hstack([q_real, q_imag, omega, omega_w1, omega_w2])

    #initialzie the ode solver
    solver = ode(propagate)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state0, 0)
    solver.set_f_params(inertia, Iwheels, Awheels, mass, Cd, cube_pos_body)

    #simulate the system for 30 seconds and get data ever 0.1 seconds
    tspan = 20
    dt = 1/60
    newstates = []
    while solver.successful() and solver.t < tspan:
        solver.integrate(solver.t + dt)
        newstates.append(solver.y)
    
    newstates = vstack(newstates)

    #newstates = vstack([solver.integrate(t) for t in linspace(0, 10, 100, dtype = float)])

    #Calculate the position of the cube at every timestep
    radius = vstack([CF.quat2dcm(state[0], state[1:4])@cube_pos_body for state in newstates])
    angle = hstack([arccos(dot(array([0,0,1]), r/norm(r))) for r in radius])

    plt.figure()
    plt.plot(angle)
    plt.title('Angle from Vertical')

    #plotting poop
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-.3,.3)
    ax.set_ylim(-.3,.3)
    ax.set_zlim(-.3,.3)
    ax.plot([0,.3], [0,0], [0,0],'k')
    ax.plot([0, 0], [0,.3], [0,0],'k')
    ax.plot([0, 0], [0,0], [0,.3],'k')
    #ax.axis('equal')
    AP.plot_earth(ax, radius = .3)

    fig = plt.figure()
    plt.plot(newstates[:,4:7])
    plt.title('Body Rates')

    plt.figure()
    plt.plot(newstates[:,7:9])
    plt.title('Wheel Rates')

    torques = []
    actual_torques = []
    for state in newstates:
        g = 9.81
        q_real = state[0]
        q_imag = state[1:4]
        omega = state[4:7] #body frame angular velocity
        wheels = state[7:9]
        dcm_body2eci = CF.quat2dcm(q_real, q_imag)
        a_gravity = -array([0,0,g])
        Tgrav = cross(cube_pos_body, dcm_body2eci.T@a_gravity*mass)
        Tcontrol = -.1*q_imag -.1*omega - Tgrav

        alpha_wheels = inv(Awheels@Iwheels)@Tcontrol[0:2]

        torques.append(Tcontrol)
        actual_torques.append(Awheels@Iwheels@alpha_wheels)

    torques = vstack(torques)
    actual_torques = vstack(actual_torques)

    plt.figure()
    plt.plot(torques)
    plt.plot(actual_torques, 'k')
    plt.title('Control Torques')

    plt.figure()
    plt.plot(newstates[:,1:4])
    plt.title('Quaternion Axis')


    lines = []
    for index in range(len(radius)-1):
        last = index - 10
        if last < 0:
            last = 0
        state = newstates[index]
        dcm_body2eci = CF.quat2dcm(state[0], state[1:4])
        xbdy = vstack([radius[index], radius[index] + dcm_body2eci[:,0]*.1])
        ybdy = vstack([radius[index], radius[index] + dcm_body2eci[:,1]*.1])
        zbdy = vstack([radius[index], radius[index] + dcm_body2eci[:,2]*.1])



        px, = ax.plot(xbdy[:,0], xbdy[:,1], xbdy[:,2], 'g')
        py, = ax.plot(ybdy[:,0], ybdy[:,1], ybdy[:,2], 'g')
        pz, = ax.plot(zbdy[:,0], zbdy[:,1], zbdy[:,2], 'g')

        pt, = ax.plot([radius[index,0]], [radius[index,1]], [radius[index,2]],'ro')
        trail, = ax.plot(radius[last:index,0], radius[last:index,1], radius[last:index,2], 'k') 
        lines.append([pt, trail, px, py, pz])

    anim = animation.ArtistAnimation(fig, lines, interval = 50, blit = True, repeat = True)
    plt.show()
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save('animation.mp4', writer=writer)

def propagate(t, state, inertia, Iwheels, Awheels, mass, Cd, r_body):

    #pull out state variables
    g = 9.81
    q_real = state[0]
    q_imag = state[1:4]
    omega = state[4:7] #body frame angular velocity
    wheels = state[7:9]

    #create the rotation matrix from body to eci
    dcm_body2eci = CF.quat2dcm(q_real, q_imag)

    #calculate disturbance torques
    a_gravity = -array([0,0,g])
    Tgrav = cross(r_body, dcm_body2eci.T@a_gravity*mass)
    Tdrag = -omega*Cd

    #calculate a control torquw
    Tcontrol = -.5*q_imag -.5*omega - Tgrav
    #Tcontrol = zeros(3)

    #calculate the acceleration of the wheels (we control those)
    #we ignore the z component of the torque because we dont have a
    #wheel that could ever respond to that axis
    alpha_wheels = inv(Awheels@Iwheels)@Tcontrol[0:2]
    #add whatever disturbance torques here that you want to account for
    Tdisturbance = Tgrav

    #disgusting equation that I wrote on the board
    #                                       This is our control moment
    d_omega = inv(inertia)@(Tdisturbance + hstack([Awheels@Iwheels@alpha_wheels, 0])
                            - cross(omega, inertia@omega
    #                                               This is our wheel momentum
                                           + hstack([Awheels@Iwheels@wheels,0]) ) )

    #kinematic equations of quaternion derivatives
    d_imag = .5*(q_real*identity(3) + CF.crux(q_imag))@omega
    d_real = -.5*dot(q_imag, omega)


    d_wheels = -alpha_wheels

    #print(hstack([d_real, d_imag, d_omega, d_wheels]))

    return hstack([d_real, d_imag, d_omega, d_wheels])



if __name__ == '__main__':
    main()

