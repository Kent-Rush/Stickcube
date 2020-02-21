from numpy import *
from numpy.linalg import *

def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1], A[0], 0]])


def quaternion_derivative(real, imag, w):
    #E is vector component
    #n is scalar component
    #w is angular rate in radians/s

    dimag = .5*(real*identity(3) + crux(imag))@w
    dreal = -.5*dot(imag, w)
    return dreal, dimag

def angular_rate_derivative(w, I, Torque):
    return inv(I)@(Torque - crux(w)@I@w)

def quat2dcm(n, E):
    """
    generates the active rotation matrix from a quaternion.
    :param n: scalar part
    :param E: vector part

    """
    
    frame_rotation = (2*n**2 - 1)*identity(3) + 2*outer(E,E) - 2*n*crux(E) 

    return frame_rotation.T

def euler_rate_deriv(eulers):
    """
    Assumes the standard x y z sequence
    """

    phi = eulers[0]
    theta = eulers[1]

    A = array([[1, sin(phi)*tan(theta),     cos(phi)*tan(theta)],
               [0, cos(phi),               -sin(phi)],
               [0, sin(phi)*(1/cos(theta)), cos(phi)*(1/cos(theta))]])

    return A.T




def inertia_cuboid(x, y, z, m):
    xx = 1/12*m*(z**2 + y**2)
    yy = 1/12*m*(x**2 + y**2)
    zz = 1/12*m*(x**2 + z**2)
    return diag([xx,yy,zz])

# def dcm2quat(C):

#     """
#     Takes a rotation matrix and returns the quaternion such that
#     the when applying a rotation via quaternions the same result will be yielded.
#     """

#     #This function assumes a frame rotation, this library assumes active rotations
#     C = C.T


#     E = zeros(3)
#     n = 0
#     tr = trace(C)

#     if (tr > 0):
#         n = sqrt( tr + 1 )/2

#         E[0] = (C[1, 2] - C[2, 1])/(4*n)
#         E[1] = (C[2, 0] - C[0, 2])/(4*n) 
#         E[2] = (C[0, 1] - C[1, 0])/(4*n) 
#     else:
#         d = diag(C)
#         if max(d) == d[1]:

#             sq_trace = sqrt(d[1] - d[0] - d[2] + 1 )

#             E[1] = .5*sq_trace 

#             if sq_trace != 0:
#                 sq_trace = .5/sq_trace

#             n    = (C[2, 0] - C[0, 2])*sq_trace 
#             E[0] = (C[0, 1] + C[1, 0])*sq_trace
#             E[2] = (C[1, 2] + C[2, 1])*sq_trace

#         elif max(d) == d[2]:
#             sq_trace = sqrt(d[2] - d[0] - d[1] + 1)

#             E[2] = .5*sq_trace 

#             if sq_trace != 0:
#                 sq_trace = .5/sq_trace

#             n    = (C[0, 1] - C[1, 0])*sq_trace
#             E[0] = (C[2, 0] + C[0, 2])*sq_trace 
#             E[1] = (C[1, 2] + C[2, 1])*sq_trace
#         else:
#             sq_trace = sqrt(d[0] - d[1] - d[2] + 1)

#             E[0] = .5*sq_trace 

#             if sq_trace != 0:
#                 sq_trace = .5/sq_trace

#             n    = (C[1, 2] - C[2, 1])*sq_trace 
#             E[1] = (C[0, 1] + C[1, 0])*sq_trace
#             E[2] = (C[2, 0] + C[0, 2])*sq_trace


#     return n, E

def dcm2quat(C):

    """
    From "A Survey on the Computation of Quaternions from Rotation Matrices" by Soheil and Federico
    """

    qi_sign = C[2,1] - C[1,2]
    qj_sign = C[0,2] - C[2,0]
    qk_sign = C[1,0] - C[0,1]

    real = .25*sqrt((C[0,0] + C[1,1] + C[2,2] + 1)**2 +
                    (C[2,1] - C[1,2])**2 +
                    (C[0,2] - C[2,0])**2 +
                    (C[1,0] - C[0,1])**2)

    qi = .25*sqrt((C[2,1] - C[1,2])**2 +
                  (C[0,0] - C[1,1] - C[2,2] + 1)**2 +
                  (C[1,0] + C[0,1])**2 +
                  (C[2,0] + C[0,2])**2)

    qj = .25*sqrt((C[0,2] - C[2,0])**2 +
                  (C[1,0] + C[0,1])**2 +
                  (C[1,1] - C[0,0] - C[2,2] + 1)**2 +
                  (C[2,1] + C[1,2])**2)

    qk = .25*sqrt((C[1,0] - C[0,1])**2 +
                  (C[2,0] + C[0,2])**2 +
                  (C[2,1] + C[1,2])**2 +
                  (C[2,2] - C[0,0] - C[1,1] + 1)**2)

    if qi_sign < 0:
        qi = -qi

    if qj_sign < 0:
        qj = -qj

    if qk_sign <0:
        qk = -qk

    return real, hstack([qi, qj, qk])





def axis_angle2dcm(axis, angle, degrees = False):

    """
    This function returns the dcm corresponding to an active rotation about an axis by and angle.
    degrees = True if using degrees
    use the transpose of this funciton to convert a vector into a frame
    """

    if degrees:
        angle = radians(angle)

    axis = axis/norm(axis)

    active_rotation = cos(angle)*identity(3) + (1 - cos(angle))*outer(axis, axis) + sin(angle)*crux(axis)

    return active_rotation

def axis_angle2quat(axis, angle, degrees = False):

    if degrees:
        angle = radians(angle)

    axis = axis/norm(axis)

    imag = axis*sin(angle/2)
    real = cos(angle/2)

    return imag, real

def quat_mult(real1, imag1, real2, imag2):
    """
    Multiplies two quaternions
    q1 (x) q2
    """

    imag3 = real1*imag2 + real2*imag1 + crux(imag1)@imag2
    real3 = real2*real1 - np.dot(imag1, imag2)

    return real3, imag3

# def quat_rot(q_real, q_imag, vector):

#     """
#     Performs the active rotation about the axis and angle represented by the quaternion
#     """

#     imag_temp, real_temp = quat_mult(vector, 0, -q_imag, q_real)
#     v_rot, _ = quat_mult(q_imag, q_real, imag_temp, real_temp)

#     return v_rot

def quat_rot(q_real, q_imag, vector):

#     """
#     Performs the active rotation about the axis and angle represented by the quaternion
#     """

    imag_temp = q_real*vector + crux(vector)@-q_imag
    real_temp = - dot(vector, -q_imag)

    v_rot = q_real*imag_temp + real_temp*q_imag + crux(q_imag)@imag_temp

    return v_rot

def dcm2axis_angle(dcm):

    angle = arccos((trace(dcm)-1)/2)

    if angle != 0:
        axis = 1/(2*sin(angle))*array([C[2,1] - C[1,2],
                                       C[0,2] - C[2,0],
                                       C[1,0] - C[0,1]])
    else:
        axis = array([1,0,0])

    return axis, angle




def propagate_quats(t, state):
    eta = state[0]
    eps = state[1:4]
    omega = state[4:]

    inertia = identity(3)

    deps = .5*(eta*identity(3) + crux(eps))@omega
    deta = -.5*dot(eps, omega)
    domega = -inv(inertia)@crux(omega)@inertia@omega

    derivatives = hstack([deta, deps, domega])

    return derivatives


def Cz(ang, degrees = False):
    if degrees:
        ang = radians(ang)

    c = cos(ang)
    s = sin(ang)
    
    C = array([[c,-s,0],
               [s,c,0],
               [0,0,1]])
    return C

def Cy(ang, degrees = False):
    if degrees:
        ang = radians(ang)

    c = cos(ang)
    s = sin(ang)
    
    C = array([[c,0,s],
               [0,1,0],
               [-s,0,c]])
    
    return C

def Cx(ang, degrees = False):
    if degrees:
        ang = radians(ang)

    c = cos(ang)
    s = sin(ang)
         
    C = array([[1,0,0],
               [0,c,-s],
               [0,s,c]])
    
    return C

def euler2dcm(sequence, angles, degrees = False):

    if degrees:
        angles = [radians(x) for x in angles]

    C = identity(3)
    for xyz, angle in zip(sequence.lower(), angles):
        if xyz == 'x':
            C = Cx(angle)@C
        elif xyz == 'y':
            C = Cy(angle)@C
        elif xyz == 'z':
            C = Cz(angle)@C

    return C


def mrp2dcm(mrps):

    MRPS = norm(mrps)

    dcm = identity(3) - 4*(1 - MRPS**2)/((1 + MRPS**2)**2)*crux(mrps) + 8/((1 + MRPS**2)**2)*crux(mrps)@crux(mrps)
    
    return dcm.T


def axis_angle2mrp(axis, angle, degrees = False):

    if degrees:
        angle = radians(angle)

    axis = axis/norm(axis)

    return axis*tan(angle/4)

def mrp_derivative(mrps, omega):

    MRPS = norm(mrps)

    return .25*((1- MRPS**2)*identity(3) + 2*crux(mrps) + 2*outer(mrps, mrps))@omega

def quat2mrp(real, imag):

    return imag/(1 + real)

def mrp2quat(mrps):

    MRPS_2 = norm(mrps)**2
    real = (1 - MRPS_2)/(1 + MRPS_2)
    imag = 2*mrps/( 1 + MRPS_2)

    return real, imag

def dcm2mrp(dcm):

    real, imag = dcm2quat(dcm)

    mrp = quat2mrp(real, imag)

    return mrp


if __name__ == '__main__':

    from pyquaternion import Quaternion
    from scipy.integrate import ode
    from scipy.spatial.transform import Rotation as R


    quat = array([1,2,3,4])/norm(array([1,2,3,4]))

    real = quat[0]
    imag = quat[1:4]

    v = array([1,1,1])

    dcm_q = quat2dcm(real, imag)

    mrps = quat2mrp(real, imag)

    qcr, qci = mrp2quat(mrps)

    #print(real, imag, qcr, qci)

    dcm_mrp = mrp2dcm(mrps)

    vq = dcm_q@v
    vmrps = dcm_mrp@v

    mrps_check = dcm2mrp(dcm_mrp)
    dcm_check_check = mrp2dcm(mrps_check)
    mrps_check_check = dcm2mrp(dcm_check_check)

    for i in range(1000):

        q = R.random().as_quat()
        q = q/norm(q)
        real = q[0]
        imag = q[1:]

        mrps = quat2mrp(real, imag)

        dcm = quat2dcm(real, imag)
        dcm2 = mrp2dcm(mrps)

        mrps2 = dcm2mrp(dcm2)

        dcm3 = mrp2dcm(mrps2)

        print(norm( dcm2 - dcm3))


    # dcm = axis_angle2dcm(array([0,0,1]), pi/2)
    # real, imag = dcm2quat(dcm)

    # print(quat_rot(real, imag, array([1,0,0])))

    # for i in range(1000):

    #     dcm = R.random().as_dcm()

    #     print(det(dcm))
    #     print(hstack(dcm2quat(dcm)))
    #     print(hstack(cayleys(dcm)))
    # v = array([1, 0, 0])
    # axis = array([0, 0, 1])
    # angle = pi/2

    # q_imag, q_real = axis_angle2quat(axis, angle)

    # q_reff = Quaternion(axis = axis, angle = angle)
    # v_rot_reff  = q_reff.rotate(v)

    # print(q_reff)
    # print(q_imag, q_real)

    # print(quat_rot(q_imag, q_real, v))
    # print(v_rot_reff)

    # dcm = quat2dcm(q_imag, q_real)

    # print(dcm@v)

    # print(dcm2quat(dcm))

    # def propagate(t, state, w):

    #     eulers = state[0:3]

    #     return euler_rate_deriv(eulers)@w


    # t_sim = 10
    # w = pi/2/t_sim

    # state0 = hstack([0,0,0])

    # solver = ode(propagate)
    # solver.set_integrator('lsoda')
    # solver.set_initial_value(state0, 0)
    # solver.set_f_params(array([0,0,w]))

    # newstate = []
    # time = []

    # DT = .01

    # while solver.successful() and solver.t < t_sim:

    #     newstate.append(solver.y)
    #     time.append(solver.t)

    #     solver.integrate(solver.t + DT)

    # newstate = vstack(newstate)
    # time = hstack(time)

    # print(newstate[-1])
    # print([0, 0, pi/2])

    # # angles = [45, 45, 45]
    # # v = array([1, 0, 0])

    # # dcm_reff = Cz(pi/4)@Cy(pi/4)@Cx(pi/4)
    # # my_dcm  = euler2dcm('xyz', angles, degrees = True)

    # # print(my_dcm@v)
    # # print(dcm_reff@v)

    # print('I have flipped the principal rotation matrices to match with the active rotation stuff. This means they are flipped from the SDAC')


