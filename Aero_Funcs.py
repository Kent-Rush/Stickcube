import os, sys
import numpy as np
from numpy.linalg import *
from numpy import *
import scipy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Aero_Plots import *
import time
import datetime
import jplephem

def coes2rv(a,ecc,TA,inc,RAAN,Arg_p,mu=398600):
    
    radTA = radians(TA)
    
    rp = a*(1-ecc**2)/(1+ecc)
    h = sqrt(rp*mu*(1+ecc))
    r_perifocal = (h**2/mu)*(1/(1 + ecc*cos(radTA)))*np.array([cos(radTA),sin(radTA),0]).T
    v_perifocal = (mu/h)*np.array([-sin(radTA),ecc+cos(radTA),0]).T
    ECI2P = Cz(Arg_p)@Cx(inc)@Cz(RAAN)
    P2ECI = ECI2P.T
    r = P2ECI@r_perifocal
    v = P2ECI@v_perifocal
    
    return r, v

def Cz(ang):
    ang = radians(ang)
    c = cos(ang)
    s = sin(ang)
    
    C = np.array([[c,s,0],
                   [-s,c,0],
                   [0,0,1]])
    return C

def Cy(ang):
    ang = radians(ang)
    c = cos(ang)
    s = sin(ang)
    
    C = np.array([[c,0,-s],
                   [0,1,0],
                   [s,0,c]])
    
    return C

def Cx(ang):
    ang = radians(ang)
    c = cos(ang)
    s = sin(ang)
         
    C = np.array([[1,0,0],
                       [0,c,s],
                       [0,-s,c]])
    
    return C

def rv2coes(r,v,mu=398600):
    K = array([0,0,1])
    h = cross(r,v)
    hmag = norm(h)
    vmag = norm(v)
    rmag = norm(r)
    vr = (dot(r,v))/norm(r)
    
    inc = degrees(arccos(h[2]/hmag))
    nodeline = cross(K,h)
    nodelinemag = norm(nodeline)
    hvec = cross(r,v)
    h = norm(hvec)
    
    if nodelinemag == 0:
        raanquad = 0
    else:
        raanquad = nodeline[0]/nodelinemag
    if nodeline[1] > 0:
        RAAN = degrees(arccos(raanquad))
    else:
        RAAN = 360 - degrees(arccos(raanquad))
    
    
    ecc = (1/mu) * ((vmag**2 - (mu/rmag))*r - rmag*vr*v)
    emag = sqrt(1 + (h**2/mu**2)*(vmag**2 - (2*mu)/rmag))
    energy = -.5*(mu**2/h**2)*(1-emag**2)
    
    #w = argument of perigee
    if nodelinemag == 0:
        arg_p = 0
    else:
        if ecc[2] >= 0:
            arg_p = degrees(arccos((dot(nodeline,ecc))/(nodelinemag*emag)))
        else:
            arg_p = 360 - degrees(arccos((dot(nodeline,ecc))/(nodelinemag*emag)))
    
    #True Anomaly
    if vr >= 0:
        TA = degrees(arccos(dot(ecc,r)/(norm(ecc)*rmag)))
    else:
        TA = 360 - degrees(arccos(dot(ecc,r)/(emag*rmag)))
    
    #Time since perigee
    rad_TA = (TA/180)*pi;
    E = 2*arctan(sqrt((1-emag)/(1+emag))*tan(rad_TA/2))
    Me = E - emag*sin(E);
    t0 = (Me*h**3/mu**2)/(1-emag**2)**(3/2)
    
    #semimajor axis
    semi_a = rmag*(1+emag*cos(rad_TA))/(1-emag**2)
    rp = (h**2/mu)*(1/(1+emag))
    ra = 2*semi_a - rp 
    #Period
    T = (2*pi*semi_a**(3/2))/sqrt(mu)
    
    coes = {'Inc':inc, 'RAAN':RAAN, 'Arg_p': arg_p, 'TA': TA, 'Semi_a':semi_a, 'Ecc':emag,
            'R_peri':rp, 'R_apo':ra, 'Period':T, 'Energy':energy, 'Ang_momentum':h, 'Time_since_peri':t0}

    return coes
    
    

def atm_dens(z):
    #Height ranges
    h = [ 0, 25, 30, 40, 50, 60, 70,
         80, 90, 100, 110, 120, 130, 140,
         150, 180, 200, 250, 300, 350, 400,
         450, 500, 600, 700, 800, 900, 1000]
    #Densities
    r = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
         1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
         2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
         1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15]
    #Scale height
    H = [ 7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
         5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
         21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
         60.980, 65.654, 76.377, 100.587, 147.203, 208.020]
    
    if z > 1000:
        z = 1000
    elif z < 0:
        z = 0
        
    i = 0
    #Determine the interpolation interval:
    for j in range(0,26):
        if z >= h[j] and z < h[j+1]:
            i = j;
    
    if z >= 1000:
        i = 26
    
    density = r[i]*exp(-(z - h[i])/H[i])
    
    return density

def julian_date(date):
    #Returns the julian date in days
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    J = 367*year - floor((7*(year+floor((month+9)/12)))/4) + floor((275*month)/9) + day + 1721013.5
    UT = hour + (minute/60) + (second/3600)
    J = J + (UT/24)
    return J

def vect_earth_to_sun(date):
    #returns a distance vector from the earth
    #to the sun in AU
    AU = 149597870.691
    JD = julian_date(date)
    n = (JD - 2451545)
    cy = n/36525
    L = (280.459 + 0.98564736*n)%360
    M = (357.529 + 0.98560023*n)%360
    gam = (L + 1.915*sin(radians(M)) + .02*sin(radians(2*M)))%360
    E = 23.439 - (3.56e-7)*n
    r = 1.00014 - 0.01671*cos(radians(M)) - .00014*cos(radians(2*M)) #AU
    r = r*AU
    u = np.array([cos(radians(gam)),
                  sin(radians(gam))*cos(radians(E)),
                  sin(radians(gam))*sin(radians(E))])
    #r is the vector magnitude in AU
    #u is the unit vector of r
    return r*u #km

def shadow(rbody_to_sat,rbody_to_sun, rbody = 6378):
    #Rough approximation of whether or not sc is in shadow
    theta = degrees(arccos(dot(rbody_to_sun,rbody_to_sat)/(norm(rbody_to_sun)*norm(rbody_to_sat))))
    t1 = degrees(arccos(rbody/norm(rbody_to_sat)))
    t2 = degrees(arccos(rbody/norm(rbody_to_sun)))
    if t1 + t2 <= theta:
        #sc is in shadow
        return 0
    else:
        #sc is illuminated
        return 1
    

def ME2TA(Me, ecc):
    Me = radians(Me)
    
    f = lambda x: x - ecc*sin(x) - Me
    
    if Me < pi:
        guess = Me + ecc
    else:
        guess = Me - ecc
        
    E = fsolve(f,guess,xtol = 1e-9)[0]
    
    angle = 2*arctan(tan(E/2)/sqrt((1-ecc)/(1+ecc)))
    angle = degrees(angle)
    return angle
    

def tle2coes(filename):
    file = open(filename)
    coes = {}
    mu = 398600
    for line in file.readlines():
        elements = line.split()
        if elements[0] == '1':
            year = 2000 + int(elements[3][:2])
            daynumber = float(elements[3][2:])
            time = datetime.datetime(year = year, month = 1, day = 1)
            time += datetime.timedelta(days = daynumber)
            coes['time'] = time
        elif elements[0] == '2':
            #In degrees
            inc = float(elements[2])
            RAAN = float(elements[3])
            arg_p = float(elements[5])
            mean_anomaly = float(elements[6])
            ecc = float('.'+elements[4])
            mean_motion = float(elements[7])
            
            TA = ME2TA(mean_anomaly,ecc)%360
            Period = 24*3600/mean_motion
            mean_motion = (2*pi/86400)*mean_motion
            semi_a = (mu/mean_motion**2)**(1/3)
            
            coes['inc'] = inc
            coes['RAAN'] = RAAN
            coes['arg_p'] = arg_p
            coes['TA'] = TA
            coes['ecc'] = ecc
            coes['period'] = Period
            coes['semi_a'] = semi_a
         
    return coes

def time_since_peri_to_true_anomaly(time,h,ecc):
    mu = 398600
    Me = (mu**2/h**3)*time*(1-ecc**2)**(3/2)
    f = lambda x: x - ecc*sin(x) - Me
    
    if Me < pi:
        guess = Me + ecc
    else:
        guess = Me - ecc
        
    E = fsolve(f,guess,xtol = 1e-9)[0]
    
    angle = 2*arctan(tan(E/2)/sqrt((1-ecc)/(1+ecc)))
    angle = degrees(angle)
    return angle#%360

def true_anomaly_to_time_since_peri(TA,h,ecc):
    #TA = TA%360
    mu = 398600
    TA = radians(TA)
    E = 2*arctan(sqrt((1-ecc)/(1+ecc))*tan(TA/2))
    Me = E - ecc*sin(E)
    t0 = (Me*h**3/mu**2)/(1-ecc**2)**(3/2)
    return t0

def eci2lvlh(r,v):
    
    #X points up, radial, outwards
    #Y completes
    #Z r x v, h vector
    
    X = r/norm(r)
    Z = cross(r,v)/norm(cross(r,v))
    Y = cross(Z,X)/norm(cross(Z,X))
    
    C = np.vstack([X,Y,Z])
    
    return C

def h2semi(h,ecc,mu=398600):
    semi_a = h**2/(mu*(1 - ecc**2))
    return semi_a

def semi2period(semi,mu = 398600):
    period = 2*pi*sqrt(semi**3/mu)
    return period

def local_sidereal_time(date, east_longitude):
    #From Curtis page 261. Calculates the rotation angle between ecef and eci.
    #longitude is EAST

    day_date = datetime.datetime(year = date.year, month = date.month, day = date.day)
    UT = date.hour + date.minute/60 + date.second/3600

    J0 = julian_date(day_date)
    T0 = (J0 - 2451545)/36525
    theta_G0 = 100.4606184 + 36000.77004*T0 + 0.000387933*T0**2 + (2.583e-8)*T0**3 #Degrees
    theta_G = theta_G0 + 360.98564724*(UT/24)
    theta = (theta_G + east_longitude)%360

    return theta

def lla_to_ecef(lat, lon, alt, geodetic = False):
    #longitude is measured east of 0.
    #angles are in degrees
    #altitude is in meters
    earth_ecc = .081819221456
    earth_radius = 6378.1363 #km
    if geodetic:
        lat_gc = arctan((1 - earth_ecc**2) * tan(radians(lat))) #radians
        
        C_ellp = earth_radius/sqrt((1 - (earth_ecc**2)*sin(radians(lat))**2))
        S_ellp = earth_radius*(1 - earth_ecc**2)/sqrt(1 - (earth_ecc**2)*sin(radians(lat))**2)

        r_d = (C_ellp + alt/1000)*cos(radians(lat))
        r_k = (S_ellp + alt/1000)*sin(radians(lat))

        r_site = sqrt(r_d**2 + r_k**2)

    else:
        lat_gc = radians(lat) #radians
        r_site = earth_radius + alt/1000

    r_ecef = r_site*array([cos(lat_gc)*cos(radians(lon)),
                          cos(lat_gc)*sin(radians(lon)),
                          sin(lat_gc)])

    return r_ecef

def pass_az_el(gnd_pos, sat_pos):

    azimuth = []
    elevation = []
    if len(gnd_pos.shape) > 1:
        for gnd, sat in zip(gnd_pos, sat_pos):
            r_obs = sat - gnd
            u_gnd = gnd/norm(gnd)
            u_obs = r_obs/norm(r_obs)

            #[north,east,down]
            down = -u_gnd
            east = cross(array([0,0,1]),u_gnd)
            north = cross(east,down)

            u_obs_ned = array([north,east,down])@u_obs

            elevation.append(-arcsin(u_obs_ned[2]))
            azimuth.append(arctan2(u_obs_ned[1], u_obs_ned[0]))

        return degrees(hstack(azimuth)), degrees(hstack(elevation))
    else:
        r_obs = sat_pos - gnd_pos
        u_gnd = gnd_pos/norm(gnd_pos)
        u_obs = r_obs/norm(r_obs)

        #[north,east,down]
        down = -u_gnd
        east = cross(array([0,0,1]),u_gnd)
        north = cross(east,down)

        u_obs_ned = array([north,east,down])@u_obs

        elevation = -arcsin(u_obs_ned[2])
        azimuth = arctan2(u_obs_ned[1], u_obs_ned[0]) 

        return degrees(azimuth), degrees(elevation)



def observation_site(lat, lst, alt):
    f = .003353
    Re = 6378.1363

    rd = (Re/sqrt(1 - (2*f - f**2)*sin(radians(lat))**2) + alt/1000)*cos(radians(lat))
    rk = Re*((1 - f)**2)/sqrt(1 - (2*f - f**2)*sin(radians(lat))**2) + alt/1000

    return array([rd*cos(radians(lst)), rd*sin(radians(lst)), rk*sin(radians(lat))])

def razel2r(_range, az, el, lat, lst, alt):
    az = radians(az)
    el = radians(el)
    #lst = radians(lst)
    rho = array([cos(az)*cos(el), sin(az)*cos(el), -sin(el)])*_range

    site = observation_site(lat, lst, alt)

    down = -site/norm(site)
    east = cross(array([0,0,1]),site/norm(site))
    north = cross(east,down)
    NED2ECI = array([north,east,down]).T

    return site + NED2ECI@rho

def gauss_method(Ras, Decs, dates, lat, lon, alt, extended = False, tolerance = 1e-12, MU_EARTH = 398600):

    ER = 6378.1363
    TU = sqrt(ER**3/MU_EARTH)

    Ras = radians(Ras)
    Decs = radians(Decs)



    L1 = array([cos(Decs[0])*cos(Ras[0]), cos(Decs[0])*sin(Ras[0]), sin(Decs[0])])
    L2 = array([cos(Decs[1])*cos(Ras[1]), cos(Decs[1])*sin(Ras[1]), sin(Decs[1])])
    L3 = array([cos(Decs[2])*cos(Ras[2]), cos(Decs[2])*sin(Ras[2]), sin(Decs[2])])

    L = vstack([L1, L2, L3]).T


    

    lst1 = local_sidereal_time(dates[0], lon)
    site1 = observation_site(lat, lst1, alt)/ER

    lst2 = local_sidereal_time(dates[1], lon)
    site2 = observation_site(lat, lst2, alt)/ER

    lst3 = local_sidereal_time(dates[2], lon)
    site3 = observation_site(lat, lst3, alt)/ER

    rsites = vstack([site1, site2, site3]).T

    M = inv(L)@rsites
    M2 = M[:,1]

    JD1 = julian_date(dates[0])
    JD2 = julian_date(dates[1])
    JD3 = julian_date(dates[2])

    T1 = (JD1 - JD2)*24*3600/TU
    T3 = (JD3 - JD2)*24*3600/TU

    a1 = T3/(T3-T1)
    a1u = T3*((T3-T1)**2 - T3**2)/(6*(T3-T1))

    a3 = -T1/(T3-T1)
    a3u = -T1*((T3-T1)**2 - T1**2)/(6*(T3 - T1))


    d1 = M[1,0]*a1  - M[1,1] + M[1,2]*a3
    d2 = M[1,0]*a1u + M[1,2]*a3u



    C = dot(L2, rsites[:,1])

    func = lambda r2: r2**8 - (d1**2 + 2*C*d1 + norm(rsites[:,1])**2)*r2**6 - 2*(C*d2 + d1*d2)*r2**3 - d2**2


    r2_est = fsolve(func, 1e6, xtol = 1e-8)[0]

    #print('R2 est', r2_est)

    u = 1/(r2_est**3)


    c1 = a1 + a1u*u
    c2 = -1
    c3 = a3 + a3u*u
    cs = array([-c1, -c2, -c3])
    cp = M@cs

    slant1 = cp[0]/c1
    slant2 = cp[1]/c2
    slant3 = cp[2]/c3

    r1 = slant1*L1 + rsites[:,0]
    r2 = slant2*L2 + rsites[:,1]
    r3 = slant3*L3 + rsites[:,2]

    v2 = gibbs_orbit_determination(r1, r2, r3, mu = 1)

    #print(v2)

    if not extended:
        return r2*ER, v2*ER/TU
    else:
        error = 1
        num_iters = 0


        while error > tolerance:
            num_iters += 1
            
            R1 = norm(r1)
            R2 = norm(r2)
            R3 = norm(r3)
            
            dTA1 = arccos(dot(r1,r2)/R1/R2)
            dTA3 = arccos(dot(r3,r2)/R2/R3)


            # semi1 = R1*R2*(1 - cos(dTA1))/(R1 + R2 - 2*sqrt(R1*R2)*cos(dTA1/2)*cosh(dTA1)/2)
            # semi3 = R3*R2*(1 - cos(dTA3))/(R3 + R2 - 2*sqrt(R3*R2)*cos(dTA3/2)*cosh(dTA3)/2)

            # _, _, _, _, _, semi ,_,_, _, _,_, _ = AF.rv2coes(r2*ER, v2*ER/TU)

            # semi /= ER

            # semi1 = get_p(r1, r2, T1, mu = 1)
            # semi3 = get_p(r2, r3, -T3, mu = 1)


            X1 = universal_variable(r2, v2, T1, mu = 1)
            X3 = universal_variable(r2, v2, T3, mu = 1)

            alpha = (2/R2) - (norm(v2)**2)
            z1 = alpha*X1**2
            z3 = alpha*X3**2

            f1 = 1 - X1**2/R2*Stumpf_C(z1)
            f3 = 1 - X3**2/R2*Stumpf_C(z3)
            g1 = T1 - X1**3*Stumpf_S(z1)
            g3 = T3 - X3**3*Stumpf_S(z3)

            c1 = g3/(f1*g3 - f3*g1)
            c2 = -1
            c3 = -g1/(f1*g3 - f3*g1)

            cs = array([-c1, -c2, -c3])
            cp = M@cs

            slant1 = cp[0]/c1
            new_slant2 = cp[1]/c2
            slant3 = cp[2]/c3


            error = norm(new_slant2 - slant2)
            slant2 = new_slant2
            #print('err:',error)

        r1 = slant1*L1 + rsites[:,0]
        r2 = slant2*L2 + rsites[:,1]
        r3 = slant3*L3 + rsites[:,2]

        #print(vstack([r1,r2,r3]).T)

        v2 = gibbs_orbit_determination(r1, r2, r3, mu = 1)
        print('Extended gauss method executed in', num_iters, 'iterations.')

        return r1*ER, r2*ER, r3*ER, v2*ER/TU

def gibbs_orbit_determination(r1, r2, r3, mu = 398600):

    Z12 = cross(r1, r2)
    Z23 = cross(r2, r3)
    Z31 = cross(r3, r1)


    N = Z23*norm(r1) + Z31*norm(r2) + Z12*norm(r3)
    D = Z12 + Z23 + Z31
    S = (norm(r2) - norm(r3))*r1 + (norm(r3) - norm(r1))*r2 + (norm(r1) - norm(r2))*r3
    B = cross(D, r2)

    Lg = sqrt(mu/(norm(N)*norm(D)))

    v2 = Lg/norm(r2)*B + Lg*S

    return v2


def Stumpf_S(z):

    if z == 0:
        return 1/6
    elif z > 0:
        return (sqrt(z) - sin(sqrt(z)))*z**(-3/2)
    else:
        return (sinh(sqrt(-z)) - sqrt(-z))*(-z)**(-3/2)

def Stumpf_C(z):

    if z == 0:
        return .5
    elif z > 0:
        return (1 - cos(sqrt(z)))/z
    else:
        return -(cosh(sqrt(-z)) - 1)/z


def universal_variable(r, v, dt, mu = 398600):
    R = norm(r)
    V = norm(v)

    alpha = (2/R) - (V**2)/mu

    def keplers_equation(X):
        z = alpha*X**2
        vr = dot(r, v)/R

        return (R*vr)/sqrt(mu)*Stumpf_C(z)*X**2 + (1 - alpha*R)*Stumpf_S(z)*X**3 + R*X - sqrt(mu)*dt

    X_guess = sqrt(mu)*abs(alpha)*dt

    X = fsolve(keplers_equation, X_guess)[0]

    return X

def lambert_UV(r0, r, dt, short = True, mu = 398600, tol = 1e-8):
    R0 = norm(r0)
    R = norm(r)
    dTA = arccos(dot(r0,r)/R0/R)

    A = sin(dTA)*sqrt(R0*R/(1 - cos(dTA)))

    z = 0
    error = 1
    y = lambda _z: R0 + R + A*(_z*Stumpf_S(_z) - 1)/sqrt(Stumpf_C(_z))

    while abs(error) > tol:
        F = Stumpf_S(z)*(y(z)/Stumpf_C(z))**(3/2) + A*sqrt(y(z)) - sqrt(mu)*dt
        if z == 0:
            Fp = sqrt(2)/40*y(0)**(3/2) + A/8*(sqrt(y(0)) + A*sqrt(1/(2*y(0))))
        else:
            Fp = (y(z)/Stumpf_C(z))**(3/2)*( (1/2/z)*(Stumpf_C(z) - 3/2*Stumpf_S(z)/Stumpf_C(z) + 3/4*Stumpf_S(z)**2/Stumpf_C(z)))
            Fp += A/8*( 3*Stumpf_S(z)/Stumpf_C(z)*sqrt(y(z)) + A*sqrt(Stumpf_C(z)/y(z)) )

        znew = z - F/Fp
        error = z-znew
        z = znew

    f = 1 - y(z)/R
    g = A*sqrt(y(z)/mu)
    f_dot = sqrt(mu)/R0/R*sqrt(y(z)/Stumpf_C(z))*(z*Stumpf_S(z) - 1)
    g_dot = 1 - y(z)/R

    v0 = 1/g*(r - f*r0)
    v = 1/g*(g_dot*r - r0)

    return v0, v

def get_p(r0, r, dt, mu = 398600):
    R0 = norm(r0)
    R = norm(r)
    dTA = arccos(dot(r0,r)/R0/R)

    L = (R0 + R)/(4*sqrt(R0*R)*cos(dTA/2)) - .5
    M = mu*(dt**2)/((2*sqrt(R0*R)*cos(dTA/2))**3)

    y = 1
    error = 1
    while abs(error) > 1e-8:
        x1 = M/(y**2) - L
        x2 = (4/3)*(1 + 6*x1/5 + 6*8*(x1**2)/5/7 + 6*8*10*(x1**3)/5/7/9)

        ynew = 1 + x2*(L+x1)

        error = ynew - y
        y = ynew

    cosDE = 1 - 2*x1

    p = R0*R*(1 - cos(dTA))/(R0 + R - 2*sqrt(R0*R)*cos(dTA/2)*cosDE)

    return p

def Lambert_gaussian(r0, r, dt, short = True, mu = 398600):
    R0 = norm(r0)
    R = norm(r)
    dTA = arccos(dot(r0,r)/R0/R)

    p = get_p(r0, r, dt, mu = mu)

    f = 1 - R/p*(1 - cos(dTA))
    g = R*R0*sin(dTA)/sqrt(mu*p)
    f_dot = sqrt(1/p)*tan(dTA/2)*((1 - cos(dTA))/p - (1/R) - (1/R0))
    g_dot = 1 - R0/p*(1 - cos(dTA))

    v0 = (r - f*r0)/g
    v = (g_dot*r - r0)/g

    return v0, v

def Lamberts_min_energy(r0, r, mu = 398600):

    R0 = norm(r0)
    R = norm(r)

    cosdTA = dot(r0,r)/R0/R

    _c = sqrt(R0**2 + R**2 - 2*R0*R*cosdTA)
    _s = (R0 + R + _c)/2

    a_min = _s/2
    p_min = R0*R*(1 - cosdTA)/_c
    e_min = sqrt(1 - 2*p_min/_s)

    alpha = pi
    beta = 2*arcsin(sqrt((_s - _c)/_s))

    t_min = sqrt(a_min**3/mu)*(pi - beta + sin(beta))

    v0 = sqrt(mu*p_min)/(R0*R*sin(arccos(cosdTA)))*(r - (1 - R/p_min*(1 - cosdTA))*r0)
    v = -sqrt(mu*p_min)/(R0*R*sin(arccos(cosdTA)))*(r0 - (1 - R0/p_min*(1 - cosdTA))*r)

    return v0, v, t_min, a_min, e_min

if __name__ == '__main__':

    today = datetime.datetime(year = 2004, month = 3, day = 3, hour = 4, minute = 30)
    east_longitude = 139.8

    lst = local_sidereal_time(today, east_longitude)


    