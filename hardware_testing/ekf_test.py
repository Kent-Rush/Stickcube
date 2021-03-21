import serial
import time
import numpy as np

def crux(A):
    return np.array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1], A[0], 0]])

def getF(p,w,b, dt):

	W = w - b

	dpdot_dp = 0.5*(np.outer(p,W) - np.outer(W,p) - crux(W) + (np.dot(W,p))*np.identity(3))
	dpdot_db = 0.5*(0.5*(1 - np.dot(p,p))*np.identity(3) + crux(p) + np.outer(p,p))

	dpdot_dx = np.hstack([dpdot_dp, dpdot_db])
	df_dx = np.vstack([np.zeros((3,6)), dpdot_dx])

	return np.identity(6) + dt*df_dx

def mrp2dcm(p):

	denom = (1+np.dot(p,p))**2

	if np.linalg.norm(p) > 1e-12:
		return np.identity(3) - ((4*(1-np.dot(p,p)))/denom)*crux(p) + 8/denom*crux(p)@crux(p)
	else:
		return np.identity(3)
def getH(p,m_abs):

	denom = (1+np.dot(p,p))**2
	A = mrp2dcm(p)
	L = (4/denom)*(A@crux(m_abs))*( (1 - np.dot(p,p))*np.identity(3) -2*crux(p) + 2*np.outer(p,p) )
	return np.hstack([np.zeros((3,3)), L])

def getG(p):

	dpdot_dn = 0.5*( 0.5*(1-np.dot(p,p))*np.identity(3) + crux(p) + np.outer(p,p) )
	dbdot_dn = np.identity(3)
	g12 = np.zeros((3,3))

	return np.vstack([np.hstack([dbdot_dn, g12       ]),
					  np.hstack([g12     , dpdot_dn])])

def f(x, w):

	b = x[0:3]
	p = x[3:6]

	dpdt = 0.5*( 0.5*(1-np.dot(p,p))*np.identity(3) + crux(p) + np.outer(p,p) )@(w-b)
	return np.hstack([0,0,0,dpdt])

def h(X, m_abs):
	p = X[3:6]
	return mrp2dcm(p)@m_abs

if __name__ == "__main__":
	arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)

	deviation = 1e-6
	dt = .025
	bias_0 = np.array([0,0,0])
	mrp_0 = np.array([0.01,0.01,0.01])
	X = np.hstack([bias_0, mrp_0])
	P = np.identity(6)*1e-2
	K = np.zeros((6,6))

	for ii in range(100):
		data = arduino.readline()
	data_str = data[:-2].decode("utf-8")
	m_abs = np.asarray([float(x) for x in data_str.split()[0:3]])

	prev_measurements = np.asarray([float(x) for x in data_str.split()])

	print('m_abs',m_abs)
	n = 100
	samples = np.zeros((n,3))
	for ii in range(n):
		data = arduino.readline()
		data_str = data[:-2].decode("utf-8")
		m = np.asarray([float(x) for x in data_str.split()[0:3]])
		samples[ii,:] = m
	stds = np.std(samples,axis=0)
	R = np.diag(stds)
	print(R)

	while True:
		data = arduino.readline()
		data_str = data[:-2].decode("utf-8")
		measurements = np.asarray([float(x) for x in data_str.split()])

		z = measurements[0:3]
		w = measurements[3:6]/180*np.pi
		b = X[0:3]
		p = X[3:6]

		F = getF(p,w,b, dt)
		H = getH(p, m_abs)
		G = getG(p)
		Q= G@G.T*deviation**2

		X_pred = X.copy()
		for ii in range(20):
			X_pred = X_pred + f(X, w)*dt/20

		P_pred = F@P@F.T + Q
		S = H@P@H.T + R
		K = P_pred@H.T@np.linalg.inv(S)

		y = z - h(X_pred,m_abs)
		X = X_pred + K@y
		P = (np.identity(6) - K@H)@P_pred

		if np.linalg.norm(X[3:6]) > 10:
			X[3:6] = -X[3:6]/np.linalg.norm(X[3:6])**2

		for ii, xx in enumerate(X[0:3]):
			if abs(xx) > 10:
				X[ii] = 10*np.sign(xx)

		# print('F')
		# print(F)
		# print('H')
		# print(H)
		# print('G')
		# print(G)
		# print('Q')
		# print(Q)
		# print('K')
		# print(K)
		# print('P')
		# print(P)
		# print('S')
		# print(S)
		# print('y')
		# print(y)
		# print('X')
		#print(X)
		print(['{:+.3f}'.format(ii) for ii in X])
