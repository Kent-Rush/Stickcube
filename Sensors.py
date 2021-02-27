from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from numpy.random import normal


class RateGyro():

	def __init__(self, bias_std, ecrw_std, white_noise_std):

		self.bias = normal(0, bias_std, 3)
		self.ecrw = array([0.0,0.0,0.0])
		self.ecrw_std = ecrw_std
		self.white_noise_std = white_noise_std


	def random_walk(self):
		self.ecrw += normal(0, self.ecrw_std, size=3)

	def measure(self, x):

		self.random_walk()

		return x + self.bias + self.ecrw + normal(0, self.white_noise_std, size=3)


class Accelerometer():

	def __init__(self, bias_std, white_noise_std):

		self.bias = normal(0, bias_std, 3)
		self.white_noise_std = white_noise_std

	def measure(self, x):

		return x + self.bias + normal(0, self.white_noise_std, size=3)

class Magnetometer():

	def __init__(self, white_noise_std):

		self.white_noise_std = white_noise_std

	def measure(self, x):

		return x + normal(0, self.white_noise_std, size=3)


if __name__ == '__main__':

	x = linspace(0,1000,10000)
	y = sin(x)

	bias_std = 0.1
	ecrw_std = .01
	white_noise_std = .05

	gyro = RateGyro(bias_std, ecrw_std, white_noise_std)

	m = array([gyro.measure(array([a, a, a]))  for a in y  ]) 


	plt.plot(x, m)
	plt.show()