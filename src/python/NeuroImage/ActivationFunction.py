#Author Parham Aram
#Date 21-11-2010
from __future__ import division
import pylab as pb

class ActivationFunction():

	"""class defining the sigmoidal activation function .

	Arguments
	----------
	v0: float
		firing threshold, mV                 
	fmax: float
		maximum firing rate, spikes/s         
	varsigma: float
		slope of sigmoid, spikes/mV          

	v: float
		Presynaptic potential


	Attributes:
	-----------
		plot:
			plots the sigmoidal activation function

	Returns
	----------
	average firing rate

	"""	

	def __init__(self,v0,fmax,varsigma):
		
		self.v0 = v0
		self.fmax = fmax
		self.varsigma = varsigma
		
	def __call__(self,v):

		return float(self.fmax/(1.+pb.exp(self.varsigma*(self.v0-v))))


	def plot(self,plot_range):
		u=pb.linspace(-plot_range,plot_range,1000)
		z=pb.zeros_like(u)
		for i,j in enumerate(u):
			z[i]=self(j)
		pb.plot(u,z)
		pb.show()


