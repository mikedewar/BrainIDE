import pylab as pb
from numpy import matlib
import sys
import logging
logging.basicConfig(stream=sys.stdout,level=logging.INFO)
log = logging.getLogger('ssmodel')

class ssmodel:
	"""class defining a linear, gaussian, discrete-time state space model.

	Arguments
	----------
	A : matrix
		State transition matrix.
	C : matrix
		Observation matrix.
	Sw : matrix
		State covariance matrix.
	Sv : matrix
		Observation noise covariance matrix
	x0: matrix
		Initial state vector
	
	Attributes
	----------
	A : matrix
		State transition matrix.
	C : matrix
		Observation matrix.
	Sw : matrix
		State covariance matrix.
	Sv : matrix
		Observation noise covariance matrix
	nx: int
		number of states
	ny: int 
		number of outputs
	x0: matrix
		intial state
	X: list of matrix
		state sequence
	K: list of matrix 
		Kalman gain sequence
	M: list of matrix
		Cross covariance matrix sequence
	P: list of matrix
		Covariance matrix sequence
	"""
	
	def __init__(self,A,C,Sw,Sv,x0):
		
		ny, nx = C.shape
		
		assert A.shape == (nx, nx)
		assert Sw.shape == (nx, nx)
		assert Sv.shape == (ny, ny)
		assert x0.shape == (nx,1)
		
		self.A = pb.matrix(A)
		self.C = pb.matrix(C)
		self.Sw = pb.matrix(Sw)
		self.Sv = pb.matrix(Sv)		
		self.ny = ny
		self.nx = nx
		self.x0 = x0
		
		# initial condition
		# TODO - not sure about this as a prior...
		self.P0 = 40000* pb.matrix(pb.ones((self.nx,self.nx)))
		
		log.info('initialised state space model')
	
	def transition_dist(self,x):
		mean = self.A*x
		return pb.multivariate_normal(mean.A.flatten(),self.Sw,[1]).T
	
	def observation_dist(self,x):
		mean = self.C*x
		return pb.multivariate_normal(mean.A.flatten(),self.Sv,[1]).T
		
	def gen(self,T):
		"""
		generator for the state space model
		
		Arguments
		----------
		T : int
			number of time points to generate

		Yields
		----------
		x : matrix
			next state vector
		y : matrix
			next observation
		"""	
		x = self.x0
		for t in range(T):
			y = self.observation_dist(x)
			yield x,y
			x = self.transition_dist(x)

	def simulate(self,T):
		"""
		simulates the state space model

		Arguments
		----------
		T : int
			number of time points to generate

		Returns
		----------
		X : list of matrix
			array of state vectors
		Y : list of matrix
			array of observation vectors
		"""
		
		log.info('sampling from the state space model')
		
		X = []
		Y = []
		for (x,y) in self.gen(T):
			X.append(x)
			Y.append(y)
		return X,Y
	
	
	def kfilter(self,Y):
		"""Vanilla implementation of the Kalman Filter
		
		Arguments
		----------
		Y : list of matrix
			A list of observation vectors
			
		Returns
		----------	
		X : list of matrix
			A list of state estimates
		P : list of matrix
			A list of state covariance matrices
		K : list of matrix
			A list of Kalman gains
		"""
		
		log.info('running the Kalman filter')
		
		# Predictor
		def Kpred(A,C,P,Sw,x):
			x = A*x 
			P = A*P*A.T + Sw
			return x,P

		# Corrector
		def Kupdate(A,C,P,Sv,x,y):
			K = (P*C.T) * (C*P*C.T + Sv).I
			x = x + K*(y-(C*x))
			P = (pb.eye(self.nx)-K*C)*P;
			return x,P,K

		## initialise	
		xhat = self.x0
		nx = self.nx
		ny = self.ny
		# filter quantities
		xhatPredStore = []
		PPredStore = []
		xhatStore = []
		PStore = []
		KStore = []
		# initialise the filter
		xhat, P = Kpred(self.A, self.C, self.P0, self.Sw, xhat)
		## filter
		for y in Y:
			# store
			xhatPredStore.append(xhat)
			PPredStore.append(P)
			# correct
			xhat, P, K = Kupdate(self.A,self.C,P,self.Sv,xhat,y)
			# store
			KStore.append(K)
			xhatStore.append(xhat) 
			PStore.append(P)
			# predict
			xhat, P = Kpred(self.A,self.C,P,self.Sw,xhat);
		
		return xhatStore, PStore, KStore
	
	def rtssmooth(self,Y):
		"""Rauch Tung Streibel(RTS) smoother
		
		Arguments
		----------
		Y : list of matrix
			A list of observation vectors
		
		Returns
		----------	
		X : list of matrix
			A list of state estimates
		P : list of matrix
			A list of state covariance matrices
		K : list of matrix
			A list of Kalman gains
		M : list of matrix
			A list of cross covariance matrices
		"""
		
		log.info('running the RTS Smoother')
		
		# predictor
		def Kpred(A,C,P,Sw,x):
			x = A*x 
			P = A*P*A.T + Sw;
			return x, P
		
		# corrector
		def Kupdate(A,C,P,Sv,x,y):
			K = (P*C.T) * (C*P*C.T + Sv).I
			x = x + K*(y-(C*x))
			P = (pb.eye(self.nx)-K*C)*P;
			return x, P, K
		
		## initialise	
		xhat = self.x0
		nx = self.nx
		ny = self.ny
		T = len(Y)
		# filter quantities
		xhatPredStore = []
		PPredStore = []
		xhatStore = []
		PStore = []
		KStore = []
		# initialise the filter
		xhat, P = Kpred(self.A, self.C, self.P0, self.Sw, xhat)
		## filter
		for y in Y:
			# store
			xhatPredStore.append(xhat)
			PPredStore.append(P)
			# correct
			xhat, P, K = Kupdate(self.A,self.C,P,self.Sv,xhat,y)
			# store
			KStore.append(K)
			xhatStore.append(xhat) 
			PStore.append(P)
			# predict
			xhat, P = Kpred(self.A,self.C,P,self.Sw,xhat);
		# initialise the smoother
		xb = [None]*T
		Pb = [None]*T
		S = [None]*T
		xb[-1], Pb[-1] = xhatStore[-1], PStore[-1]
		## smooth
		for t in range(T-2,0,-1):
			S[t] = PStore[t]*self.A.T * PPredStore[t+1].I
			xb[t] = xhatStore[t] + S[t]*(xb[t+1] - xhatPredStore[t])
			Pb[t] = PStore[t] + S[t] * (Pb[t+1] - PPredStore[t+1]) * S[t].T
		# finalise
		xb[0] = xhatStore[0]
		Pb[0] = PStore[0]
		# iterate a final time to calucate the cross covariance matrices
 		M = [None]*T
		M[-1]=(pb.eye(nx)-KStore[-1]*self.C) * self.A*PStore[-2]
		for t in range(T-2,1,-1):
		    M[t]=PStore[t]*S[t-1].T + S[t]*(M[t+1] - self.A*PStore[t])*S[t-1].T
		M[1] = matlib.eye(self.nx)
		M[0] = matlib.eye(self.nx)
		
		return xb, Pb, KStore, M


if __name__ == "__main__":
	import os
	os.system('py.test')