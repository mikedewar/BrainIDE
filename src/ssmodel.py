"""An unforced state space model module: 	
 with Kalman filter & RTS smoother attributes"""

import numpy, pylab

class ssmodel:
	"""class defining a linear, gaussian, discrete-time state space model
	
	properties:
	
	A - state transition matrix
	C - observation matrix
	Sw - state noise covariance matrix
	Sv - observation noise covariance matrix
	T - number of time points
	time - list of time indices
	nx - number of states
	ny - number of outputs
	x0 - intial state
	X - state sequence
	K - Kalman matrix
	M - Cross covariance matrix sequence
	P - Covariance matrix sequence
	"""
	
	def __init__(self,A,C,Sw,Sv,T):

		"""State State model parameters"""
		self.A = numpy.matrix(A)
		self.C = numpy.matrix(C)
		self.Sw = numpy.matrix(Sw)
		self.Sv = numpy.matrix(Sv)
		self.T = T
		self.time = range(T)		
		self.ny,self.nx = self.C.shape

		# Dimension consistency
		assert self.A.shape == (self.nx,self.nx), 'ssmodel consistency check failed (A)'
		assert self.C.shape == (self.ny,self.nx), 'ssmodel consistency check failed (C)'
		assert self.Sw.shape == (self.nx,self.nx), 'ssmodel consistency check failed (Sw)'
		assert self.Sv.shape == (self.ny,self.ny), 'ssmodel consistency check failed (Sv)'
		
		
		self.x0 = numpy.matrix(numpy.zeros([self.nx,1]))
		self.X = zerosList(self.nx,1,T)
		self.Y = zerosList(self.ny,1,T)

		
		# filter quantities
		self.K = numpy.matrix(numpy.zeros([self.nx,self.ny]))
		self.M = zerosList(self.nx,self.nx,T)
		self.P = zerosList(self.nx,self.nx,T)
		# initial condition
		self.P0 = 40000* numpy.matrix(numpy.ones((self.nx,self.nx)))

	
	def simulate(self):
		"""simulates the state space model"""

		x = numpy.matrix(self.x0)
		x.shape = self.nx, 1

		if self.Sw.any()<>0: self.Swc = numpy.linalg.cholesky(self.Sw)  
		if self.Sv.any()<>0:self.Svc = numpy.linalg.cholesky(self.Sv) 

		for t in self.time:

			v = numpy.random.randn(self.ny,1)
			w = numpy.random.randn(self.nx,1)
			
			self.X[t] = x
			self.Y[t] = self.C*x + self.Svc*v
			x = self.A*x + self.Swc*w
	

	
	def kfilter(self):
		"""Kalman Filter

		xhatPredStore: Predicted state
		xhatSrore -Aanalysed state
		PPredStore:Predicted covariance
		PStore: Aanalysed covariance """

		# Predictor
		def Kpred(A,C,P,Sw,x):
			x = A*x 
			P = A*P*A.T + Sw
			return x,P

		# Corrector
		def Kupdate(A,C,P,Sv,x,z):
			K = (P*C.T) * (C*P*C.T + Sv).I
			x = x + K*(z-(C*x))
			P = (numpy.eye(self.nx)-K*C)*P;
			return x,P,K

		## initialise	
		xhat = self.x0
		nx = self.nx
		ny = self.ny
		T = self.T
		# filter quantities
		xhatPredStore = zerosList(nx,1,T)
		PPredStore = zerosList(nx,nx,T)
		xhatStore = zerosList(nx,1,T)
		PStore = zerosList(nx,nx,T)
		KStore = zerosList(nx,ny,T)
		# initialise the filter
		xhat,P = Kpred(self.A,self.C,self.P0,self.Sw,xhat)
		xhatPredStore[0], PPredStore[0] = xhat, P
		## filter
		for t in self.time:
			# update state estimate with measurement and store Kalman gain
			xhat, P, KStore[t] = Kupdate(self.A,self.C,P,self.Sv,xhat,self.Y[t])
			# store corrected values of xhat and P
			xhatStore[t], PStore[t] = xhat, P
			# predict new state
			xhat, P = Kpred(self.A,self.C,P,self.Sw,xhat);
			xhatPredStore[t],PPredStore[t] = xhat,P
						
		self.K = KStore
		self.X = xhatStore
		self.P = PStore	
	
	def rtssmooth(self):
		"""Rauch Tung Streibel(RTS) smoother"""
		
		# predictor
		def Kpred(A,C,P,Sw,x):
			assert type(x) is numpy.matrix
			assert x.shape == (self.nx,1), str(x.shape)
			x = A*x 
			P = A*P*A.T + Sw;
			return x,P
		
		# corrector
		def Kupdate(A,C,P,Sv,x,z):
			K = (P*C.T) * (C*P*C.T + Sv).I
			x = x + K*(z-(C*x))
			P = (numpy.eye(self.nx)-K*C)*P;
			return x,P,K
		
		## initialise	
		xhat = self.x0
		nx = self.nx
		ny = self.ny
		T = self.T
		# filter quantities
		xhatPredStore = zerosList(nx,1,T)
		PPredStore = zerosList(nx,nx,T)
		xhatStore = zerosList(nx,1,T)
		PStore = zerosList(nx,nx,T)
		KStore = zerosList(nx,ny,T)
		# smoother quantities
		xb = zerosList(nx,1,T)
		Pb = zerosList(nx,nx,T)
		M = zerosList(nx,nx,T)
		S = zerosList(nx,nx,T)
		# initialise the filter
		xhat,P = Kpred(self.A,self.C,self.P[0],self.Sw,xhat)
		xhatPredStore[0], PPredStore[0] = xhat, P
		## filter
		for t in self.time:
		    # store prediction values of xhat and P
		    xhatPredStore[t], PPredStore[t] = xhat, P
		    # update state estimate with measurement and store Kalman gain
		    xhat, P, KStore[t] = Kupdate(self.A,self.C,P,self.Sv,xhat,self.Y[t])
		    # store corrected values of xhat and P
		    xhatStore[t], PStore[t] = xhat, P
	    	# predict new state
		    xhat, P = Kpred(self.A,self.C,P,self.Sw,xhat);


		# initialise the smoother
		xb[-1], Pb[-1] = xhatStore[-1], PStore[-1]
		## smooth
		for t in range(T-2,0,-1):
			S[t] = PStore[t]*self.A.T * PPredStore[t+1].I
			xb[t] = xhatStore[t] + S[t]*(xb[t+1] - xhatPredStore[t])
			Pb[t] = PStore[t] + S[t] * (Pb[t+1] - PPredStore[t+1]) * S[t].T
		# iterate a final time to calucate the cross covariance matrices
 		M[-1]=(numpy.eye(nx)-KStore[-1]*self.C) * self.A*PStore[-2]
		for t in range(T-2,1,-1):
		    M[t]=PStore[t]*S[t-1].T + S[t]*(M[t+1] - self.A*PStore[t])*S[t-1].T
		
		self.K = KStore
		self.M = M
		self.X = xb
		self.P = Pb
		
def zerosList(rows,cols,listLength):
	"""returns a list of zero matrices"""
	return [numpy.matrix(numpy.zeros((rows,cols)))]*listLength
	

