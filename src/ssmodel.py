"""An unforced state space model module: 	
 with Kalman filter & RTS smoother attributes"""

import pylab as pb

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
		
		assert A.shape == (nx, nx), 'ssmodel consistency check failed (A)'
		assert C.shape == (ny, nx), 'ssmodel consistency check failed (C)'
		assert Sw.shape == (nx, nx), 'ssmodel consistency check failed (Sw)'
		assert Sv.shape == (ny, ny), 'ssmodel consistency check failed (Sv)'
		assert x0.shape == (nx,1)
		
		self.A = pb.matrix(A)
		self.C = pb.matrix(C)
		self.Sw = pb.matrix(Sw)
		self.Sv = pb.matrix(Sv)		
		self.ny = ny
		self.nx = nx
		self.x0 = x0
		
		# initial condition
		self.P0 = 40000* pb.matrix(pb.ones((self.nx,self.nx)))

	
	def simulate(self,T):
		"""simulates the state space model"""

		x = pb.matrix(self.x0)
		x.shape = self.nx, 1

		if self.Sw.any()<>0: self.Swc = pb.linalg.cholesky(self.Sw)  
		if self.Sv.any()<>0:self.Svc = pb.linalg.cholesky(self.Sv) 

		for t in range(T):

			v = pb.random.randn(self.ny,1)
			w = pb.random.randn(self.nx,1)
			
			self.X[t] = x
			self.Y[t] = self.C*x + self.Svc*v
			x = self.A*x + self.Swc*w
	
	def kfilter(self,Y):
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
		def Kupdate(A,C,P,Sv,x,y):
			K = (P*C.T) * (C*P*C.T + Sv).I
			x = x + K*(y-(C*x))
			P = (pb.eye(self.nx)-K*C)*P;
			return x,P,K

		## initialise	
		xhat = self.x0
		nx = self.nx
		ny = self.ny
		T = self.T
		# filter quantities
		xhatPredStore = []
		PPredStore = []
		xhatStore = []
		PStore = []
		KStore = []
		# initialise the filter
		xhat, P = Kpred(self.A, self.C, self.P0, self.Sw, xhat)
		xhatPredStore[0], PPredStore[0] = xhat, P
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
						
		self.K = KStore
		self.X = xhatStore
		self.P = PStore	
	
	def rtssmooth(self,Y):
		"""Rauch Tung Streibel(RTS) smoother"""
		
		# predictor
		def Kpred(A,C,P,Sw,x):
			assert type(x) is pb.matrix
			assert x.shape == (self.nx,1), str(x.shape)
			x = A*x 
			P = A*P*A.T + Sw;
			return x,P
		
		# corrector
		def Kupdate(A,C,P,Sv,x,y):
			K = (P*C.T) * (C*P*C.T + Sv).I
			x = x + K*(y-(C*x))
			P = (pb.eye(self.nx)-K*C)*P;
			return x,P,K
		
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
		xhatPredStore[0], PPredStore[0] = xhat, P
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
		# store
		self.K = KStore
		self.M = M
		self.X = xb
		self.P = Pb
	