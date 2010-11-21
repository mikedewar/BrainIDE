#Author Parham Aram
#Date 21-11-2010
from __future__ import division
import pylab as pb
import numpy as np
import scipy as sp
class ukf():

	"""class defining the Unscented Kalman filter for nonlinear estimation.

	Arguments
	----------
	nx: int
		the dimension of the states
	x0: ndarray
		initial state
	P0: ndarray
		initial state covariance matrix
	C:  ndarray
		observation matrix
	Sigma_e: ndarray
		process noise covariance matrix
	Sigma_varepsilon: ndarray
		observation noise covariance matrix
	state_equation: function
		non-linear state equation
	kappa: float
		secondary scaling parameter, see A New Approach for Filtering Nonlinear Systems by Julier.
	alpha_sigma_points: 
		Determines the spread of sigma points around xbar.
	beta_sigma_points:
		is used to incorporate prior knowledge of the distribution of x
		beta=2 is optimal for Gaussian distribution
	

	Attributes
	----------
	lambda: float
		scaling parameter
	gamma_sigma_points: float
		composite scaling parameter

	"""

	def __init__(self,nx,x0,P0,C,Sigma_e,Sigma_varepsilon,state_equation,kappa=0.0,alpha_sigma_points=1e-3,beta_sigma_points=2):


		assert 	type(x0) is np.ndarray, 'Initial state must be ndarray'
		assert 	type(P0) is np.ndarray, 'Initial state covariance matrix must be ndarray'
		assert 	type(Sigma_e) is np.ndarray, 'Process noise covariance matrix must be ndarray'
		assert 	type(Sigma_varepsilon) is np.ndarray, 'Observation noise covariance matrix must be ndarray'



		self.nx=nx
		self.x0=x0
		self.P0=P0
		self.C=C
		self.Sigma_e=Sigma_e
		self.Sigma_varepsilon =Sigma_varepsilon 
		self.state_equation=state_equation
		self.alpha_sigma_points=alpha_sigma_points
		self.beta_sigma_points=beta_sigma_points
		self.kappa=3-self.nx
		self.lamda=self.alpha_sigma_points**2*(self.nx+self.kappa)-self.nx    
		self.gamma_sigma_points=pb.sqrt(self.nx+self.lamda) 





	def sigma_vectors(self,x,P):

		"""
		generator for the sigma vectors

		Arguments
		----------
		x : ndarray
			state at time instant t
		P:  ndarray
			state covariance matrix at time instant t

		Returns
		----------
		Xi : ndarray
			matrix of sigma points, each column is a sigma vector: [x0 x0+ x0-];nx by 2nx+1
		"""
		Pc=sp.linalg.cholesky(P,lower=1)
		Weighted_Pc=self.gamma_sigma_points*Pc		
		Xi_plus=[]
		Xi_minus=[]
		for i in range(self.nx):
			Xi_plus.append(x+Weighted_Pc[:,i].reshape(self.nx,1)) #list of ndarray with length nx
			Xi_minus.append(x-Weighted_Pc[:,i].reshape(self.nx,1)) #list of ndarray with length nx

		Xi=pb.hstack((x,pb.hstack((pb.hstack(Xi_plus),pb.hstack(Xi_minus))))) 
		return Xi



	def sigma_vectors_weights(self):

		"""
		generator for the sigma vectors' weights

		Returns
		----------
		Wm_i : ndarray
			array of sigma points' weights 
		Wc_i : ndarray
			array of sigma points' weights 
		"""
		Wm0=[self.lamda/(self.lamda+self.nx)]
		Wc0=[(self.lamda/(self.lamda+self.nx))+1-self.alpha_sigma_points**2+self.beta_sigma_points]
		Wmc=[1./(2*(self.nx+self.lamda))]
		Wm_i=pb.concatenate((Wm0,2*self.nx*Wmc)) #ndarray 2n_x+1
		Wc_i=pb.concatenate((Wc0,2*self.nx*Wmc)) #ndarray 2n_x+1
		return Wm_i,Wc_i

	def _filter(self,Y):

		## initialise
		xf=self.x0
		Pf=self.P0
		# filter quantities
		xfStore =[]
		PfStore=[]

		#calculate the weights
		Wm_i,Wc_i=self.sigma_vectors_weights()

		for y in Y:
			#calculate the sigma points matrix, each column is a sigma vector
			Xi_f_=self.sigma_vectors(xf,Pf)
			#propogate sigma verctors through non-linearity
			Xi_f=self.state_equation(Xi_f_)
			#pointwise multiply by weights and sum along y-axis
			xf_=pb.sum(Wm_i*Xi_f,1)
			xf_=xf_.reshape(self.nx,1)
			#purturbation
			Xi_purturbation=Xi_f-xf_
			weighted_Xi_purturbation=Wc_i*Xi_purturbation
			Pf_=pb.dot(Xi_purturbation,weighted_Xi_purturbation.T)+self.Sigma_e			
			#measurement update equation
			Pyy=dots(self.C,Pf_,self.C.T)+self.Sigma_varepsilon 
			Pxy=pb.dot(Pf_,self.C.T)
			K=pb.dot(Pxy,pb.inv(Pyy))
			yf_=pb.dot(self.C,xf_)
			xf=xf_+pb.dot(K,(y-yf_))
			Pf=pb.dot((pb.eye(self.nx)-pb.dot(K,self.C)),Pf_)
			xfStore.append(xf)
			PfStore.append(Pf)

		return xfStore,PfStore
		

	def rtssmooth(self,Y):
		## initialise
		xf=self.x0
		Pf=self.P0
		# filter quantities
		xfStore =[]
		PfStore=[]


		#calculate the weights
		Wm_i,Wc_i=self.sigma_vectors_weights()



		for y in Y:

			#calculate the sigma points matrix, each column is a sigma vector
			Xi_f_=self.sigma_vectors(xf,Pf)
			#propogate sigma verctors through non-linearity
			Xi_f=self.state_equation(Xi_f_)
			#pointwise multiply by weights and sum along y-axis
			xf_=pb.sum(Wm_i*Xi_f,1)
			xf_=xf_.reshape(self.nx,1)
			#purturbation
			Xi_purturbation=Xi_f-xf_
			weighted_Xi_purturbation=Wc_i*Xi_purturbation
			Pf_=pb.dot(Xi_purturbation,weighted_Xi_purturbation.T)+self.Sigma_e			
			#measurement update equation
			Pyy=dots(self.C,Pf_,self.C.T)+self.Sigma_varepsilon 
			Pxy=pb.dot(Pf_,self.C.T)
			K=pb.dot(Pxy,pb.inv(Pyy))
			yf_=pb.dot(self.C,xf_)
			xf=xf_+pb.dot(K,(y-yf_))
			Pf=pb.dot((pb.eye(self.nx)-pb.dot(K,self.C)),Pf_)
			xfStore.append(xf)
			PfStore.append(Pf)



		# initialise the smoother

		T=len(Y)
		xb = [None]*T
		Pb = [None]*T
	

		xb[-1], Pb[-1] = xfStore[-1], PfStore[-1]

		# backward iteration

		for t in range(T-2,-1,-1):
			#calculate the sigma points matrix from filterd states, each column is a sigma vector
			Xi_b_=self.sigma_vectors(xfStore[t],PfStore[t]) 
			#propogate sigma verctors through non-linearity
			Xi_b=self.state_equation(Xi_b_) 
			#calculate xb_
			#pointwise multiply by weights and sum along y-axis
			xb_=pb.sum(Wm_i*Xi_b,1)
			xb_=xb_.reshape(self.nx,1)
			#purturbation
			Xi_b__purturbation=Xi_b_-xfStore[t] 
			Xi_b_purturbation=Xi_b-xb_ 
			#weighting
			weighted_Xi_b__purturbation=Wc_i*Xi_b__purturbation 
			weighted_Xi_b_purturbation=Wc_i*Xi_b_purturbation
			Pb_=pb.dot(Xi_b_purturbation,weighted_Xi_b_purturbation.T)+self.Sigma_e
			Mb_=pb.dot(weighted_Xi_b__purturbation,Xi_b_purturbation.T)

			#Calculate Smoother outputs
			S=pb.dot(Mb_,pb.inv(Pb_))
			xb[t]=xfStore[t]+pb.dot(S,(xb[t+1]-xb_))
			Pb[t]=PfStore[t]+dots(S,(Pb[t+1]-Pb_),S.T)


		return xb,Pb




def dots(*args):
	lastItem = 1.
	for arg in args:
		lastItem = pb.dot(lastItem, arg)
	return lastItem
