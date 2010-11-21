#Author Parham Aram
#Date 21-11-2010
'''estimate the connectivity kernel parameters and synaptic dynamics using least squares method'''
from __future__ import division
import pylab as pb
import numpy as np
from scipy import signal


class para_state_estimation():

	def __init__(self,model):

		self.model=model


	def Q_calc(self,X):

		"""
			calculates Q (n_x by n_theta) matrix of the IDE model at  each time step
	
			Arguments
			----------
			X: list of ndarray
				state vectors

			Returns
			---------
			Q : list of ndarray (n_x by n_theta)
		"""

		Q=[]	
		T=len(X)
		Psi=self.model.Gamma_inv_psi_conv_Phi
		Psi_T=pb.transpose(self.model.Gamma_inv_psi_conv_Phi,(0,2,1))

		for t in range(T):

			firing_rate_temp=pb.dot(X[t].T,self.model.Phi_values)
			firing_rate=self.model.act_fun.fmax/(1.+pb.exp(self.model.act_fun.varsigma*(self.model.act_fun.v0-firing_rate_temp)))	

			#calculate q
			g=pb.dot(firing_rate,Psi_T)

			g *=(self.model.spacestep**2)	
			q=self.model.Ts*g
			q=q.reshape(self.model.nx,self.model.n_theta)
			Q.append(q)
		return Q


	def estimate_kernel(self,X):

		"""
			estimate the ide parameters using least squares method
	
			Arguments
			----------
			X: list of ndarray
				state vectors

			Returns
			---------
			least squares estimation of the IDE parameters
		"""

		Q=self.Q_calc(X)
		Z=pb.vstack(X[1:])
		X_t_1=pb.vstack(X[:-1])
		Q_t_1=pb.vstack(Q[:-1])
		X_ls=pb.hstack((Q_t_1,X_t_1))
		theta=dots(pb.inv(pb.dot(X_ls.T,X_ls)),X_ls.T,Z)
		parameters=[float(theta[i]) for i in range(theta.shape[0])]
		return parameters



def dots(*args):
	lastItem = 1.
	for arg in args:
		lastItem = pb.dot(lastItem, arg)
	return lastItem
