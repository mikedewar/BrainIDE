import pylab as pb
import sys
sys.path.append("../src")
import ssmodel

def setup():
	A = pb.matrix([[0.8,-0.4],[1,0]])
	C = pb.matrix([[1,0],[0,1],[1,1]])
	Q = 2.3*pb.matrix(pb.eye(2))
	R = 0.2*pb.matrix(pb.eye(3))
	x0 = pb.matrix([[1],[1]])
	return ssmodel.ssmodel(A,C,Q,R,x0)

def within_dist(mean, covariance, test_point):
	"""	tests to see if a test point is within the 99% confidence interval of 
	the normal distribution specified by its mean and covariance
	
	Arguments
	----------
	mean : matrix
		Mean of the normal distribution
	covariance : matrix
		Covariance matrix of the normal distribution
	test_point : matrix
		The sample to be tested
	
	Notes
	----------
	Note that this function doesn't really properly test to see if the point
	is within a multivariate distribution, it simply tests to see if each
	element of the test point is within the univariate Gaussian defined by the
	diagonal elements of the covariance matrix.
	
	TODO create a proper test (therefore make this within the 95% interval)
	"""
	err = abs(test_point - mean).A.flatten()
	variances = pb.array(covariance.diagonal()).flatten()
	for (e,variance) in zip(err,variances):
		std = pb.sqrt(variance)
		if e > 3*std:
			return False
	return True

def test_kfilter():
	model = setup()
	T = 100
	Xo, Y = model.simulate(T)
	X, P, K = model.kfilter(Y)
	for (x, xo, p) in zip(X,Xo,P):
		assert within_dist(xo,p,x)

def test_rtssmoother():
	model = setup()
	T = 100
	Xo, Y = model.simulate(T)
	X, P, K, M = model.rtssmooth(Y)
	for (x, xo, p) in zip(X,Xo,P):
		assert within_dist(xo,p,x)

if __name__ == "__main__":
	import os
	os.system('py.test')