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

def test_kfilter():
	model = setup()
	T = 100
	Xo, Y = model.simulate(T)
	X, P, K = model.kfilter(Y)

def test_rtssmoother():
	model = setup()
	T = 100
	Xo, Y = model.simulate(T)
	X, P, K, M = model.rtssmooth(Y)

if __name__ == "__main__":
	import os
	os.system('py.test')