from __future__ import division
import pylab as pb


def freq_analysis(vc,rho,gamma):

	'''
	Arguments:
	----------
		vc:cut of frequency shouldn't be in vector form
		rho: oversampling parameter rho>=1
		gamma: amount of attenuation at cut of frequency for 3db attenuation it should be set to 0.5	'''

	vc_vector=pb.array([[vc],[vc]]) 
	sigma_v2=-(2*pb.dot(vc_vector.T,vc_vector))/(pb.log(gamma))
	sigma_s2=1./((pb.pi**2)*sigma_v2)
	delta=1./(rho*(2*vc))
	#------------------------------------------
	#Sanner
	#-----------------------------
	sigma_san_v2=(vc**2)*pb.pi
	delta_san=pb.sqrt(1./(8*sigma_san_v2))
	sigma_san_s2=1./((pb.pi**2)*sigma_san_v2)
	#-----------------------------------------
	#print results
	#-----------------------------------------
	print '----------------------------------'
	print 'sigma_v2=',float(sigma_v2)
	print 'sigma_s2=',float(sigma_s2)
	print 'delta=',float(delta)
	print'----------------------------------'
	print 'results from Sanner analysis:'
	print 'sigma_v2=',float(sigma_san_v2)
	print 'sigma_s2=',float(sigma_san_s2)
	print 'delta=',float(delta_san)
