from __future__ import division
import pylab as pb
import numpy as np

class Basis():
	'''This is a class to define Gaussian basis functions'''

	def __init__(self,center,width,dimension,constant=1):
		'''
		Arguments
		---------
			center: matrix dimension X 1
				center of the gaussian	
			width: covariance of the gaussian sigma^2
				float
			dimension: int
					
			constant: float
				must be one, this is just to calculate the covariance of two gaussian basis functions'''

		self.center=pb.matrix(center)
		self.width=width
		self.dimension=dimension
		self.constant=constant
		assert type(self.center) is pb.matrix, 'center must be a matrix'
		assert self.center.shape==(self.dimension,1), 'center shape mismatch'
		assert type(width) is float,'width must be float type'

	def __call__(self,s):

		assert type(s) is pb.matrix, 'input must be a matrix'
		assert s.shape==(self.dimension,1), 'input shape mismatch'
		return float(self.constant*pb.exp(-1/(self.width)*(s-self.center).T*(s-self.center)))

	def __repr__(self):
		return 'Gaussian basis function;'+'center='+str(self.center) +',width='+str(self.width) 

	def conv(self,In):

		'''calculates the convolution of two gaussian basis functions
			Argument
			--------
			In: gaussian basis function
			Returns
			--------
			convolution of two gaussian basis functions'''

		assert In.__class__ is basis, 'The input must be a Gaussian basis function'
		
		convolution_weight=((pb.pi*self.width*In.width)/(self.width+In.width))**(self.dimension*0.5)
		convolution_width=self.width+In.width
		convolution_center=self.center+In.center
		return Basis(convolution_center,convolution_width,self.dimension,constant=convolution_weight)

	def FT(self):

		#assert self.center==pb.matrix(pb.zeros((self.dimension,1))),'center must be at zero'
		FT_weight=(pb.pi*self.width)**(self.dimension*0.5)
		FT_width=1./(((pb.pi)**2)*self.width)
		return Basis(self.center,FT_width,self.dimension,constant=FT_weight)
		
		


	def __mul__(self,In):

		'''calculates the Inner product of two gaussian basis functions

			Argument
			--------
			In: gaussian basis function
			Returns
			--------
			Inner product of two gaussian basis functions'''

		assert In.__class__ is basis, 'The input must be a Gaussian basis function'
		cij=(self.width+In.width)/(self.width*In.width)
		rij=(self.center-In.center).T*(1/(self.width+In.width))*(self.center-In.center)
		return float(self.constant*In.constant*(pb.pi)**(self.dimension*0.5)*(cij**(-0.5*self.dimension))*pb.exp(-rij))

#main=0
if __name__=="__main__":
#if main:

	from test import gen_spatial_lattice
	import time
	import scipy as sp
	#simulations properties
	dimension=2
	spacestep=0.5
	field_width=40
	field_space=pb.arange(-field_width/2.,(field_width/2.)+spacestep,spacestep)


	#field basis functions
	#field basis functions'centers
	S=pb.linspace(-10,10,9)
	field_centers=gen_spatial_lattice(S)
	#field basis functions'widths
	phi_widths=2.5
	#place field basis functions in an array in the form of n_x*1
	Phi=pb.array([Basis(cen,phi_widths,dimension) for cen in field_centers],ndmin=2).T
	nx=len(Phi)
	total_time=time.time()
	t_Gamma=time.time()
	#calculate Gamma=PhixPhi.T; inner product of the field basis functions
	Gamma=pb.matrix(Phi*Phi.T,dtype=float)
	Gamma_inv=Gamma.I
	print 'Elapsed time in seconds to calculate Gamma inverse is',time.time()-t_Gamma

	#Define kernel basis functions
	#Kernel basis functions' centers
	psi1_center=pb.matrix([[0],[0]])
	psi2_center=pb.matrix([[0],[0]])
	psi3_center=pb.matrix([[0],[0]])
	#Kernel basis functions' widths
	psi1_width=1.8**2
	psi2_width=2.4**2
	psi3_width=6.**2
	#Kernel basis functions' widths
	psi1_weight=10
	psi2_weight=-8
	psi3_weight=0.5
	psi1=Basis(psi1_center,psi1_width,dimension)
	psi2=Basis(psi2_center,psi2_width,dimension)
	psi3=Basis(psi3_center,psi3_width,dimension)
	#vectorizing convolution function of the kernel basis functions
	psi1_convolution_vectorized=pb.vectorize(psi1.conv) 	
	psi2_convolution_vectorized=pb.vectorize(psi2.conv) 	
	psi3_convolution_vectorized=pb.vectorize(psi3.conv) 	
	#convolving each kernel basis functions with the field basis functions
	psi1_conv_Phi=psi1_convolution_vectorized(Phi) #[psi1*phi1 psi1*phi2 ... psi1*phin].T
	psi2_conv_Phi=psi2_convolution_vectorized(Phi) #[psi2*phi1 psi2*phi2 ... psi2*phin].T
	psi3_conv_Phi=psi3_convolution_vectorized(Phi) #[psi3*phi1 psi3*phi2 ... psi3*phin].T

	#place convolution of the kernel basis functions with field basis functions in a matrix of (n_x X n_theta) dimension
	#[psi1*phi1 psi2*phi1 psi3*phi1;psi1*phi2 psi2*phi2 psi3*phi2;... psi1*phin psi2*phin psi3*phin]
	Psi_conv_Phi=pb.hstack((psi1_conv_Phi,psi2_conv_Phi,psi3_conv_Phi))

	##Finding the convolution of the kernel basis functions with field basis functions at discritized spatial points
	psi1_conv_Phi_values=[]
	psi2_conv_Phi_values=[]
	psi3_conv_Phi_values=[]
	Psi_conv_Phi_values=[]
	Phi_values=[]

	psi1_conv_Phi_values_temp=[]
	psi2_conv_Phi_values_temp=[]
	psi3_conv_Phi_values_temp=[]
	Psi_conv_Phi_values_temp=[]
	Phi_values_temp=[]

	t_Gamma_inv_Psi=time.time()
	for m in range(nx):
		for i in field_space:
			for j in field_space:
				psi1_conv_Phi_values_temp.append(psi1_conv_Phi[m,0](pb.matrix([[i],[j]])))
				psi2_conv_Phi_values_temp.append(psi2_conv_Phi[m,0](pb.matrix([[i],[j]])))
				psi3_conv_Phi_values_temp.append(psi3_conv_Phi[m,0](pb.matrix([[i],[j]])))
				Psi_conv_Phi_values_temp.append(psi1_weight*pb.array(psi1_conv_Phi_values_temp[-1])+ \
				psi2_weight*pb.array(psi2_conv_Phi_values_temp[-1])+psi3_weight*pb.array(psi3_conv_Phi_values_temp[-1]))


				Phi_values_temp.append(Phi[m,0](pb.matrix([[i],[j]])))


		psi1_conv_Phi_values.append(psi1_conv_Phi_values_temp)
		psi2_conv_Phi_values.append(psi2_conv_Phi_values_temp)
		psi3_conv_Phi_values.append(psi3_conv_Phi_values_temp)
		Psi_conv_Phi_values.append(Psi_conv_Phi_values_temp)
		Phi_values.append(Phi_values_temp)

		psi1_conv_Phi_values_temp=[]
		psi2_conv_Phi_values_temp=[]
		psi3_conv_Phi_values_temp=[]
		Psi_conv_Phi_values_temp=[]
		Phi_values_temp=[]

	Phi_values_array=pb.array(Phi_values,ndmin=3).T

	#Here psi1_conv_Phi_values is a list but behaves as a matrix of the form of [[psi1*phi1(s0) psi1*phi1(s1) ... psi1*phi1(sn)];[psi1*phi2(s0) psi1*phi2(s1) ... psi1*phi2(sn)]...;[psi1*phin(s0) psi1*phin(s1) ... psi1*phin(sn)]]
	Gamma_inv_psi1_conv_Phi=Gamma_inv*psi1_conv_Phi_values #matrix in a form of n_x X number of spatiol points
	Gamma_inv_psi2_conv_Phi=Gamma_inv*psi2_conv_Phi_values #matrix in a form of n_x X number of spatiol points
	Gamma_inv_psi3_conv_Phi=Gamma_inv*psi3_conv_Phi_values #matrix in a form of n_x X number of spatiol points
	Gamma_inv_Psi_conv_Phi=Gamma_inv*Psi_conv_Phi_values   #matrix in a form of n_x X number of spatiol points
	#or we can write 
	#Gamma_inv_Psi_conv_Phi_test=psi1_weight*Gamma_inv_psi1_conv_Phi+ \
	#psi2_weight*Gamma_inv_psi2_conv_Phi+psi3_weight*Gamma_inv_psi3_conv_Phi
	print 'Elapsed time in seconds to calculate the convolution and their products with Gamma is', time.time()-t_Gamma_inv_Psi



	#Define observation locations
	field_width=40;                        # -20 to 20 mm, twice the estimated field, # must be EVEN!!!
	observedfieldwidth = field_width/2;    # mm, -field_width/4 to field_width/4 
	spacestep=0.5;
	steps_in_field = field_width/spacestep + 1;

	Delta = 1./spacestep;
	Nspacestep_in_observed_field = Delta*observedfieldwidth+1	

	observation_offest = field_width/4;     # mm
	observation_offset_units = observation_offest / spacestep -1;

	f_space=pb.arange(-(field_width)/2.,(field_width)/2.+spacestep,spacestep)# the step size should in a way that we have (0,0) in our kernel as the center
	spatial_location_num=(len(f_space))**2
	Delta_s = 1.5	# mm
	Delta_s_units = Delta_s/spacestep	
	nonsymmetric_obs_location_units = pb.arange(1,Nspacestep_in_observed_field,Delta_s_units)
	offset = ((Nspacestep_in_observed_field - nonsymmetric_obs_location_units[-1])/2.)
	symmetricobslocation_units = nonsymmetric_obs_location_units + offset + observation_offset_units

	observation_locs_mm = symmetricobslocation_units*spacestep - field_width/2.
	print observation_locs_mm

	S_obs= observation_locs_mm

	obs_locns=gen_spatial_lattice(S_obs)





	#Define Sensor Kernel
	sensor_center=pb.matrix([[0],[0]])
	sensor_width=0.9**2 
	sensor_kernel=Basis(sensor_center,sensor_width,dimension)
	sensor_kernel_convolution_vecrorized=pb.vectorize(sensor_kernel.conv)
	sensor_kernel_conv_Phi=sensor_kernel_convolution_vecrorized(Phi).T #first row
	t_observation_matrix=time.time()
	sensor_kernel_conv_Phi_values_temp=[]
	sensor_kernel_conv_Phi_values=[]
	for m in range(sensor_kernel_conv_Phi.shape[1]):
		for n in obs_locns:
			sensor_kernel_conv_Phi_values_temp.append(sensor_kernel_conv_Phi[0,m](n))
		sensor_kernel_conv_Phi_values.append(sensor_kernel_conv_Phi_values_temp)
		sensor_kernel_conv_Phi_values_temp=[] 
	C=pb.matrix(pb.squeeze(sensor_kernel_conv_Phi_values).T)
	print 'Elapsed time in seconds to calculate observation matrix C is',time.time()-t_observation_matrix
	#Define field covariance function
	gamma_center=pb.matrix([[0],[0]])
	gamma_width=1.3**2 
	gamma_weight=0.01
	gamma=Basis(gamma_center,gamma_width,dimension)
	t_Sigma_e_c=time.time()
	gamma_convolution_vecrorized=pb.vectorize(gamma.conv)
	gamma_conv_Phi=gamma_convolution_vecrorized(Phi).T 
	Pi=pb.matrix(Phi*gamma_conv_Phi,dtype=float)
	Sigma_e=gamma_weight*Gamma_inv*Pi*Gamma_inv
	Sigma_e_c=pb.matrix(sp.linalg.cholesky(Sigma_e)).T
	print 'Elapsed time in seconds to calculate Cholesky decomposition of the noise covariance matrix',time.time()-t_Sigma_e_c
	print 'Elapsed time in seconds to generate the model',time.time()-total_time
	#files=[]
	#for i in range(81):
    	#	b=pb.array(Psi_conv_Phi_values[i])
	#	bob=pb.vstack(pb.split(b,81))
    	#	pb.imshow(bob.T,origin='lower',extent=[-20,20,-20,20])
    	#	fname = '_tmp%05d.jpg'%i
	#	pb.savefig(fname,format='jpg')
	#	pb.close()
	#	files.append(fname)
	#os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")

