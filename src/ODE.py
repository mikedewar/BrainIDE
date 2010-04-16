import pylab as pb
import numpy as np
import matplotlib.axes3d
import time,warnings
import os
import copy
from scipy import signal
import scipy as sp
class IDE():

	"""class defining a non-linear, Integro-Difference, discrete-time state space model.

	Arguments
	----------
	kernel: 
		IDE kernel.
	field : 
		spatial field.
	act_fun : 
		Sigmoidal activation function.
	alpha : Integer
		alpha = 1/tau, post-synaptic time constant, (Wendling, 2002, 100 for excitatory, 500 for inhibitory), Schiff = 3
	field_noise_variance: float
		The process noise variance, which drives the dynamic system
	obs_noise_covariance: matrix
		Observation noise covariance
	Sensorwidth: float
		Sensor diameter in mm
	obs_locns: List of matrix
		Sensors locations
	spacestep: float
		Spatial step size for descritization in mm
	Ts: float
		sampling time, seconds
	Attributes
	----------
	"""

	def __init__(self,kernel,field_space,Sensor,act_fun,alpha,field_noise_variance,beta_variance,obs_noise_covariance,spacestep,Ts):

		self.kernel = kernel
		self.field_space = field_space
		self.beta_variance=beta_variance
		self.act_fun = act_fun
		self.Sensor=Sensor
		self.alpha=alpha
		self.field_noise_variance=field_noise_variance
		self.obs_noise_covariance=obs_noise_covariance
		self.spacestep=spacestep
		self.Ts=Ts
		

	def gen_ssmodel(self):

		"""
		generates nonlinear IDE
		Arguments
		---------
		if sim=1 it generates K (sum of kernels which is used in simulation), the reason is finding K is faster
		    than finding each kernel individually and for the simulation we dont need to find each kernel individually
		Returns
		----------
		K: ndarray
			matrix of kernel values over the spatial domain of the kernel

		Sw: matrix
			covariance of the field
		Swc: matrix
			cholesky decomposiotion of Sw
		Svc: matrix
			cholesky decomposiotion of the observation noise covariance
			

		"""
		print "generating nonlinear IDE"

		#Calculating the disturbance covariance matrix

		spatial_location_num=(len(self.field_space))**2

		K_center=(pb.floor(self.kernel.space.shape[0]/2.),pb.floor(self.kernel.space.shape[0]/2.))


		# This is to check if the spatial domain of the  is odd, it must be odd in order 
		#to have center
		if ((pb.mod(self.kernel.space.shape[0],2.)) or (pb.mod(self.kernel.space.shape[1],2.)))==0:
			warnings.warn('Kernel doesnt have center')

		# Here we find the center of the kernel and the sensor they must be lcated at (0,0)
		print 'center of kernel spatial domain is',(self.kernel.space[K_center[0]],self.kernel.space[K_center[1]])

		K = pb.empty((len(self.kernel.space),len(self.kernel.space)))
		S= pb.empty((len(self.kernel.space),len(self.kernel.space)))
		


		print "pre-calculating observation kernel vectors"
		t0=time.time()
		Values=[]
		for s1 in self.Sensor.space:
			for s2 in self.Sensor.space:
				Values.append(self.Sensor.Sensor_vectors(pb.matrix([[s1],[s2]])))
			#we squeeze the result to have a matrix in a form of
			#[m1(s1) m1(s2) ...m1(sl);m2(s1) m2(s2) ... m2(sl);...;mny(s1) mny(s2) ... mny(sl)]
			#where sl is the number of spatial points after discritization
		self.Sensor.Values=pb.matrix(pb.squeeze(Values).T)
		print "Elapsed time in seconds is", time.time()-t0

		for i in range(len(self.kernel.space)):
			for j in range(len(self.kernel.space)):
				K[i,j]=self.kernel.__call__(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]))
				S[i,j]=gaussian(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]),pb.matrix([[0],[0]]),self.Sensor.widths[0],self.Sensor.dimension)
		self.K=K
		self.S=S

		#Calculating Sw
		print "calculating Covariance matrix"
		t0=time.time()
		beta_space=pb.empty((self.field_space.size**2,2),dtype=float)
		l=0
		for i in self.field_space:
			for j in self.field_space:
				beta_space[l]=[i,j]
				l+=1
		N1,D1 = beta_space.shape
		diff = beta_space.reshape(N1,1,D1) - beta_space.reshape(1,N1,D1)
		self.Sw=self.field_noise_variance*np.exp(-np.sum(np.square(diff),-1)*(1./self.beta_variance))
		print "Elapsed time in seconds is", time.time()-t0
		self.Swc=sp.linalg.cholesky(self.Sw).T
		
		Svc=sp.linalg.cholesky(self.obs_noise_covariance)
		self.Svc=Svc.T

	def simulate(self,T):

		Y=[]
		#V_vector=[] 
		V_matrix=[] #I don't save initial field
		V_filtered=[]
		#V_matrix_temp=[]
		spatial_location_num=(len(self.field_space))**2
		initial_field=self.Swc*pb.matrix(np.random.randn(spatial_location_num,1))		
		v_membrane=initial_field
		spiking_rate=pb.array([self.act_fun(float(v_membrane[i,0])) for i in range(v_membrane.shape[0])])


		for t in T[1:]:

			v = self.Svc*pb.matrix(np.random.randn(len(self.Sensor.centers),1))
			w = self.Swc*pb.matrix(np.random.randn(spatial_location_num,1))
			print "simulation at time",t

			Kernel_convolution_at_s=signal.convolve2d(self.K,pb.vstack(pb.split(spiking_rate,len(self.field_space))),mode='same',boundary='fill')


			Kernel_convolution_at_s= (self.spacestep**2)*pb.matrix(pb.ravel(Kernel_convolution_at_s)).T
			v_membrane=self.Ts*Kernel_convolution_at_s +(1-self.Ts*self.alpha)*v_membrane+w

			spiking_rate=pb.array([self.act_fun(float(v_membrane[i,0])) for i in range(v_membrane.shape[0])])


			#------------------------------
			#If you need field in vector form you can uncomment this line
			#V_vector.append(v_membrane)
			#------------------------------

			# we need to transform the field vector into a matrix to plot, the vector starts
			#at the left top corner and it goes right to the right top corner e.g. [(-2,-2),(-2,-1)...(-2,2),(-1,-2),(-1,-1)...(-1,2),...
			#(2,-2),(2,-1)...(2,2)] for x,y in [-2,2] we need to transform it to [(-2,-2),(-2,-1)...(-2,2);(-1,-2),(-1,-1)...(-1,2);...
			#(2,-2),(2,-1)...(2,2)]
			v_membrane_matrix=pb.hstack(pb.split(v_membrane,len(self.field_space))).T #Result is a matrix
			#--------------------------------------------------------------------------------------------------
			# This is to test PBC and it's not needed for the main simulation
			#V_matrix_temp.append(pb.hstack(pb.split(self.Ts*Kernel_convolution_at_s+(1-self.Ts*self.alpha)*v_membrane,len(self.field_space))).T)
			#V_matrix_temp.append(pb.hstack(pb.split(w,len(self.field_space))).T)
			#--------------------------------------------------------------------------------------------------
			#Finding the convolution of the Sensor with the field
			v_filtered=(self.spacestep**2)*signal.convolve2d(self.S,v_membrane_matrix,mode='same',boundary='fill')
			#-------------------------------------------------------------------------------------------------------
			#Observation
			Y.append((self.spacestep**2)*(self.Sensor.Values*v_membrane)+v)
			#-------------------------------------------------------------------------------------------------------
			#Free Boundary Condition
			# we simulate over a bigger field to avoid boundary conditions and then we look at the middle part of
			# the field

			field_width=pb.absolute(self.field_space[-1])
			#column
			v_membrane_matrix=v_membrane_matrix[:,field_width/(self.spacestep*2.):field_width/float(self.spacestep)+field_width/(self.spacestep*2.)+1]
			v_filtered=v_filtered[:,field_width/(self.spacestep*2.):field_width/float(self.spacestep)+field_width/(self.spacestep*2.)+1]
			#row
			v_membrane_matrix=v_membrane_matrix[field_width/(self.spacestep*2.):field_width/float(self.spacestep)+field_width/(self.spacestep*2.)+1,:]
			v_filtered=v_filtered[field_width/(self.spacestep*2.):field_width/float(self.spacestep)+field_width/(self.spacestep*2.)+1,:]
			#--------------------------------------------------------------------------------------------------------
			V_matrix.append(v_membrane_matrix)
			V_filtered.append(v_filtered)

		return V_matrix,V_filtered,Y




class Sensor():

	def __init__(self,centers, widths, dimension,space,n,spacestep):
		
		
		self.centers=centers
		self.n=n
		self.widths=self.n*widths
		self.dimension=dimension
		self.space = space
		self.spacestep=spacestep

	def Sensor_vectors(self,s):

		"""
		generates vector of sensors at a given spatial location

		Arguments
		----------
		s : float
			spatail location
		Returns
		----------
		vector of observation kernels: matrix
			matrix of ny x 1 dimension [m_1(s) m_2(s) ... m_ny(s)].T
		"""
		return pb.matrix([gaussian(s,cen,wid,self.dimension) for cen,wid, in zip(self.centers,self.widths)]).T

		

class Kernel():

	"""class defining the Connectivity kernel of the brain.

	Arguments
	----------
	weights: list
		
	centers: list of matrix
			each of the entery should be a matrix in a form of (2x1)
	widths: list of matrix
			each 2x2 matrix is the width of the corresponding basis function.

	dimension: int
			dimension of the kernel
	space: array
			spatial domain of the kernel


	Attributes
	----------
	evaluate: 
		evaluate the kernel at a given spatial location
	"""
	
	def __init__(self, weights, centers, widths, space, dimension):
		self.weights = weights
		self.centers=centers
		self.widths = widths
		self.dimension=dimension
		self.space=space
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		"""
		evaluates the kernel at a given spatial location

		Arguments
		----------
		s : matrix
			spatail location, it must be in a form of dimension x 1

		"""
		return self.evaluate(s)





class ActivationFunction():

	"""class defining the sigmoidal activation function .

	Arguments
	----------
	threshold: float
		firing threshold, mV                  (Wendling, 2002, 6 mV), (Schiff, 2007, threshold = 0.24 (Heaviside))
	nu: float
		maximum firing rate, spikes/s         (Wendling, 2002, 2*e0 = 5, or nu = 5
	beta: float
		slope of sigmoid, spikes/mV           (Wendling, 2002, 0.56 mV^-1)

	v: float
		Presynaptic potential

	Returns
	----------
	average firing rate
	"""	
	def __init__(self,threshold=6,nu=1,beta=0.56):
		
		self.threshold = threshold
		self.nu = nu
		self.beta = beta
		
	def __call__(self,v):

		return float(self.nu/(1.+pb.exp(self.beta*(self.threshold-v))))
	def plot(self,plot_range):
		u=pb.linspace(-plot_range,plot_range,1000)
		z=pb.zeros_like(u)
		for i,j in enumerate(u):
			z[i]=self.__call__(j)
		pb.plot(u,z)
		pb.show()


def gaussian(s,centre,width,dimension):

		"""
		defines Gaussian basis function

		Arguments
		----------
		s : matrix
			it should be dimension x 1
			
		center:  matrix
			it should be a matrix in form of dimension x 1
		width: matrix
			*variance or *covariance matrix of the Gaussian function
		dimension: int
			dimension of the basis function
		"""
		centre = pb.matrix(centre)
		centre.shape = (dimension,1)
		width = pb.matrix(width)
		return float(pb.exp(-(s-centre).T*width.I*(s-centre)))

def anim_ode_field(V_matrix,f_space):	

	"""
		animates spatial field

		Arguments
		----------
		V: matrix
			Spatial field at each time instant

		f_space: array
			x,y of the spatial locations	
	"""	
	pb.ion()
	fig_hndl =pb.figure()
	files=[]
	for t in range(len(V_matrix)):
		image=pb.imshow(V_matrix[t],animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])
		pb.axis([f_space[spacestep],f_space[-spacestep],f_space[spacestep],f_space[-spacestep]])

	pb.ioff()


def avi_field(V_matrix,f_space,spacestep,filename,play=0):

	"""
		creat avi for the neural field

		Arguments
		----------
		V_matrix: list of matrix
			field at each time 

		f_space: array
			x,y of the spatial locations

		filename: 	
		Returns
		---------
		avi of the neural field

	"""
	fig_hndl =pb.figure()
	files=[]
	filename=" "+filename


	for t in range(len(V_matrix)):

		pb.figure()	
		pb.imshow(V_matrix[t],animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])
		#pb.imshow(pb.hstack([V_matrix[t]]*2),animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])

		pb.colorbar()

		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")



def avi_observation(Y,f_space,spacestep,filename,play=0):

	Y_matrix=[]
	for i in range(len(Y)):
    		Y_matrix.append(pb.hstack(pb.split(Y[i],pb.sqrt(len(Y[0])))).T)
		fig_hndl =pb.figure()
	files=[]
	filename=" "+filename


	for t in range(len(Y_matrix)):


		pb.figure()		
		pb.imshow(Y_matrix[t],animated=True,origin='lower',aspect=1, alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])
		pb.colorbar()

		fname = '_tmp%05d.jpg'%t
		pb.savefig(fname,format='jpg')
		pb.close()
		files.append(fname)
	os.system("ffmpeg -r 5 -i _tmp%05d.jpg -y -an"+filename+".avi")
	# cleanup
	for fname in files: os.remove(fname)
	if play: os.system("vlc"+filename+".avi")







def plot_field(V,f_space):

	"""
		plots spatial field

		Arguments
		----------
		V: matrix
			Spatial field at each time instant

		f_space: array
			x,y of the spatial locations	
		Returns
		---------
		plot the spatial field at a given time instant

	"""
	
	pb.imshow(V,aspect=1,origin='lower', alpha=1,extent=[min(f_space),max(f_space),min(f_space),max(f_space)])#interpolation='nearest',	
	pb.colorbar(shrink=.55) 
	pb.xlabel(r'$ s_1$',fontsize=25)
	pb.ylabel(r' $s_2$',fontsize=25)
	pb.show()

	
	
def circle(center,radius,color):
	"""
		plot a circle with given center and radius

		Arguments
		----------
		center : matrix or ndarray
			it should be 2x1 ndarray or matrix
		radius:  float
			masses per mm
	"""
	u=pb.linspace(0,2*np.pi,200)
	x0=pb.zeros_like(u)
	y0=pb.zeros_like(u)
	for i,j in enumerate(u):
		x0[i]=radius*pb.sin(j)+center[0,0]
		y0[i]=radius*pb.cos(j)+center[1,0]
	pb.plot(x0,y0,color)	

def gen_obs_locations(FieldWidth,SensorWidth,SensorSpacing,BoundryEffectWidth):
	'''
	SensorWidth:  diameter in mm	
	SensorSpacing  in mm
	BoundryEffectWidth 
	'''
	ObservationCentre = pb.linspace(-FieldWidth/2. + BoundryEffectWidth,FieldWidth/2. - BoundryEffectWidth,FieldWidth/SensorSpacing)
	#ObservationCentre = pb.arange(-FieldWidth/2. + BoundryEffectWidth,FieldWidth/2. - BoundryEffectWidth+1,SensorSpacing)
	return [np.matrix([[i,j]]).T for i in ObservationCentre for j in ObservationCentre]	
	
def gen_spatial_lattice(S):
	"""generates a list of vectors, where each vector is a co-ordinate in a lattice"""
	number_of_points = len(S)**2

	space = [0 for i in range(number_of_points)]
	count = 0
	for i in S:
		for j in S:
			space[count] = pb.matrix([[i],[j]])
			count += 1
	return space

def obs_frequency_response(Y,scale,spacestep,db=0,vmin=None,vmax=None,save=0,filename='filename'):

	'''Generates average FFT of all observed surface
	Argument:
	---------
		Y: list of matrix
			all observed surface
		spacestep: float
			distance beetween adjacent sensors

		db: zero or one
			if one result is given in db
		save: zero or one
			if one it saves the image in the working directory
	Return:
	-------
		average magnitude of the fft of the observation field: ndarray

		freq: ndarray
			the frequency range over which fft2 is taken
	'''


	Y_matrix=[]
	for i in range(len(Y)):
    		Y_matrix.append(pb.hstack(pb.split(Y[i],pb.sqrt(len(Y[0])))).T)
	y=0

	if db:
		for i in range(len(Y_matrix)):
			y=y+10*pb.log10((spacestep**2)*pb.absolute(signal.fft2(Y_matrix[i])))
	else:
		for i in range(len(Y_matrix)):
			y=y+(spacestep**2)*pb.absolute(signal.fft2(Y_matrix[i]))
	y=y/len(Y_matrix)
	freq=pb.fftfreq(y.shape[0],float(spacestep)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	#Nyquist_freq=freq.max() #find the Nyquist frequency it is not exact (depending y.shape[0] is odd or even) but close to Nyquist frequency
	#freq=pb.fftshift(freq)
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	params = {'axes.labelsize': 15,'text.fontsize': 15,'legend.fontsize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15}
	pb.rcParams.update(params) 
	#pb.imshow((y*scale),origin='lower',extent=[freq[0],freq[-1],freq[0],freq[-1]],interpolation='nearest',vmin=vmin,vmax=vmax)#,cmap=pb.cm.gray,interpolation='nearest',cmap=pb.cm.gray,
	pb.imshow(y,origin='lower',extent=[0,freq_range,0,freq_range],interpolation='nearest',vmin=vmin,vmax=vmax,cmap=pb.cm.gray)
	pb.colorbar(shrink=.55)
	pb.xlabel('Hz',fontsize=25)
	pb.ylabel('Hz',fontsize=25)
	if save:
		pb.savefig(filename+'.pdf',dpi=300)

	pb.show()	
	return pb.flipud(y),freq



def field_frequency_response(V_matrix,scale,spacestep,db=0,vmin=None,vmax=None,save=0,filename='filename'):

	'''Generates  FFT of the spatial kernel
	Argument:
	---------
		K: matrix
			spatial kernel
		spacestep: float
			distance beetween adjacent spatial point

		db: zero or one
			if one result is given in db
		save: zero or one
			if one it saves the image in the working directory
	Return:
	-------
		 magnitude of the fft of the spatial kernel: ndarray

		freq: ndarray
			the frequency range over which fft2 is taken
	'''

	V_f=0
	if db:
		for i in range(len(V_matrix)):
			V_f=V_f+10*pb.log10((spacestep**2)*pb.absolute(signal.fft2(V_matrix[i])))

	else:
		for i in range(len(V_matrix)):
			V_f=V_f+(spacestep**2)*pb.absolute(signal.fft2(V_matrix[i]))

	V_f=V_f/len(V_matrix)
	freq=pb.fftfreq(V_f.shape[0],float(spacestep)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	#freq=pb.fftshift(freq)
	#Nyquist_freq=freq.max() #find the Nyquist frequency it is not exact (depending y.shape[0] is odd or even) but close to Nyquist frequency
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	params = {'axes.labelsize': 15,'text.fontsize': 15,'legend.fontsize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15}
	pb.rcParams.update(params) 
	#pb.imshow(V_f,origin='lower',extent=[0,freq_range,0,freq_range],vmin=vmin,vmax=vmax)#,cmap=pb.cm.gray,interpolation='nearest'
	pb.imshow(scale*V_f,origin='lower',extent=[0,freq_range,0,freq_range],cmap=pb.cm.gray,interpolation='nearest',vmin=vmin,vmax=vmax)
	pb.colorbar(shrink=.55)
	pb.xlabel('Hz',fontsize=25)
	pb.ylabel('Hz',fontsize=25)
	if save:
		pb.savefig(filename+'.pdf',dpi=300)

	pb.show()	
	return pb.flipud(V_f),freq



def kernel_frequency_response(K,spacestep,db=0,save=0,filename='filename'):

	'''Generates  FFT of the spatial kernel
	Argument:
	---------
		K: matrix
			spatial kernel
		spacestep: float
			distance beetween adjacent spatial point

		db: zero or one
			if one result is given in db
		save: zero or one
			if one it saves the image in the working directory
	Return:
	-------
		 magnitude of the fft of the spatial kernel: ndarray

		freq: ndarray
			the frequency range over which fft2 is taken
	'''

	if db:	k_f=10*pb.log10((spacestep**2)*pb.absolute(signal.fft2(K)))
	else:	k_f=pb.absolute((spacestep**2)*signal.fft2(K))
	freq=pb.fftfreq(k_f.shape[0],float(spacestep)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	#Nyquist_freq=freq.max() #find the Nyquist frequency it is not exact (depending y.shape[0] is odd or even) but close to Nyquist frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
	pb.rcParams.update(params) 
	pb.imshow(k_f,origin='lower',extent=[0,freq_range,0,freq_range],vmin=k_f.min(),vmax=k_f.max())#,cmap=pb.cm.grayinterpolation='nearest',
	pb.xlabel('Hz',fontsize=25)
	pb.ylabel('Hz',fontsize=25)
	pb.colorbar(shrink=.55)
	if save:
		pb.savefig(filename+'.pdf',dpi=300)

	pb.show()
	return pb.flipud(k_f),freq
