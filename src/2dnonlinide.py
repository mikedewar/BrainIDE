import pylab as pb
import numpy as np
import matplotlib.axes3d
import time,warnings
class IDE():

	def __init__(self,kernel,field,act_fun,alpha,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,stepsize,T):
		self.kernel = kernel
		self.field = field
		self.act_fun = act_fun
		self.alpha=alpha
		self.field_noise_variance=field_noise_variance
		self.obs_locns=obs_locns
		self.obs_noise_covariance=obs_noise_covariance
		self.stepsize=stepsize
		self.Sensorwidth=Sensorwidth
		self.T=T
		

	def gen_ssmodel(self):
		print "generating nonlinear IDE"
		# do some splurging!
		K_center=(pb.floor(self.kernel.space.shape[0]/2.),pb.floor(self.kernel.space.shape[0]/2.))
		H_center=(pb.floor(self.kernel.space.shape[0]/2.),pb.floor(self.kernel.space.shape[0]/2.))
		if ((pb.mod(self.kernel.space.shape[0],2.)) or (pb.mod(self.kernel.space.shape[1],2.)))==0:
			warnings.warn('Kernel doesnt have center')
		print 'center of kernel spatial domain is',(model.kernel.space[K_center[0]],model.kernel.space[K_center[1]])
		print 'center of observation function spatial domain is',(model.kernel.space[H_center[0]],model.kernel.space[H_center[1]])
		print "pre-calculating kernel matrices"
		ns = len(self.field.lattice)
		K = pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)
		H = pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)
		#beta = pb.empty((len(self.kernel.space),len(self.kernel.space)),dtype=object)
		for i in range(len(self.kernel.space)):
			for j in range(len(self.kernel.space)):
				K[i,j]=self.kernel.__call__(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]))	
				H[i,j] = gaussian(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]),pb.matrix([0,0]),pb.matrix([[self.Sensorwidth,0],[0,self.Sensorwidth]]),2)
				#beta[i,j]=gaussian(pb.matrix([[self.kernel.space[i]],[self.kernel.space[j]]]),pb.matrix([0,0]),pb.matrix([[5,0],[0,5]]),2)


		self.K=K
		self.H=H
		#self.beta=beta
		print "pre-calculating basis function vectors"
		t0=time.time()
		fbases=[]
		for s1 in self.field.space:
			for s2 in self.field.space:
				fbases.append(self.field.field_bases(pb.matrix([[s1],[s2]])))
		self.field.fbases=fbases
		print "Elapsed time in seconds is", time.time()-t0


		print "calculating Psi_x"		
		Psi_x=(self.stepsize**2)*sum([f*f.T for f in fbases])
		self.Psi_x=Psi_x
		print "calculating Psix_inv"
		Psi_xinv=Psi_x.I
		self.Psi_xinv=Psi_x.I		
		print "calculating field noise covariances"		
		#Sw=self.field_noise_variance*Psi_xinv* self.field.field_noise()*Psi_xinv.T
		Sw=self.field_noise_variance*Psi_xinv
		self.Sw=Sw
		try:
			self.Swc=pb.linalg.cholesky(Sw)
		except pb.LinAlgError:
		#	print Sw
		#	print self.field.field_noise()
		#	print Psi_xinv
			raise
		Svc=pb.linalg.cholesky(self.obs_noise_covariance)
		self.Svc=Svc

	def simulate(self,init_field,T):
		Y = []
		X = []		
		x=init_field
		X.append(x)
		K_center=(pb.floor(self.K.shape[0]/2.),pb.floor(self.K.shape[1]/2.))
		H_center=(pb.floor(self.H.shape[0]/2.),pb.floor(self.H.shape[1]/2.))
		origin=pb.matrix(self.field.lattice[0])		
		obs_locns_shift=observation_locations_shift(self.obs_locns,origin,self.stepsize)
		obs_locns_shift_index=[int(obs_locns_shift[i][0]*self.H.shape[0]+obs_locns_shift[i][1]) for i in range(len(self.obs_locns))]
		print "iterating"
		for t in range(T):
			w = self.Swc*pb.matrix(np.random.randn(self.field.nx,1))
			v = self.Svc*pb.matrix(np.random.randn(len(self.obs_locns),1))
			field_update=np.array([self.act_fun(fr.T*x) for fr in self.field.fbases])
			print "simulation at time",t

			Kernel_convolution_at_s=[]
			for i in range(len(self.field.space)):
				K_shift_x=pb.roll(self.K,i*len(self.kernel.space)) #shift kernel along x
				for j in range(len(self.field.space)):
					K_shift_y=pb.roll(K_shift_x,j) #shift kernel along y
					K_truncate=K_shift_y[K_center[0]:,K_center[1]:]
					Kernel_convolution_at_s.append(pb.sum(field_update*pb.ravel(K_truncate)))	
			sum_s=pb.hstack(self.field.fbases)*pb.matrix(Kernel_convolution_at_s).T
			

			sum_s *= (self.stepsize**4)
			x=self.Psi_xinv*sum_s-self.alpha*x+w
			X.append(x)


			observation_convolution_at_s=[]
			for i in range(len(self.obs_locns)):
				H_shift_x_y=pb.roll(self.H,obs_locns_shift_index[i]) #shift H to observation locations
				H_truncate=H_shift_x_y[H_center[0]:,H_center[1]:]
				observation_convolution_at_s.append((self.stepsize**2)*sum(field_update*pb.ravel(H_truncate)))
			Y.append(pb.matrix(observation_convolution_at_s).T+self.Svc*v)
			

		return X,Y


class Field():	

	def __init__(self, weights,centers, widths, dimension,lattice,space,nx,stepsize):

		self.nx=nx
		self.weights = weights
		self.widths=self.nx*widths
		self.centers=centers
		self.dimension=dimension
		self.space = space
		self.lattice=pb.array(lattice)
		self.stepsize=stepsize
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,self.dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		return float(self.evaluate(s))		

	def field_bases(self,s):
		return pb.matrix([gaussian(s,cen,wid,self.dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)]).T

	def field_noise(self):
		print "calculating covariance matrix"
		beta= np.zeros((len(self.space),len(self.space)),dtype=object)
		for si,s in enumerate(self.space):
			for ri,r in enumerate(self.space):
				beta[si,ri]=gaussian(s-r,pb.matrix([0,0]),pb.matrix([[5,0],[0,5]]),2)
		beta_at_si=[]
		for si in range(len(self.space)):
			beta_at_si.append(sum([beta[si,ri]*fr.T for ri,fr in enumerate(self.fbases)]))

		sum_s=pb.hstack(self.fbases)*np.vstack(beta_at_si)
		sum_s *= (self.stepsize**4)
		return sum_s
	

	def plot(self,centers):
		for center in centers:
			circle(center,2*pb.sqrt(pb.log(2))*(pb.sqrt(self.widths[0][0,0])))
		pb.title('field decomposition')	
		pb.show()


class Kernel():
	
	def __init__(self, weights, centers, widths, space, dimension):
		self.weights = weights
		self.centers=centers
		self.widths = widths
		self.dimension=dimension
		self.space=space
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		return self.evaluate(s)


	def plot(self,space):
		y = pb.zeros((space.shape[0],space.shape[0]))
		for i in range(len(space)):
			for j in range(len(space)):
				y[i,j]=self.__call__(pb.matrix([[space[i]],[space[j]]]))			
		fig = pb.figure()
		ax = matplotlib.axes3d.Axes3D(fig)
		s1,s2=pb.meshgrid(space,space)
		ax.plot_wireframe(s1,s2,y,color='k')
		pb.title('kernel')
		pb.show()
		
class ActivationFunction():
	
	def __init__(self,threshold=6,max_firing_rate=1,slope=0.56):
		
		self.threshold = threshold
		self.max_firing_rate = max_firing_rate
		self.slope = slope
		
	def __call__(self,v):

		return float(self.max_firing_rate/(1+pb.exp(self.slope*(self.threshold-v))))


def gaussian(s, centre, width,dimension):
		dimension = dimension
		centre = pb.matrix(centre)
		centre.shape = (dimension,1)
		width = pb.matrix(width)
		return float(pb.exp(-(s-centre).T*width.I*(s-centre)))

def circle(center_matrix,r):
	u=pb.linspace(0,2*np.pi,200)
	x0=pb.zeros_like(u)
	y0=pb.zeros_like(u)
	for i in range(len(u)):
		x0[i]=r*pb.sin(u[i])+center_matrix[0,0]
		y0[i]=r*pb.cos(u[i])+center_matrix[0,1]
	pb.plot(x0,y0)
	


def gen_obs_lattice(observation_centers):

	return [np.matrix([[i,j]]).T for i in observation_centers for j in observation_centers]
	
def gen_obs_locations(FieldWidth,SensorWidth,SensorSpacing,BoundryEffectWidth):
	'''
	SensorWidth:  diameter in mm	
	SensorSpacing  in mm
	BoundryEffectWidth 
	'''
	ObservationCentre = pb.arange(-FieldWidth/2. + BoundryEffectWidth,FieldWidth/2. - BoundryEffectWidth+1,SensorSpacing)
	NSensors = len(ObservationCentre**2)
	RightDistance = np.abs((FieldWidth/2. - BoundryEffectWidth) - ObservationCentre[-1]);              
	ObservationCentre = ObservationCentre + RightDistance/2.;   
	return ObservationCentre

def gen_spatial_lattice(xmin,xmax,ymin,ymax,stepsize):

	x_space=pb.arange(xmin,xmax+stepsize,stepsize)
	y_space=pb.arange(ymin,ymax+stepsize,stepsize)

	space=[]
	for si1,s1 in enumerate(x_space):			
		for si2,s2 in enumerate(y_space):
			space.append(pb.matrix([[s1],[s2]]))
	return space

def field_centers(FieldWidth,Massdensity,field_basis_separation,field_basis_width):

	N_masses_in_width=Massdensity*FieldWidth+1
	
	x_center_outer =pb.linspace(-FieldWidth/2.,FieldWidth/2.,N_masses_in_width/(2.*field_basis_separation/pb.sqrt(2)));
	y_center_outer = x_center_outer;

	distance_between_centers = abs(x_center_outer[2]-x_center_outer[1]);
	x_center_inner = pb.linspace(-FieldWidth/2.+distance_between_centers/2.,FieldWidth/2.-distance_between_centers/2.,N_masses_in_width/		(2.*field_basis_separation/pb.sqrt(2))-1)
	y_center_inner = x_center_inner

	f_centers_inner=[np.array([[i,j]]) for i in x_center_inner for j in y_center_inner]
	f_centers_outer=[np.array([[i,j]]) for i in x_center_outer for j in y_center_outer]
	f_centers=np.concatenate([f_centers_outer,f_centers_inner])
	return [pb.matrix(x) for x in f_centers]


def observation_locations_shift(obs_locations,origin,stepsize):
	'''This is to find appropriate shifts
	in order to calculate the observation convolution '''
	w=[(1./stepsize) *(obs_locations[i]-origin) for i in range(len(obs_locations))]
	return w


def plot_field(X,fbases):
	z=pb.zeros((int(pb.sqrt(len(fbases))),int(pb.sqrt(len(fbases)))))
	m=0
	for i in range(int(pb.sqrt(len(fbases)))):
		for j in range(int(pb.sqrt(len(fbases)))):
			z[i,j]=float(X.T*fbases[m]) 
			m+=1
	pb.imshow(z)	
	pb.colorbar(shrink=.55) 
	pb.show()
	 

def plot_states(Xreal,Xest):
	T=len(Xest)
	for i in range(len(Xest[0])):
		x1=pb.zeros(T)
		x2=pb.zeros(T)
		for tau in range(T):
			x1[tau]=(float(Xreal[tau][i]))
			x2[tau]=(float(Xest[tau][i]))
		pb.plot(range(T),x1+5*i,'b',range(T),x2+5*i,':b')

if __name__ == "__main__":

	#-------------field--------------------
	Massdensity=2#1
	field_basis_separation=3#4#3
	field_basis_width=2
	field_width=10#20
	f_centers=field_centers(field_width,Massdensity,field_basis_separation,field_basis_width)
	nx=len(f_centers)
	f_widths=[pb.matrix([[field_basis_width,0],[0,field_basis_width]])]*nx
	f_weights=[1]*nx
	stepsize=.5

	f_lattice=gen_spatial_lattice(-field_width/2.,field_width/2.,-field_width/2.,field_width/2.,stepsize)
	f_space=pb.arange(-field_width/2.,(field_width/2.)+stepsize,stepsize)# the step size should in a way that we have (0,0) in our kernel as the center
	f=Field(f_weights,f_centers,f_widths,2,f_lattice,f_space,nx,stepsize)
	f.plot(f_centers)
	#------------Kernel-------------
	k_centers=[np.matrix([[0,0]]),np.matrix([[0,0]]),np.matrix([[0,0]])]
	k_weights =[1e-5,-.8e-5,0.05e-5]
	#k_weights =[1,-.8,.05]
	k_widths=[pb.matrix([[4**2,0],[0,4**2]]),pb.matrix([[6**2,0],[0,6**2]]),pb.matrix([[15**2,0],[0,15**2]])]
	k_space=pb.arange(-field_width,(field_width)+stepsize,stepsize) #It should be chosen in a way that the center of the kernel lies on (0,0)
	k=Kernel(k_weights,k_centers,k_widths,k_space,2)
	#k.plot(k_space)
	#-------Brain----------------
	alpha=.5
	field_noise_variance=.8
	act_func=ActivationFunction(threshold=6,max_firing_rate=1,slope=0.56)
	#----------observations--------------------------
	Sensorwidth = .36     # equals to 1mm 
	SensorSpacing = 2*stepsize     # mm factor of stepsize
	BoundryEffectWidth = 1 #mm 

	observation_centers=gen_obs_locations(field_width,Sensorwidth,SensorSpacing,BoundryEffectWidth)
	obs_locns =gen_obs_lattice(observation_centers)
	#f.plot(obs_locns)
	[circle(cent.T,2*Sensorwidth) for cent in obs_locns]
	pb.title('Sensors locations')
	pb.show()

	obs_noise_covariance =.1*pb.matrix(np.eye(len(obs_locns),len(obs_locns)))
	
	#----------initialasation----------------------------
	mean=[0]*nx
	Initial_field_covariance=10*pb.eye(len(mean))
	init_field=pb.multivariate_normal(mean,Initial_field_covariance,[1]).T
	#--------------model and simulation------------------
	T=5
	model=IDE(k,f, act_func,alpha,field_noise_variance,obs_noise_covariance,Sensorwidth,obs_locns,stepsize,T)
	model.gen_ssmodel() 
	X,Y=model.simulate(init_field,50)
	plot_field(X[2],model.field.fbases)


