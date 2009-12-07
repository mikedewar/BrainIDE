import pylab as pb
import numpy as np

class IDE():

	def __init__(self,kernel,field,act_fun,alpha,field_noise_variance,obs_noise_covariance,obs_locns,stepsize):
		self.kernel = kernel
		self.field = field
		self.act_fun = act_fun
		self.alpha=alpha
		self.field_noise_variance=field_noise_variance
		self.obs_locns=obs_locns
		self.obs_noise_covariance=obs_noise_covariance
		self.stepsize=stepsize
		

	def sim(self,init_field,T):
		print "simulating nonlinear IDE"
		#This is to check, this bit can be done Analitically
		
		# do some splurging!
		print "pre-calculating kernel matrices"
		ns = len(self.field.space)
		K = np.zeros((ns,ns),dtype=object)
		H = np.zeros((ns,ns),dtype=object)
		for si,s in enumerate(self.field.space):
			for ri,r in enumerate(self.field.space):
				K[si,ri] = self.kernel(s-r)
				H[si,ri] = gaussian(s-r,pb.matrix([0,0]),pb.matrix([[1,0],[0,1]]),2)
		print "pre-calculating basis function vectors"
		fbases = np.empty(ns,dtype=object)
		for ri,r in enumerate(self.field.space):
			fbases[ri] = self.field.field_bases(r)
		self.field.fbases=fbases
		
		print "calculating Psi_x"		
		Psi_x=(self.stepsize**2)*sum([f*f.T for f in fbases])
		print "calculating Psix_inv"
		Psi_xinv=Psi_x.I
		
		print "calculating field noise covariances"
		Sw=self.field_noise_variance*Psi_xinv* self.field.field_noise()*Psi_xinv.T
		Swc=pb.linalg.cholesky(Sw)
		Svc=pb.linalg.cholesky(self.obs_noise_covariance)
		
		Y = []
		X = []		
		x=init_field
		X.append(x)
		
		print "iterating"
		for t in range(T):
			print t
			print "\t calculating noise"
			w = Swc*pb.matrix(np.random.randn(self.field.nx,1))
			v = Svc*pb.matrix(np.random.randn(len(self.obs_locns),1))
			K_at_si=[]
			y = []
			print "\t summing over space for state"
			for si in range(len(self.field.space)):
				K_at_si.append(pb.sum([K[si,ri]*self.act_fun(fr.T*x) for ri,fr in enumerate(fbases)]))
			sum_s=pb.hstack(fbases)*pb.matrix(K_at_si).T
			sum_s *= (self.stepsize**4)
			print "\t sampling for y"
			for si, s in enumerate(self.obs_locns):
				y.append((self.stepsize**2)*pb.sum([
					H[si,ri]*float((fr.T*X[t])) for ri,fr in enumerate(fbases)
				]))
			x=Psi_xinv*sum_s-self.alpha*x+w
			Y.append(pb.matrix(y).T+Svc*v)
			X.append(x)
		return X,Y


class Field():
	
	def __init__(self, weights,centers, widths, dimension,space,nx,stepsize):

		self.nx=nx
		self.weights = weights
		self.widths=self.nx*widths
		self.centers=centers
		self.dimension=dimension
		self.space = pb.array(space)
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
				beta[si,ri]=gaussian(s-r,pb.matrix([0,0]),pb.matrix([[1,0],[0,1]]),2)
		beta_at_si=[]
		for si in range(len(self.space)):
			beta_at_si.append(sum([beta[si,ri]*fr.T for ri,fr in enumerate(self.fbases)]))

		sum_s=pb.hstack(self.fbases)*np.vstack(beta_at_si)
		sum_s *= (self.stepsize**4)
		return sum_s

	def plot(self,centers):
		for center in centers:
			circle(center,2*self.widths[0][0,0])
		pb.title('field decomposition')	
		pb.show()


class Kernel():
	
	def __init__(self, weights, centers, widths, dimension):
		self.weights = weights
		self.centers=centers
		self.widths = widths
		self.dimension=dimension

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
	
	def __init__(self,threshold=0.1,max_firing_rate=1,slope=0.2):
		
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
	


def gen_obs_locations(xmin,xmax,ymin,ymax,resolution=1):

	x_range = pb.arange(xmin,xmax+resolution,resolution)
	y_range = pb.arange(ymin,ymax+resolution,resolution)
	return [np.matrix([[i,j]]).T for i in x_range for j in y_range]
	

def gen_spatial_lattice(xmin,xmax,ymin,ymax,stepsize):

	x_space=pb.arange(xmin,xmax+stepsize,stepsize)
	y_space=pb.arange(ymin,ymax+stepsize,stepsize)

	space=[]
	for si1,s1 in enumerate(x_space):			
		for si2,s2 in enumerate(y_space):
			space.append(pb.matrix([[s1],[s2]]))
	return space


if __name__ == "__main__":

	#-------------field--------------------
	f_centers=[np.matrix([[i,j]]) for i in pb.arange(0,5) for j in pb.arange(0,5)]
	nx=len(f_centers)
	f_widths=[pb.matrix([[1,0],[0,1]])]*nx
	f_weights=[1]*nx

	stepsize=0.5	
	f_space=gen_spatial_lattice(0,5,0,5,stepsize)
	f=Field(f_weights,f_centers,f_widths,2,f_space,nx,stepsize)
	f.plot(f_centers)
	#------------Kernel-------------
	k_centers=[np.matrix([[.5,.5]])]
	k_weights =[.1]
	k_widths=[pb.matrix([[1,0],[0,1]])]
	k_space = pb.linspace(-2,3,50) #This is just to plot the kernel, doesn't affect the simulation
	k=Kernel(k_weights,k_centers,k_widths,2)
	# k.plot(k_space)
	#-------Brain----------------
	alpha=.05
	field_noise_variance=0.8 
	act_func=ActivationFunction()
	#----------observations--------------------------
	obs_locns =gen_obs_locations(0,5,0,5,resolution=1)
	obs_noise_covariance =.2*np.eye(len(obs_locns),len(obs_locns))
	#----------initialasation----------------------------
	init_field=[0]*nx
	init_field[2]=1
	init_field=pb.matrix(init_field).T
	#--------------model and simulation------------------
	T=30
	model=IDE(k,f, act_func,alpha,field_noise_variance,obs_noise_covariance,obs_locns,stepsize)
	X,Y=model.sim(init_field,T)

