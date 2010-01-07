import pylab as pb
import numpy as np
import matplotlib.axes3d

class IDE():
	
	def __init__(self, kernel, field, act_fun,alpha,field_noise_variance,obs_noise_covariance,obs_locns,stepsize):
		self.kernel = kernel
		self.field = field
		self.act_fun = act_fun
		self.alpha=alpha
		self.field_noise_variance=field_noise_variance
		self.obs_locns=obs_locns
		self.obs_noise_covariance=obs_noise_covariance
		self.stepsize=stepsize

	def sim(self,init_field,T):
		#This is to check, this bit can be done Analitically		
		Psi_x=(self.stepsize*sum([self.field.field_bases(s)*self.field.field_bases(s).T for s in self.field.space]))
		Psi_xinv=Psi_x.I
		Sw=self.field_noise_variance*Psi_xinv* self.field.field_noise()*Psi_xinv.T
		Swc=pb.linalg.cholesky(Sw)		
		X=[]		
		x=init_field
		X.append(x)


		##simulation
		#states		
		for t in range(T):
			w = pb.matrix(np.random.randn(self.field.nx,1))
			sum_s=[]
			for s in self.field.space:
				sum_r=0		
				for r in self.field.space:			
					sum_r+=self.kernel(s-r)*self.act_fun(self.field.field_bases(r).T*x)
					#sum_r+=self.kernel(s-r)*self.field.field_bases(r).T*x
				sum_s.append(self.field.field_bases(s)*sum_r)
			sum_int=(self.stepsize**2)*sum(sum_s)
			x=Psi_xinv*sum_int-self.alpha*x+Swc*w
			X.append(x)
		#observations
		Y=[]
		Svc=pb.linalg.cholesky(self.obs_noise_covariance)
		for t in range(T):
			v = pb.matrix(np.random.randn(len(self.obs_locns),1))
			y=[]
			for s in self.obs_locns:
				sum_r=0
				for r in self.field.space:
					sum_r+=gaussian(s-r,0,1,1)*float((self.field.field_bases(r).T*X[t]))
				y.append(self.stepsize*sum_r)
			Y.append(pb.matrix(y).T+Svc*v)
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
		sum_s=[]
		for s in self.space:
			sum_r=0		
			for r in self.space:			
				sum_r+=gaussian(s-r,0,1,1)*self.field_bases(r).T
			sum_s.append(self.field_bases(s)*sum_r)		
		return (self.stepsize**2)*sum(sum_s)





	def plot(self,space):

		if self.dimension==1:
			y=[]
			z=[]
			specs=zip(self.weights,self.centers,self.widths)
			for j in specs:
				for i in space:
					z.append(float(j[0]*gaussian(i,j[1],j[2],1)))
				y.append(z)
				z=[]
		[pb.plot(space,y[i],'k') for i in range(self.nx)]
		pb.show()	

		if self.dimension==2:
			for i in range(len(centers)):
				for j in range(len(centers)):
					circle(j*15,i*15,self.width[0])	
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
		y=[]
		for i in space:
			y.append(self.__call__(i))		
		pb.plot(space,y,'k')
		pb.show()


		




class ActivationFunction():
	
	def __init__(self,threshold=0.1,max_firing_rate=1,slope=0.2):
		
		self.threshold = threshold
		self.max_firing_rate = max_firing_rate
		self.slope = slope
		
	def __call__(self,v):

		return self.max_firing_rate/(1+pb.exp(self.slope*(self.threshold-v)))





def gaussian(s, centre, width,dimension):
		dimension = dimension
		centre = pb.matrix(centre)
		centre.shape = (dimension,1)
		width = pb.matrix(width)
		return float(pb.exp(-(s-centre).T*width.I*(s-centre)))




if __name__ == "__main__":

	#-------------field--------------------
	min_field, max_field, inc = 1.8, 18, 1.8
	f_centers=pb.arange(min_field,max_field+inc,inc)
	nx=len(f_centers)
	f_widths=[1]*nx
	f_weights=[1]*nx
	stepsize=0.5	
	f_space = pb.arange(-5,25,stepsize)
	f=Field(f_weights,f_centers,f_widths,1,f_space,nx,stepsize)
	f.plot(f_space)
	#------------Kernel-------------
	k_centers=[-.5,0,.5]
	k_weights =[-1,1,.5]
	k_widths = [.1,.1,.1]
	k_space = pb.arange(-3,3,0.1)#This is just to plot the kernel and doesn't affect simulation
	k=Kernel(k_weights,k_centers,k_widths,1)
	k.plot(k_space)
	#-------Brain----------------
	alpha=.05
	field_noise_variance=0.8 
	act_func=ActivationFunction()
	#----------observations--------------------------
	obs_locns = pb.arange(0.5,21,1)
	obs_noise_covariance =.2*np.eye(len(obs_locns),len(obs_locns))	
	#----------initialasation----------------------------
	init_field=[0]*nx
	init_field[6]=1
	init_field=pb.matrix(init_field).T
	#--------------model and simulation------------------
	T=3
	model=IDE(k, f, act_func,alpha,field_noise_variance,obs_noise_covariance,obs_locns,stepsize)
	X,Y=model.sim(init_field,T)

