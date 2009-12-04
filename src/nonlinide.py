import pylab as pb
import numpy as np
import matplotlib.axes3d

class IDE():
	
	def __init__(self, kernel, field, act_fun,alpha,field_noise_variance):
		self.kernel = kernel
		self.field = field
		self.act_fun = act_fun
		self.alpha=alpha
		self.field_noise_variance=field_noise_variance

	def sim(self,init_field,T,stepsize=.2):
		#This is to check, this bit can be done Analitically		
		Psi_x=(stepsize*sum([self.field.field_bases(s)*self.field.field_bases(s).T for s in self.field.space]))
		Psi_xinv=Psi_x.I
		Sw=self.field_noise_variance*Psi_xinv* self.field.field_noise()*Psi_xinv.T
		Swc=pb.linalg.cholesky(Sw)
				
		X=[]		
		x=init_field
		X.append(x)


		##simulation
		for t in range(T):
			w = np.random.randn(self.field.nx,1)
			sum_s=[]
			for s in self.field.space:
				sum_r=0		
				for r in self.field.space:			
					#sum_r+=self.kernel(s-r)*self.act_fun(self.field.field_bases(r).T*x)
					sum_r+=self.kernel(s-r)*self.field.field_bases(r).T*x
				sum_s.append(stepsize*sum_r)
			sum_int=0
			for i in range(len(self.field.space)):
				sum_int+=stepsize*self.field.field_bases(self.field.space[i])*sum_s[i]

			x=Psi_xinv*sum_int#-self.alpha*x+Swc*w
			X.append(x)
		return X



class Field():
	
	def __init__(self, weights,centers, widths, dimension,space,nx):

		self.nx=nx
		self.weights = weights
		self.widths=self.nx*widths
		self.centers=centers
		self.dimension=dimension
		self.space = pb.array(space)
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,self.dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		return float(self.evaluate(s))		

	def field_bases(self,s):
		return pb.matrix([gaussian(s,cen,wid,self.dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)]).T



	def field_noise(self,stepsize=.2):
		sum_s=[]
		for s in self.space:
			sum_r=0		
			for r in self.space:			
				sum_r+=gaussian(s-r,0,1,1)*self.field_bases(r).T
			sum_s.append(stepsize*sum_r)
		sum_int=0
		for i in range(len(self.space)):
			sum_int+=self.field_bases(self.space[i])*sum_s[i]
		return stepsize*sum_int





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
		if self.dimension==1:
			y=[]
			for i in space:
				y.append(self.__call__(i))		
			pb.plot(space,y,'k')
			pb.show()

		if self.dimension==2:
			y = pb.zeros((space.shape[0],space.shape[0]))
			for i in range(len(space)):
				for j in range(len(space)):
					y[i,j]=self.__call__(pb.matrix([[space[i]],[space[j]]]))			
			fig = pb.figure()
			ax = matplotlib.axes3d.Axes3D(fig)
			s1,s2=pb.meshgrid(space,space)
			ax.plot_wireframe(s1,s2,y,color='k')
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

def circle(x,y,r):
	u=np.linspace(0,2*np.pi,200)
	x0=np.zeros_like(u)
	y0=np.zeros_like(u)
	for i in range(len(u)):
		x0[i]=r*np.sin(u[i])+x
		y0[i]=r*np.cos(u[i])+y
	pylab.plot(x0,y0)
	pylab.axis([0,90,0,90])	


if __name__ == "__main__":

	#min_field, max_field, inc = 0, 80, 8
	min_field, max_field, inc = 1.8, 18, 1.8
	#f_centers=pb.arange(min_field,max_field+inc,inc)
	f_centers=pb.arange(min_field,max_field+inc,inc)
	#nx=11
	nx=len(f_centers)
	#f_widths=[5]
	f_widths=[1]*nx
	#f_weights=[1]*11
	f_weights=[1]*nx
	T=3	
	#f_space = pb.arange(-10,90,.1)
	f_space = pb.arange(-5,25,0.2)
	f=Field(f_weights,f_centers,f_widths,1,f_space,nx)

	#f.plot(f_space)

	#k_centers=pb.array([0,0,0])
	k_centers=[-.5,0,.5]
	#k_centers=[1]
	#k_weights = pb.array([.5,-.3,.05])
	k_weights =[-1,1,.5]
	#k_weights =[1]
	#k_widths = pb.array([1.8,36,18**2]) 
	k_widths = [.1,.1,.1]
	#k_widths = [.1]
	#k_space = pb.linspace(-40,40,1)
	k_space = pb.arange(-3,3,0.2)
	k=Kernel(k_weights,k_centers,k_widths,1)
	#k.plot(k_space)
	alpha=.05
	field_noise_variance=0.8 
	act_func=ActivationFunction()	
	model=IDE(k, f, act_func,alpha,field_noise_variance)
	init_field=[0]*nx
	init_field[6]=1
	init_field=pb.matrix(init_field).T

