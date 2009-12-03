import pylab as pb
import matplotlib.axes3d



class Field():
	
	def __init__(self, weights,centers, widths, dimension,N):

		self.N=N
		self.weights = weights
		self.widths=N*widths
		self.centers=centers
		self.dimension=dimension
		
		self.evaluate = lambda s: sum([w*gaussian(s,cen,wid,dimension) for w,cen,wid, in zip(self.weights,self.centers,self.widths)
				])

	def __call__(self,s):
		return float(self.evaluate(s))		


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
		[pb.plot(space,y[i],'k') for i in range(self.N)]
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
		return float(self.evaluate(s))


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
		return pb.exp(-(s-centre).T*width.I*(s-centre))

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

	min_field, max_field, inc = 0, 80, 8
	f_centers=pb.arange(min_field,max_field+inc,inc)
	f_widths=[5]
	f_weights=[1]*11
	N=11	
	f=Field(f_weights,f_centers,f_widths,1,11)
	f_space = pb.linspace(-10,90,500)
	f.plot(f_space)

	k_centers=pb.array([0,0,0])
	k_weights = pb.array([.5,-.3,.05])
	k_widths = pb.array([1.8,36,18**2]) 
	k_space = pb.linspace(-40,40,500)
	k=Kernel(k_weights,k_centers,k_widths,1)
	k.plot(k_space)
	



