import pylab as pb
import matplotlib.axes3d






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
			pb.plot(space,y)
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




