import pylab as pb
import os

class IDE():
	
	def __init__(self, kernel, init_field, act_fun, space):
		self.kernel = kernel
		self.init_field = init_field
		self.act_fun = act_fun
		self.space = pb.array(space)
		
	def sim(self, T):
		fields = [self.init_field]
		space = self.space
		for t in range(T):
			new_field = Field(space)
			print t
			for si in range(len(space)):
				if si > 80 and si < 120:
					new_field[si] = -0.8*fields[-1](si) + sum([
						self.kernel(si,ri)*act_fun(fields[-1](ri)) for ri in range(len(space))
					]) + 0.1*pb.randn()
				else:
					new_field[si] = -0*fields[-1](si) + sum([
						self.kernel(si,ri)*act_fun(fields[-1](ri)) for ri in range(len(space))
					])
			fields.append(new_field)
		return fields[1:]

class Field():
	
	def __init__(self, space, smoothness=1):
		self.space = space
		self.vals = pb.zeros(space.shape)
		self.smoothness = smoothness
		
	def __call__(self,s):
			"""return pb.sum([
				v*gaussian(s,locn,self.smoothness) 
				for v,locn in zip(self.vals,self.space)
			])"""
			return self.vals[s]
			
	def __setitem__(self,si,val):
		self.vals[si] = val
	
class Kernel():
	
	def __init__(self, weights, widths, space):
		self.weights = weights
		self.widths = widths
		self.kernel_matrix = self.gen_matrix(space)
		u,v = pb.eig(self.kernel_matrix)
		if sum([abs(ui)>1 for ui in u]):
			print [abs(ui) for ui in u]
			#raise ValueError('unstable kernel')
	
	def gen_matrix(self, space):
		kernel_matrix = pb.zeros((space.shape[0],space.shape[0]))
		for si,s in enumerate(space):
			for ri,r in enumerate(space):
				kernel_matrix[si,ri] = pb.sum([
					w*gaussian(abs(s-r),0,c) 
					for w,c in zip(self.weights,self.widths)
				])
		return kernel_matrix
				
	def __call__(self,s,r):
			return self.kernel_matrix[s,r]
	
	def plot(self,space):
		y = [pb.sum([w*gaussian(abs(s-0),0,c) 
			for w,c in zip(self.weights,self.widths)]) for s in space]
		pb.plot(y)

def gaussian(x, centre, width):
	return pb.exp(-0.5*(x-centre)**2/width)

def anim_field_list(fields,f0,space,threshold=0.1):
	fname = '_tmp%03d.png'
	files=[]
	for i,f in enumerate([f0]+fields):
		pb.figure()
		pb.plot(space, f.vals)
		pb.plot(space, [threshold]*len(space),'r--')
		pb.ylim([-3,3])
		pb.savefig(fname%i)
		files.append(fname%i)
		pb.close()
	os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=5 \
		  		-ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o brainIDE.avi")
	# cleanup
	for fname in files: os.remove(fname)
	

class ActivationFunction():
	
	def __init__(self,threshold=0.1,max_firing_rate=1):
		
		self.threshold = threshold
		self.max_firing_rate = max_firing_rate
		
	def __call__(self,v):
		if v > self.threshold:
			return self.max_firing_rate
		else:
			return 0
				
if __name__ == "__main__":
	space = pb.arange(-10,10,0.1)
	weights = pb.array([4,-8,5])
	widths = pb.array([0.05,0.5,2])
	k = Kernel(0.008*weights, 5*widths, space)
	pb.figure()
	k.plot(space)
	pb.show()
	f = Field(space)
	f.vals[100] = 1
	act_fun = ActivationFunction()
	model = IDE(k,f,act_fun,space)
	T = 50
	fields = model.sim(T)
	pb.figure()
	anim_field_list(fields,f,space)
	
	
	
		
		