import pylab as pb

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
			new_field = field()
			for si,s in enuermate(space):
				new_field[si] = sum([
					kernel(s,r)*act_fun(fields[-1](r)) for r in space
				])
			field_list.append(new_field)
		return fields[1:]

class Field():
	
	def __init__(self, space, smoothness=1):
		self.space = space
		self.vals = pb.zeros(space.shape)
		self.smoothness = smoothness
		
	def __call__(self,s):
			return pb.sum([
				v*gaussian(s,locn,self.smoothness) 
				for v,locn in zip(self.vals,self.space)
			])
			
	def __setitem__(self,si,val):
		self.vals[si] = val
		print self.vals
	
class Kernel():
	
	def __init__(self, weights, widths):
		self.weights = weights
		self.widths = widths
	
	def __call__(self,s,r):
			return pb.sum([
				w*gaussian(abs(s-r),0,c) 
				for w,c in zip(self.weights,self.widths)
			])

def gaussian(x, centre, width):
	return pb.exp(-0.5*(x-centre)**2/width)
				
if __name__ == "__main__":
	space = pb.arange(-10,10,0.1)
	k = Kernel([0.85,-0.8,0.1], [1,2,8])
	f = Field(space)
	f[100] = 1
	print f.vals
	
	pb.plot([f(s) for s in space])
	pb.show()
	
	
		
		