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
			for s in space:
				new_field[s] = sum([
					kernel(s,r)*act_fun(fields[-1][r]) for r in space
				])
			field_list.append(new_field)
		return fields[1:]

class Field():
	
	def __init__(self):
		self.vals = pb.zeros(space.shape)
	
	def __getitem__(self,s):
		return self.vals[s]
		
	def __setitem__(self,s,val):
		self.vals[s] = val
	
class Kernel():
	
	def __init__(self, weights, widths):
		self.weights = weights
		self.widths = widths
	
	def gaussian(self, x, centre, width):
		return pb.exp(-0.5*(x-centre)**2/width)
		
	
	def __call__(self,s,r):
			return pb.sum([
				w*self.gaussian(abs(s-r),0,c) 
				for w,c in zip(self.weights,self.widths)
			])
		
				
if __name__ == "__main__":
	k = Kernel([0.85,-0.8,0.1],[1,2,8])
	space = pb.arange(-10,10,0.1)
	y = [k(s,0) for s in space]
	print sum(y)
	pb.plot(space,y)
	pb.show()
	
	
		
		