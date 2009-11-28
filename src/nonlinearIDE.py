class IDE():
	
	def __init__(self, kernel, init_field, act_fun, space):
		
		self.kernel = kernel
		self.init_field = init_field
		self.act_fun = act_fun
		self.space = space
		
	def sim(self, T):
		fields = [self.init_field]
		space = self.space
		for t in range(T):
			new_field = field()
			for s in space:
				new_field(s) = sum([kernel(s,r)*act_fun(fields[-1](r)) for r in space])
			field_list.append(new_field)
		return fields[1:]


				