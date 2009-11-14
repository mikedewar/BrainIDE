"""base classes of the integrodifference equation.

This module is quite light and basically just glues bases together. The file
bases.py defines individual basis functions and their respective functionality.
"""
import numpy as np
import pylab as pb
class ideBase:
	"""class defining common properties of the basis, kernel and field"""
	def __str__(self):
		return self.name
	def plot(self,space):
		if self.dimension==1:
			y = np.array([self.evaluate(s) for s in space])
			pb.plot(space,y[:,0])
			pb.show()

class ideElement(ideBase):
	"""class defining common properties of the kernel and field"""
	
	def __init__(self,dimension,bases,weights):
		self.dimension = dimension
		assert type(bases) is list, 'the bases must be in a list (even if there is only one)'
		# check each basis function object has the necessary attributes
		for b in bases:
			assert hasattr(b,'__mul__'), 'the basis function must have a __mul__ attribute'
			assert hasattr(b,'conv'), 'the basis function must have a conv attribute'
			assert hasattr(b,'evaluate'), 'the basis function must have an evaluate attribute'
		self.bases = bases
		if np.isscalar(weights):
			weights=[weights]
		assert type(weights) is list, 'the weights must be in a list'
		self.weights = weights
		self.evaluate = lambda s: sum([self.bases[i].evaluate(s)*self.weights[i] for i in range(len(bases))])	
	

class kernel(ideElement):
	"""class defining a spatial mixing kernel"""
	
	def __init__(self,dimension,bases,weights):
		ideElement.__init__(self,dimension,bases,weights)
		self.name = "kernel"

class field(ideElement):
	"""class defining a spatial field"""
	
	def __init__(self,dimension,bases,weights,name):
		ideElement.__init__(self,dimension,bases,weights)
		self.name = name
		





