import pylab,numpy

class block_matrix():
	
	def __init__(self,matrix_array):
		self._data = matrix_array
	
	def __getitem__(self,index):
		return self._data[index]
		
	def __setitem__(self,index,A):
		self._data[index] = A
	
	def asmatrix(self):
		rows = [pylab.hstack(row) for row in self._data]
		return pylab.vstack(rows)

	def trace(self):
		temp=0
		for i in range(len(self._data)):
			temp+=self._data[i,i]
		return temp

	
	def transpose(self):
		'''This is to transpose both the matrix and its block components'''
		temp_matrix= pylab.empty((len(self._data),len(self._data)),dtype=object)
		for i in range(len(self._data)): 
			for j in range(len(self._data)): 
				temp_matrix[i,j]=self._data[i,j].T
		temp_matrix=temp_matrix.T
		return block_matrix(temp_matrix)

	

	def __mul__(self,A):
		temp_matrix= 0
		if hasattr(A,'_data'):A=A._data
		temp_matrix=self._data*A
		return block_matrix(temp_matrix)

	def __rmul__(self,A):
		temp_matrix= 0
		if hasattr(A,'_data'):A=A._data
		temp_matrix=A*self._data
		return block_matrix(temp_matrix)


def directsum(*arg):
	#Check if all matrices are square
	for i in range(len(arg)):
		assert arg[i].shape[0]==arg[i].shape[1], 'Matrix '+str(arg[i])+ ' is not square.'
	#Finding the shape of the Matrix result of the directsum operation
	#-----------------------------------------------------------------	
	temp=(0,0)
	for i in range(len(arg)):
		temp=numpy.add(temp,arg[i].shape)		
	dsum = numpy.matrix(numpy.zeros(temp))


	#Place the first Matrix
	#--------------------------
	dsum[:arg[0].shape[0],:arg[0].shape[1]]=arg[0]

	#Place the rest of matrices
	#--------------------------	
	rc_Indices=0 #This is the starting row and column indices for each matrix
	for i in range(1,len(arg)):	
		rc_Indices+=arg[i-1].shape[0]
		dsum[rc_Indices:rc_Indices+arg[i].shape[0],rc_Indices:rc_Indices+arg[i].shape[1]]=arg[i]	
	return dsum


def hslice(Input_matrix,*arg):
	'''This function creats slices of Input_matrix acording to the number of rows
	for each slice in arg, Input_matrix should be square and also args
        should add up to the Input_matrix dimension.'''
	#This is to make sure Input_matrix is aquare and args sum up to Input_matrix dimension 
	#-------------------------------------------------------------------------------------		
	assert Input_matrix.shape[0]==Input_matrix.shape[1], 'Matrix is not squre'
	assert sum(arg)==Input_matrix.shape[0], 'Dimension mismatch'

	temp=[0]*len(arg)

	#Creating the first slice
	#--------------------------
	temp[0]=Input_matrix[0:arg[0]]

	#Creating the rest of slices
	#---------------------------
	index=0
	for i in range(1,len(arg)):
		index+=arg[i-1]
		temp[i]=Input_matrix[index:index+arg[i]]
	return temp
		


def Matrixlist_product(MatrixList1,MatrixList2):
	'''This function takes two list of Matrices and multiply
	them elementwise and gather the result in another list'''
	assert type(MatrixList1) is list, 'the inputs must be in a list'
	assert type(MatrixList2) is list, 'the inputs must be in a list'
	assert len(MatrixList1)==len(MatrixList2), 'Matrix lists must have equal lengths'
	Product_temp=[0] *len(MatrixList1)
	for i in range(len(MatrixList1)):
		assert numpy.shape(MatrixList1[i])[1]==numpy.shape(MatrixList2[i])[0], 'Matrices with index '+str(i)+' are not aligned'
		Product_temp[i]=MatrixList1[i]*MatrixList2[i]
	return Product_temp
	



if __name__=="__main__":

	
	#A = pylab.empty((2,2),dtype=object)
	
	#A[0,0] = pylab.matrix([[1,2],[3,4]])
	#A[0,1] = 2*pylab.matrix([[1,2],[3,4]])
	#A[1,0] = 4*pylab.matrix([[1,2],[3,4]])
	#A[1,1] = 3*pylab.matrix([[1,2],[3,4]])

	A=numpy.matrix([[1,2,3],[4,5,6],[7,8,9]])

	U = pylab.empty((3,3),dtype=object)
	
	U[0,0] = pylab.matrix([[1,2]])
	U[0,1] = pylab.matrix([[3,4]])
	U[0,2] = pylab.matrix([[5,6]])
	U[1,0] = pylab.matrix([[7,8]])
	U[1,1] = pylab.matrix([[9,10]])
	U[1,2] = pylab.matrix([[11,12]])
	U[2,0] = pylab.matrix([[13,14]])
	U[2,1] = pylab.matrix([[15,16]])
	U[2,2] = pylab.matrix([[17,18]])
	
	B = block_matrix(U)
	C = B.asmatrix()
	print '\n################################Matrix################################'	
	print '\nMatrix B is:\n',C
	print '\n B_(1,0) is :',B[1,0]
	
	#Trace:
	print '\n################################Trace################################'	
	print 'Trace of B is: \n\n',B.trace()

        
	#Transpose Type2
	print '\n################################Tanspose Type II################################'	
	T=B.transpose()
	print '\nTranspose type II of B is: \n', T.asmatrix()
	 
	# Block matrix and ordinary matrix product
	print '\n################################Product################################'	
	E=A*B
	print '\nA=',A, 
	print '\nB=',B.asmatrix()
	print '\nA*B=',E.asmatrix()
	#print type(C)
	
	
