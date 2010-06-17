import cPickle, os

def write(filename, data,mode, overwrite=False):
	assert isinstance(filename, str), "The file name is required to be of type string.";
	assert isinstance(mode, str), "The mode is required to be of type string";
	assert isinstance(data, dict), "The input data is required to be a dictionary."
	assert not os.path.exists(filename) or overwrite, "The file already exists. To overwrite, set overwrite flag to True";
	
	# File exists. Delete file first
	if os.path.exists(filename):
		os.remove(filename);
	
	try:
		myFile = open(filename, mode);
		cPickle.dump(data, myFile);
		myFile.close();
		return True;
	except:
		print "An error occurred when attempting to write to the file";
		return False;

def read(filename):
	assert isinstance(filename, str), "The file name is required to be of type string";
	assert os.path.exists(filename), "The file to read does not exist.";
	
	try:
		myFile = open(filename, 'r');
		output = cPickle.load(myFile);
		myFile.close();
		return output;
	except:
		print "An error occurred when attempting to read the file";
		return False;

def writed(filename,mode, *arg):
	assert isinstance(filename, str), "The file name is required to be of type string";
	assert isinstance(mode, str), "The mode is required to be of type string";
	# File exists
	if os.path.exists(filename):
		myFile= open(filename, 'r');	
		output = cPickle.load(myFile);
 		n=len(output.keys())
		myFile.close();		
		for i, j in enumerate(arg):
			output['var' + str(i+n)] = j;
		return write(filename,output,mode, True)
	else:
		data = dict();
		for i, j in enumerate(arg):
			data['var' + str(i)] = j;
		return write(filename, data,mode, True);
	
