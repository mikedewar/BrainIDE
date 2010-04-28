from __future__ import division
import pylab as pb
import numpy as np
from scipy import signal
import scipy as sp

def obs_frequency_response(Y,spacestep,ignore_at_zero=False,vmin=None,vmax=None,save=0,filename='filename'):

	'''Generates average FFT of all observed surface
	Argument:
	---------
		Y: list of matrix
			all observed surface
		spacestep: float
			distance beetween adjacent sensors

		save: zero or one
			if one it saves the image in the working directory
	Return:
	-------
		average magnitude of the fft of the observation field: ndarray

		freq: ndarray
			the frequency range over which fft2 is taken
	'''


	Y_matrix=[]
	for i in range(len(Y)):
    		Y_matrix.append(pb.hstack(pb.split(Y[i],pb.sqrt(len(Y[0])))).T)
	y=0

	for i in range(len(Y_matrix)):
		Y_temp=pb.detrend_mean(Y_matrix[i])
		#y=y+20*pb.log10((spacestep**2)*pb.absolute(signal.fft2(Y_matrix[i])))
		y=y+20*pb.log10(pb.absolute(signal.fft2(Y_temp)))
	y=y/len(Y_matrix)
	freq=pb.fftfreq(y.shape[0],float(spacestep)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	#Nyquist_freq=freq.max() #find the Nyquist frequency it is not exact (depending y.shape[0] is odd or even) but close to Nyquist frequency
	#freq=pb.fftshift(freq)
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	params = {'axes.labelsize': 15,'text.fontsize': 15,'legend.fontsize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15}
	pb.rcParams.update(params) 
	#pb.imshow((y*scale),origin='lower',extent=[freq[0],freq[-1],freq[0],freq[-1]],interpolation='nearest',vmin=vmin,vmax=vmax)#,cmap=pb.cm.gray,interpolation='nearest',cmap=pb.cm.gray,
	if ignore_at_zero: y=y[1:,1:]
	pb.imshow(y,origin='lower',extent=[0,freq_range,0,freq_range],vmin=vmin,vmax=vmax,cmap=pb.cm.hot)
	pb.colorbar(shrink=.7,orientation='vertical')
	pb.xlabel('Hz',fontsize=25)
	pb.ylabel('Hz',fontsize=25)
	if save:
		pb.savefig(filename+'.pdf',dpi=300)

	pb.show()	
	return pb.flipud(y),freq[1:]



def field_frequency_response(V_matrix,spacestep,ignore_at_zero=False,vmin=None,vmax=None,save=0,filename='filename'):

	'''Generates  FFT of the spatial kernel
	Argument:
	---------
		K: matrix
			spatial kernel
		spacestep: float
			distance beetween adjacent spatial point

		save: zero or one
			if one it saves the image in the working directory
	Return:
	-------
		 magnitude of the fft of the spatial kernel: ndarray

		freq: ndarray
			the frequency range over which fft2 is taken
	'''
	
	V_f=0
	for i in range(len(V_matrix)):
		V_temp=pb.detrend_mean(V_matrix[i])
		#V_f=V_f+20*pb.log10((spacestep**2)*pb.absolute(signal.fft2(V_temp)))
		V_f=V_f+20*pb.log10(pb.absolute(signal.fft2(V_temp)))

	V_f=V_f/len(V_matrix)
	freq=pb.fftfreq(V_f.shape[0],float(spacestep)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	params = {'axes.labelsize': 15,'text.fontsize': 15,'legend.fontsize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15}
	pb.rcParams.update(params) 
	if ignore_at_zero: V_f=V_f[1:,1:]
	fig = pb.figure(figsize=(6,6))
	pb.imshow(V_f,origin='lower',extent=[0,freq_range,0,freq_range],cmap=pb.cm.hot,vmin=vmin,vmax=vmax)
	pb.colorbar(shrink=.7,orientation='vertical')
	pb.xlabel('Hz',fontsize=25)
	pb.ylabel('Hz',fontsize=25)

	if save:
		pb.savefig(filename+'.pdf',dpi=300)

	pb.show()	
	return pb.flipud(V_f),freq[1:]


def kernel_frequency_response(K,spacestep,deterend=False,ignore_at_zero=False,save=0,vmin=None,vmax=None,filename='filename'):

	'''Generates  FFT of the spatial kernel
	Argument:
	---------
		K: matrix
			spatial kernel
		spacestep: float
			distance beetween adjacent spatial point

		deterend: Bool
			if True deterend the signal


		ignore_at_zero: Bool
			if True ignore the power at zero frequency


		save: zero or one
			if one it saves the image in the working directory
	Return:
	-------
		 magnitude of the fft of the spatial kernel: ndarray

		freq: ndarray
			the frequency range over which fft2 is taken
	'''

	if deterend: K=pb.detrend_mean(K)
	#k_f=20*pb.log10((spacestep**2)*pb.absolute(signal.fft2(K)))
	k_f=20*pb.log10(pb.absolute(signal.fft2(K)))
	freq=pb.fftfreq(k_f.shape[0],float(spacestep)) #generates the frequency array [0,pos,neg] over which fft2 is taken
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	#Nyquist_freq=freq.max() #find the Nyquist frequency it is not exact (depending y.shape[0] is odd or even) but close to Nyquist frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	if ignore_at_zero: k_f=k_f[1:,1:]
	params = {'axes.labelsize': 20,'text.fontsize': 20,'legend.fontsize': 10,'xtick.labelsize': 20,'ytick.labelsize': 20}
	pb.rcParams.update(params) 
	pb.imshow(k_f,origin='lower',extent=[0,freq_range,0,freq_range],cmap=pb.cm.hot,vmin=vmin,vmax=vmax)#,cmap=pb.cm.grayinterpolation='nearest',
	pb.xlabel('Hz',fontsize=25)
	pb.ylabel('Hz',fontsize=25)
	pb.colorbar(shrink=.7,orientation='vertical')

	if save:
		pb.savefig(filename+'.pdf',dpi=300)

	pb.show()
	#return power
	return pb.flipud(k_f),freq

def frequency_response_diag(fft_matrix,spacestep,dbline=0,hline=None,vline=None,color='k'):

	'''Plot the diagonal part of the frequency resose obtained from  obs_frequency_response, field_frequency_response or kernel_frequency_response

	Arguments:
	----------
		fft_matrix: the frequency response from obs_frequency_response, field_frequency_response or kernel_frequency_response
		spacestep:float
			distance beetween adjacent spatial point
		hline:3db attenuation, 3db under The maximum
		vline: Cut off frequency'''

	
	#fft_matrix and freq are the output of obs_frequency_response,field_frequency_response,kernel_frequency_response
	Sampling_frequency=1./spacestep
	Nyquist_freq=Sampling_frequency/2 #find the Nyquist frequency which is half a smapling frequency
	freq_range=2*Nyquist_freq #the frequency range over which fft2 is taken
	fft_matrix_diag=pb.diag(pb.flipud(fft_matrix)) #Finding the diagonal of the V_f
	freq=pb.linspace(0,freq_range,len(fft_matrix_diag))
	if dbline:
		pb.vlines(vline,ymin=fft_matrix_diag.min(),ymax=hline,linestyle='dashed',linewidth=2)
		pb.hlines(hline,xmin=0,xmax=vline,linestyle='dashed',linewidth=2)
		pb.axis([0,freq.max(),fft_matrix_diag.min(),fft_matrix_diag.max()])
		pb.text(0.8,-20,r"$\nu_c=10$",fontsize=20)
	pb.xlabel('Hz',fontsize=20)
	pb.ylabel('(db)',fontsize=20)
	pb.plot(freq,fft_matrix_diag,color,linewidth=2) 	

