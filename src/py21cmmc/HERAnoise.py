import numpy as np
import sys
import scipy
from glob import glob
from time import time, sleep
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from tools21cm.usefuls import *
from tools21cm import cosmology as cm
from tools21cm import conv
from tools21cm.telescope_functions import *
import tools21cm as t21c
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera
KB_SI   = 1.38e-23
c_light = 2.99792458e+10  #in cm/s
janskytowatt = 1e-26

def kn_noise_image_hera(z, uv_map, depth_mhz, obs_time, int_time, N_ant_hera=331., verbose=True):
	"""
	@ Kanan Datta
	
	It calculates the rms of the noise added by the interferrometers of ska. 
	Parameters
	----------
	z         : float
		Redhsift of the slice observed.
	uv_map    : ndarray
		ncells x ncells numpy array containing the number of baselines observing each pixel.
	depth_mhz : float
		The bandwidth of the observation (in MHz).
	obs_time  : float
		The total hours of observations time.
	N_ant_ska : float
		Number of anntennas in SKA. Default: 564.
	Returns
	-------
	sigma     : float
		The rms of the noise in the image produced by SKA for uniformly distributed antennas.
	rms_noise : float
		The rms of the noise due to the antenna positions in uv field.
	"""
	z = float(z)
	nuso  = 1420.0/(1.0 + z)
	delnu = depth_mhz*1e3	                                            # in kHz
	effective_baseline = np.sum(uv_map)
	# Standard definition of sky temperature
	T_sky_atnu300MHz= 60.0                                              #K
	# Koopmans et al. (2015) definition of sky temperature
	# T_sky_atnu300MHz= 68.3  					    #K
	T_sky = T_sky_atnu300MHz*(300.0/nuso)**2.55
	# Koopmans et al. 
	# Receiver temperature
	T_rcvr = 100                                                         #K
	T_sys  = T_sky + T_rcvr
	ant_radius_hera  = 14./2. 	                                    #in m
	nu_crit = 1.1e5 						    # in kHz
	if nuso>nu_crit: ep = (nu_crit/nuso)**2
	else: ep = 1. 	
	A_ant_hera = ep*np.pi*ant_radius_hera*ant_radius_hera
	sigma     = np.sqrt(2.0)*KB_SI*(T_sys/A_ant_hera)/np.sqrt((depth_mhz*1e6)*(obs_time*3600.0))/janskytowatt*1e3/np.sqrt(N_ant_hera*N_ant_hera/2.0) ## in mJy
	rms_noi  = 1e6*np.sqrt(2)*KB_SI*T_sys/A_ant_hera/np.sqrt(depth_mhz*1e6*int_time)/janskytowatt #in muJy
	sigma    = rms_noi/np.sqrt(N_ant_hera*(N_ant_hera-1)/2.0)/np.sqrt(3600*obs_time/int_time)      #in muJy
	if verbose:
		print('\nExpected: rms in image in muJy per beam for full =', sigma)
		print('Effective baseline =', sigma*np.sqrt(N_ant_hera*N_ant_hera/2.0)/np.sqrt(effective_baseline), 'm')
		print('Calculated: rms in the visibility =', rms_noi, 'muJy')
	return sigma, rms_noi


def noise_map_hera(ncells, z, depth_mhz, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=38., uv_map=np.array([]), N_ant=None, verbose=True, fft_wrap=False):
	"""
	@ Ghara et al. (2017), Giri et al. (2018b)

	It creates a noise map by simulating the radio observation strategy.

	Parameters
	----------
	z: float
		Redshift.
	ncells: int
		The grid size.
	depth_mhz: float
		The bandwidth in MHz.
	obs_time: float
		The observation time in hours.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	uv_map: ndarray
		numpy array containing gridded uv coverage. If nothing given, then the uv map 
		will be simulated
	N_ant: int
		Number of antennae
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	noise_map: ndarray
		A 2D slice of the interferometric noise at that frequency (in muJy).
	"""
	#if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not uv_map.size: uv_map, N_ant  = get_uv_map_hera(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	sigma, rms_noi = kn_noise_image_hera(z, uv_map, depth_mhz, obs_time, int_time, N_ant_hera=N_ant, verbose=False)
	noise_real = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_imag = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_arr  = noise_real + 1.j*noise_imag
	noise_four = t21c.apply_uv_response_noise(noise_arr, uv_map)
	if fft_wrap: noise_map  = t21c.ifft2_wrap(noise_four)*np.sqrt(int_time/3600./obs_time)
	else: noise_map  = np.fft.ifft2(noise_four)*np.sqrt(int_time/3600./obs_time)
	return np.real(noise_map)

def get_uv_map_hera(ncells, z, filename=None, total_int_time=6., int_time=10., boxsize=None, declination=38., verbose=True):
	"""
	Parameters
	----------
	ncells: int
		Number of cells
	z: float
		Redshift.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	filename: str
		The path to the file containing the telescope configuration.
			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc	
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	uv_map: ndarray
		array of gridded uv coverage.
	N_ant: int
		Number of antennae
	"""
	if not filename: N_ant = 331
	uv_map, N_ant  = get_uv_daily_observation_hera(ncells, z, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
	return uv_map, N_ant

def get_uv_daily_observation_hera(ncells, z, Nbase=None,N_ant=331, total_int_time=4., int_time=10., boxsize=None, declination=38., include_mirror_baselines=False, verbose=True):
	"""
	The radio telescopes observe the sky for 'total_int_time' hours each day. The signal is recorded 
	every 'int_time' seconds. 
	Parameters
	----------
	ncells         : int
		The number of cell used to make the image.
	z              : float
		Redhsift of the slice observed.
	filename       : str
		Name of the file containing the antenna configurations (text file).
	total_int_time : float
		Total hours of observation per day (in hours).
	int_time       : float
		Integration time of the telescope observation (in seconds).
	boxsize        : float
		The comoving size of the sky observed. Default: It is determined from the simulation constants set.
	declination    : float
		The declination angle of the SKA (in degree). Default: 30. 
	Returns
	-------
	(uv_map, N_ant)
	"""
	z = float(z)
	#if 'numba' in sys.modules: 
	#	from .numba_functions import get_uv_daily_observation_numba
	#	uv_map, N_ant = get_uv_daily_observation_numba(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
	#	return uv_map, N_ant
	if Nbase is None:
		print("get n base.")
		Nbase=get_nbase(z)
	uv_map0      = get_uv_coverage(Nbase, z, ncells, boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)
	uv_map       = np.zeros(uv_map0.shape)
	tot_num_obs  = int(3600.*total_int_time/int_time)
	if verbose: 
		print("Making uv map from daily observations.")
		time.sleep(5)
	for i in tqdm(range(tot_num_obs-1)):
		new_Nbase = earth_rotation_effect(Nbase, i+1, int_time, declination=declination)
		uv_map1   = get_uv_coverage(new_Nbase, z, ncells, boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)
		uv_map   += uv_map1
		# if verbose:
		# 	perc = int((i+2)*100/tot_num_obs)
		# 	msg = '%.1f %%'%(perc)
		# 	loading_verbose(msg)
	uv_map = (uv_map+uv_map0)/tot_num_obs
	print('...done')
	return uv_map, N_ant

def get_nbase(z):
    nu=1420/(1+z)
    sensitivity = PowerSpectrum(
        observation = Observation(
            observatory = Observatory(
                antpos = hera(hex_num=11, separation=14, dl=12.12),
                beam = GaussianBeam(frequency=nu, dish_size=14),
                latitude=38*np.pi/180.0
            )
        )
    )
    observatory = sensitivity.observation.observatory
    baseline=observatory.baselines_metres.reshape(-1,3)
    baseline=np.array(baseline/(c_light/(nu*1e6)/1e2))
    return baseline

def get_uv_coverage(Nbase, z, ncells, boxsize=None, include_mirror_baselines=False):
	"""
	It calculated the uv_map for the uv-coverage.
	Parameters
	----------
	Nbase   : ndarray
		The array containing all the ux,uy,uz values of the antenna configuration.
	z       : float
		Redhsift of the slice observed.
	ncells  : int
		The number of cell used to make the image.
	boxsize : float
		The comoving size of the sky observed. Default: It is determined from the simulation constants set.
	Returns
	-------
	uv_map  : ndarray
		ncells x ncells numpy array containing the number of baselines observing each pixel.
	"""
	z = float(z)
	if not boxsize: boxsize = conv.LB
	uv_map = np.zeros((ncells,ncells))
	theta_max = boxsize/cm.z_to_cdist(z)
	Nb  = np.round(Nbase*theta_max)
	Nb  = Nb[(Nb[:,0]<ncells/2)]
	Nb  = Nb[(Nb[:,1]<ncells/2)]
	#Nb  = Nb[(Nb[:,2]<ncells/2)]
	Nb  = Nb[(Nb[:,0]>=-ncells/2)]
	Nb  = Nb[(Nb[:,1]>=-ncells/2)]
	#Nb  = Nb[(Nb[:,2]>=-ncells/2)]
	xx,yy,zz = Nb[:,0], Nb[:,1], Nb[:,2]
	for p in range(xx.shape[0]): 
		uv_map[int(xx[p]),int(yy[p])] += 1
		if include_mirror_baselines: uv_map[-int(xx[p]),-int(yy[p])] += 1
	if include_mirror_baselines: uv_map /= 2
	return uv_map

def noise_cube_coeval_hera(ncells, z, depth_mhz=None, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=38., uv_map=np.array([]), N_ant=None, verbose=True, fft_wrap=False):
	"""
	@ Ghara et al. (2017), Giri et al. (2018b)
	It creates a noise coeval cube by simulating the radio observation strategy.
	Parameters
	----------
	ncells: int
		The grid size.
	z: float
		Redshift.
	depth_mhz: float
		The bandwidth in MHz.
	obs_time: float
		The observation time in hours.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	uv_map: ndarray
		numpy array containing gridded uv coverage. If nothing given, then the uv map 
		will be simulated
	N_ant: int
		Number of antennae
	filename: str
		The path to the file containing the telescope configuration.
			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	noise_cube: ndarray
		A 3D cube of the interferometric noise (in mK).
		The frequency is assumed to be the same along the assumed frequency (last) axis.	
	"""
	if not filename: N_ant = 331
	if not boxsize: boxsize = conv.LB
	if not depth_mhz: depth_mhz = (cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)-boxsize/2))-cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)+boxsize/2)))/ncells
	if not uv_map.size: uv_map, N_ant  = get_uv_map_hera(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	noise3d = np.zeros((ncells,ncells,ncells))
	if verbose: print("Creating the noise cube...")
	sleep(1)
	for k in tqdm(range(ncells), disable=False if verbose else True):
		noise2d = noise_map_hera(ncells, z, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, verbose=verbose, fft_wrap=fft_wrap)
		noise3d[:,:,k] = noise2d
		verbose = False
		# perc = np.round((k+1)*100/ncells, decimals=1) 
		# loading_verbose(str(perc)+'%')
	if verbose: print("...noise cube created.")
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)