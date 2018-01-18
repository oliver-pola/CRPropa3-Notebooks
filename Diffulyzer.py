from crpropa import *
from pylab import *
from mpl_toolkits.mplot3d import axes3d
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np


def larmor_radius(E, Brms):
	return E / (c_light * eplus * Brms)


# returns list of candidates
def create_candidates(num_candidates, EMin, EMax):
	source = Source()
	source.add(SourceParticleType(nucleusId(1, 1)))
	source.add(SourcePowerLawSpectrum(EMin, EMax, -1))
	source.add(SourceIsotropicEmission())
	source.add(SourcePosition(Vector3d(0, 0, 0)*Mpc))
	candidates = []
	for i in range(0,num_candidates):
		candidates.append(source.getCandidate())
	return candidates


# returns propagation module with B field
def create_prop(grid_size, grid_spacing, Brms, lMin, lMax, alpha):
	vgrid = VectorGrid(Vector3d(0), grid_size, grid_spacing)
	initTurbulence(vgrid, Brms, lMin, lMax, alpha)
	Bfield = MagneticFieldGrid(vgrid)
	prop = PropagationCK(Bfield)
	return prop


# simulates one B field setting, return data
def sim(num_candidates, num_samples, max_trajectory_length, EMin, EMax, grid_size, grid_spacing, Brms, lMin, lMax, alpha):
	sync_step = max_trajectory_length / num_samples
	candidates = create_candidates(num_candidates, EMin, EMax)
	prop = create_prop(grid_size, grid_spacing, Brms, lMin, lMax, alpha)
	data = np.empty([0,4])
	for s in range(1,num_samples+1):
		stop = MaximumTrajectoryLength(s * sync_step)
		for c in candidates:
			c.setActive(True);
			while c.isActive():
				prop.process(c.get())
				stop.process(c.get())
			pos = c.current.getPosition()
			data = np.append(data, [[s * sync_step, pos.getX(), pos.getY(), pos.getZ()]], axis=0)
	# data in Mpc
	data = data / Mpc;
	return data


def plot_trajectory(data):
	x, y, z = data[:,1], data[:,2], data[:,3]
	fig = plt.figure(figsize=(9, 5))#plt.figaspect(0.5))
	ax = fig.gca(projection='3d')# , aspect='equal'
	ax.scatter(x,y,z, 'o', lw=0)
	ax.set_xlabel('x [Mpc]', fontsize=18)
	ax.set_ylabel('y [Mpc]', fontsize=18)
	ax.set_zlabel('z [Mpc]', fontsize=18)
	ax.set_xlim((-10, 10))
	ax.set_ylim((-10, 10))
	ax.set_zlim((-10, 10))
	ax.xaxis.set_ticks((-10, -5, 0, 5, 10))
	ax.yaxis.set_ticks((-10, -5, 0, 5, 10))
	ax.zaxis.set_ticks((-10, -5, 0, 5, 10))
	show()
	return


def fit_func(x, a, b):
	return a*(x**b)


# also returns D, exponent
def plot_rms(data, plot_title=''):
	grouped = defaultdict(list)
	d = []
	r_sq = []
	d_end = 0
	for di, X, Y, Z in data:
		grouped[di].append(X**2 + Y**2 + Z**2)
	for di in grouped:
		d.append(di)
		r_sq.append(np.mean(grouped[di]))
		d_end = max(d_end, di)
	print('After last step:')
	print('r_min   = ' + str(sqrt(np.min(grouped[d_end]))))
	print('r_max   = ' + str(sqrt(np.max(grouped[d_end]))))
	r_avg = np.mean(sqrt(grouped[d_end]))
	print('< r >   = ' + str(r_avg))
	print('< r >²  = ' + str(r_avg**2))
	r_sq_avg = np.mean(grouped[d_end])
	print('< r² >  = ' + str(r_sq_avg))
	print('sigma_r = ' + str(sqrt(abs(r_sq_avg - r_avg**2))))

	popt, pcov = curve_fit(fit_func, d, r_sq, bounds=(0, [20.0, 2.0]))
	D = popt[0]
	exponent = popt[1]
	print()
	print('Over all data points:')
	print('D       = ' + str(D))
	print('exponent= ' + str(exponent))
	print()
	print('r²     ~= ' + str(D) + ' d ^ ' + str(exponent))

	#sort d and r_sq ordered by d, maybe needed on some system setup
	for j in range(len(d)-1):
		for i in range(len(d)-1):
			if(d[i]>d[i+1]):
				temp = d[i]
				d[i] = d[i+1]
				d[i+1] = temp
				temp2 = r_sq[i]
				r_sq[i] = r_sq[i+1]
				r_sq[i+1] = temp2

	figure(figsize=(9, 5))
	title(plot_title)
	plot(d, r_sq, 'b-')
	plot(d, fit_func(d, D, exponent), 'b:', label='%.3f d ^ %.3f' % tuple(popt))
	grid()
	ylabel('$<$r²$>$ [Mpc²]')
	xlabel('d [Mpc]')
	legend()
	show()
	return D, exponent


# returns D, exponent
def calc_diffusion(data):
	grouped = defaultdict(list)
	d = []
	r_sq = []
	d_end = 0
	for di, X, Y, Z in data:
		grouped[di].append(X**2 + Y**2 + Z**2)
	for di in grouped:
		d.append(di)
		r_sq.append(np.mean(grouped[di]))
		d_end = max(d_end, di)

	popt, pcov = curve_fit(fit_func, d, r_sq, bounds=(0, [20.0, 2.0]))
	D = popt[0]
	exponent = popt[1]
	return D, exponent
