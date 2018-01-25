from crpropa import *
from pylab import *
from mpl_toolkits.mplot3d import axes3d
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np


def larmor_radius(E, Brms):
	return E / (c_light * eplus * Brms)


def B_from_larmor(E, r_L):
	return E / (c_light * eplus * r_L)


def E_from_larmor(Brms, r_L):
	return r_L * c_light * eplus * Brms


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


# returns propagation module with combined B field
def create_prop(grid_size, grid_spacing, Brms, lMin, lMax, alpha, B_z = 0):
	vgrid = VectorGrid(Vector3d(0), grid_size, grid_spacing)
	initTurbulence(vgrid, Brms, lMin, lMax, alpha)
	BField = None
	if B_z > 0:
		BField = MagneticFieldList()
		UniformField = UniformMagneticField(Vector3d(0,0,1) * B_z)
		Turbulentfield = MagneticFieldGrid(vgrid)
		BField.addField(UniformField)
		BField.addField(Turbulentfield)
	else:
		BField = MagneticFieldGrid(vgrid)
	prop = PropagationCK(BField)
	return prop


# simulates combined B field setting, return data
def sim(num_candidates, num_samples, max_trajectory_length, EMin, EMax, grid_size, grid_spacing, Brms, lMin, lMax, alpha, B_z = 0):
	sync_step = max_trajectory_length / num_samples
	candidates = create_candidates(num_candidates, EMin, EMax)
	prop = create_prop(grid_size, grid_spacing, Brms, lMin, lMax, alpha, B_z)
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


def plot_rms(data, plot_title=''):
	return plot_rsq(data, plot_title)


# also returns D, exponent
def plot_rsq(data, plot_title=''):
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

	popt, pcov = curve_fit(fit_func, d, r_sq, bounds=(0, [30.0, 3.0]))
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


# also returns D, exponent parralel to z
def plot_rsq_par(data, plot_title=''):
	grouped = defaultdict(list)
	d = []
	z_sq = []
	d_end = 0
	for di, X, Y, Z in data:
		grouped[di].append(Z**2)
	for di in grouped:
		d.append(di)
		z_sq.append(np.mean(grouped[di]))
		d_end = max(d_end, di)
	print('After last step:')
	print('z_min   = ' + str(sqrt(np.min(grouped[d_end]))))
	print('z_max   = ' + str(sqrt(np.max(grouped[d_end]))))
	z_avg = np.mean(sqrt(grouped[d_end]))
	print('< z >   = ' + str(z_avg))
	print('< z >²  = ' + str(z_avg**2))
	z_sq_avg = np.mean(grouped[d_end])
	print('< z² >  = ' + str(z_sq_avg))
	print('sigma_z = ' + str(sqrt(abs(z_sq_avg - z_avg**2))))

	popt, pcov = curve_fit(fit_func, d, z_sq, bounds=(0, [30.0, 3.0]))
	D_par = popt[0]
	exp_par = popt[1]
	print()
	print('Over all data points:')
	print('D_par   = ' + str(D_par))
	print('exp_par = ' + str(exp_par))
	print()
	print('z²     ~= ' + str(D_par) + ' d ^ ' + str(exp_par))

	#sort d and r_sq ordered by d, maybe needed on some system setup
	for j in range(len(d)-1):
		for i in range(len(d)-1):
			if(d[i]>d[i+1]):
				temp = d[i]
				d[i] = d[i+1]
				d[i+1] = temp
				temp2 = z_sq[i]
				z_sq[i] = z_sq[i+1]
				z_sq[i+1] = temp2

	figure(figsize=(9, 5))
	title(plot_title)
	plot(d, z_sq, 'b-')
	plot(d, fit_func(d, D_par, exp_par), 'b:', label='%.3f d ^ %.3f' % tuple(popt))
	grid()
	ylabel('$<$z²$>$ [Mpc²]')
	xlabel('d [Mpc]')
	legend()
	show()
	return D_par, exp_par


# also returns D, exponent orthogonal to z
# R is defined: R = sqrt(x² + y²)
def plot_rsq_ort(data, plot_title=''):
	grouped = defaultdict(list)
	d = []
	R_sq = []
	d_end = 0
	for di, X, Y, Z in data:
		grouped[di].append(X**2 + Y**2)
	for di in grouped:
		d.append(di)
		R_sq.append(np.mean(grouped[di]))
		d_end = max(d_end, di)
	print('After last step:')
	print('R_min   = ' + str(sqrt(np.min(grouped[d_end]))))
	print('R_max   = ' + str(sqrt(np.max(grouped[d_end]))))
	R_avg = np.mean(sqrt(grouped[d_end]))
	print('< R >   = ' + str(R_avg))
	print('< R >²  = ' + str(R_avg**2))
	R_sq_avg = np.mean(grouped[d_end])
	print('< R² >  = ' + str(R_sq_avg))
	print('sigma_R = ' + str(sqrt(abs(R_sq_avg - R_avg**2))))

	popt, pcov = curve_fit(fit_func, d, R_sq, bounds=(0, [30.0, 3.0]))
	D_ort = popt[0]
	exp_ort = popt[1]
	print()
	print('Over all data points:')
	print('D_ort   = ' + str(D_ort))
	print('exp_ort = ' + str(exp_ort))
	print()
	print('R²     ~= ' + str(D_ort) + ' d ^ ' + str(exp_ort))

	#sort d and r_sq ordered by d, maybe needed on some system setup
	for j in range(len(d)-1):
		for i in range(len(d)-1):
			if(d[i]>d[i+1]):
				temp = d[i]
				d[i] = d[i+1]
				d[i+1] = temp
				temp2 = R_sq[i]
				R_sq[i] = R_sq[i+1]
				R_sq[i+1] = temp2

	figure(figsize=(9, 5))
	title(plot_title)
	plot(d, R_sq, 'b-')
	plot(d, fit_func(d, D_ort, exp_ort), 'b:', label='%.3f d ^ %.3f' % tuple(popt))
	grid()
	ylabel('$<$R²$>$ [Mpc²]')
	xlabel('d [Mpc]')
	legend()
	show()
	return D_ort, exp_ort


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

	popt, pcov = curve_fit(fit_func, d, r_sq, bounds=(0, [30.0, 3.0]))
	D = popt[0]
	exponent = popt[1]
	return D, exponent


# test borders of the magnetic field
def test_bField_borders(grid_size, grid_spacing, Brms, lMin, lMax, alpha):
    vgrid = VectorGrid(Vector3d(0), grid_size, grid_spacing)
    initTurbulence(vgrid, Brms, lMin, lMax, alpha)
    Bfield = MagneticFieldGrid(vgrid)
    xs = np.linspace(0, 7* grid_size * grid_spacing, 1000)
    B_list = []
    for x in xs:
        B_list.append(Bfield.getField(Vector3d(0,0,x)).getR())
    plt.plot(xs/Mpc, B_list)
