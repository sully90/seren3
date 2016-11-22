import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def _cic(np.ndarray value, np.ndarray pos, int nn, **kwargs):
	cdef int ndim = len(pos[0])
	cdef int size = len(pos)
	cdef int nx = nn
	cdef int ny = 1
	cdef int nz = 1
	cdef np.ndarray x = np.zeros(size)
	cdef np.ndarray y, z

	print size, ndim

	if ndim > 1:
		y = np.zeros(size)
		if ndim > 2:
			z = np.zeros(size)

	for i in range(size):
		x[i] = pos[i,0]
		if ndim > 1:
			y[i] = pos[i,1]
			if ndim > 2:
				z[i] = pos[i,2]

	return cic(value, x, nn, y, nn, z, nn, **kwargs)

def cic(np.ndarray value, np.ndarray x, int nx, np.ndarray y=None,
			 int ny=1, np.ndarray z=None, int nz=1, bool wraparound=True, 
			 bool average=True, bool smooth=False, float sigma=1.):
	return _ccic(value, x, nx, y, ny, z, nz, wraparound, average, smooth, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[DTYPE_t, ndim=2] _ccic(np.ndarray value, np.ndarray x, int nx, np.ndarray y,
			 int ny, np.ndarray z, int nz, bool wraparound, 
			 bool average, bool smooth, float sigma):
	""" Interpolate an irregularly sampled field using Cloud in Cell
	method.

	This function interpolates an irregularly sampled field to a
	regular grid using Cloud In Cell (nearest grid point gets weight
	1-dngp, point on other side gets weight dngp, where dngp is the
	distance to the nearest grid point in units of the cell size).
	
	Inputs
	------
	value: array, shape (N,)
		Sample weights (field values). For a temperature field this
		would be the temperature and the keyword average should be
		True. For a density field this could be either the particle
		mass (average should be False) or the density (average should
		be True).
	x: array, shape (N,)
		X coordinates of field samples, unit indices: [0,NX>.
	nx: int
		Number of grid points in X-direction.
	y: array, shape (N,), optional
		Y coordinates of field samples, unit indices: [0,NY>.
	ny: int, optional
		Number of grid points in Y-direction.
	z: array, shape (N,), optional
		Z coordinates of field samples, unit indices: [0,NZ>.
	nz: int, optional
		Number of grid points in Z-direction.
	wraparound: bool (True)
		If True, then values past the first or last grid point can
		wrap around and contribute to the grid point on the opposite
		side (see the Notes section below).
	average: bool (True)
		If True, average the contributions of each value to a grid
		point instead of summing them.
	smooth: bool (False)
		If True, smooth the field with a Gaussian kernel.
		Will search kwargs for sigma to use in kernel.

	Returns
	-------
	dens: ndarray, shape (nx, ny, nz)
		The grid point values.

	Notes
	-----
	Example of default allocation of nearest grid points: nx = 4, * = gridpoint.

	  0   1   2   3     Index of gridpoints
	  *   *   *   *     Grid points
	|---|---|---|---|   Range allocated to gridpoints ([0.0,1.0> -> 0, etc.)
	0   1   2   3   4   posx

	Example of ngp allocation for wraparound=True: nx = 4, * = gridpoint.

	  0   1   2   3        Index of gridpoints
	  *   *   *   *        Grid points
	|---|---|---|---|--    Range allocated to gridpoints ([0.5,1.5> -> 1, etc.)
	  0   1   2   3   4=0  posx


	References
	----------
	R.W. Hockney and J.W. Eastwood, Computer Simulations Using Particles
		(New York: McGraw-Hill, 1981).

	Modification History
	--------------------
	IDL code written by Joop Schaye, Feb 1999.
	Avoid integer overflow for large dimensions P.Riley/W.Landsman Dec. 1999
	Translated to Python by Neil Crighton, July 2009.
	
	Examples
	--------
	>>> nx = 20
	>>> ny = 10
	>>> posx = np.random.rand(size=1000)
	>>> posy = np.random.rand(size=1000)
	>>> value = posx**2 + posy**2
	>>> field = cic(value, posx*nx, nx, posy*ny, ny)
	# plot surface
	"""
	def findweights(np.ndarray pos, int ngrid):
		""" Calculate CIC weights.
		
		Coordinates of nearest grid point (ngp) to each value. """
		cdef int i
		cdef np.ndarray ngp
		cdef np.ndarray distngp
		cdef np.ndarray weight1
		cdef np.ndarray weight2
		cdef np.ndarray ind1
		cdef np.ndarray ind2
		cdef np.ndarray bad

		if wraparound:
			# grid points at integer values
			ngp = np.fix(pos + 0.5)
		else:
			# grid points are at half-integer values, starting at 0.5,
			# ending at len(grid) - 0.5
			ngp = np.fix(pos) + 0.5

		# Distance from sample to ngp.
		distngp = ngp - pos

		# weight for higher (right, w2) and lower (left, w1) ngp
		weight2 = np.abs(distngp)
		weight1 = 1.0 - weight2

		# indices of the nearest grid points
		if wraparound:
			ind1 = ngp
		else:
			ind1 = ngp - 0.5
		ind1 = ind1.astype(int)

		ind2 = ind1 - 1
		# Correct points where ngp < pos (ngp to the left).
		ind2[distngp < 0] += 2

		# Note that ind2 can be both -1 and ngrid at this point,
		# regardless of wraparound. This is because distngp can be
		# exactly zero.
		bad = (ind2 == -1)
		ind2[bad] = ngrid - 1
		if not wraparound:
			weight2[bad] = 0.
		bad = (ind2 == ngrid)
		ind2[bad] = 0
		if not wraparound:
			weight2[bad] = 0.

		if wraparound:
			ind1[ind1 == ngrid] = 0

		return dict(weight=weight1, ind=ind1), dict(weight=weight2, ind=ind2)


	def update_field_vals(np.ndarray[DTYPE_t, ndim=1] field, np.ndarray[DTYPE_t, ndim=1] totalweight,
			 dict a, dict b, dict c, np.ndarray value):
		""" This updates the field array (and the totweight array if
		average is True).

		The elements to update and their values are inferred from
		a,b,c and value.
		"""
		cdef int ind
		cdef int i
		cdef np.ndarray indices
		cdef np.ndarray weights

		print 'Updating field vals'
		# indices for field - doesn't include all combinations
		indices = a['ind'] + b['ind'] * nx + c['ind'] * nxny
		# weight per coordinate
		weights = a['weight'] * b['weight'] * c['weight']
		# Don't modify the input value array, just rebind the name.
		value = weights * value
		if average:
			for i,ind in enumerate(indices):
				field[ind] += value[i]
				totalweight[ind] += weights[i]
		else:
			for i,ind in enumerate(indices):
				field[ind] += value[i]
			#if debug: print ind, weights[i], value[i], field[ind]

	cdef int nxny=nx*ny
	cdef dict x1, x2, y1, y2, z1, z2
	cdef np.ndarray[DTYPE_t, ndim=1] field = np.zeros(nx * ny * nz, dtype=np.float32)
	cdef np.ndarray[DTYPE_t, ndim=1] totalweight
	cdef np.ndarray good

	x1, x2 = findweights(x, nx)
	y1 = z1 = dict(weight=1., ind=0)
	if y is not None:
		y1, y2 = findweights(y, ny)
		if z is not None:
			z1, z2 = findweights(z, nz)

	if average:
		totalweight = np.zeros(nx * ny * nz, dtype=np.float32)
	else:
		totalweight = None

	update_field_vals(field, totalweight, x1, y1, z1, value)
	update_field_vals(field, totalweight, x2, y1, z1, value)
	if y is not None:
		update_field_vals(field, totalweight, x1, y2, z1, value)
		update_field_vals(field, totalweight, x2, y2, z1, value)
		if z is not None:
			update_field_vals(field, totalweight, x1, y1, z2, value)
			update_field_vals(field, totalweight, x2, y1, z2, value)
			update_field_vals(field, totalweight, x1, y2, z2, value)
			update_field_vals(field, totalweight, x2, y2, z2, value)

	if average:
		good = totalweight > 0
		field[good] /= totalweight[good]

	if smooth:
		#Apply the smoothing kernel
		from scipy.ndimage.filters import gaussian_filter
		return gaussian_filter(field.reshape((nx, ny, nz)).squeeze(), sigma=sigma)

	return field.reshape((nx, ny, nz)).squeeze()