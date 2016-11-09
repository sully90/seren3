from seren3.array import SimArray

h = SimArray(6.6262e-34, "m**2 kg s**-1")
c = SimArray(2.9979250e8, "m s**-1")

egy_func = lambda wavelength: ((h * c)/wavelength.in_units("m")).in_units("eV")
wavelength_func = lambda egy: ((h * c)/egy.in_units("m**2 kg s**-2")).in_units("1e-10 m")