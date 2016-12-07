'''
Routines for binning RAMSES datasets
'''

import abc
import numpy as np
from seren3.array import SimArray
from seren3.core.serensource import SerenSource

class ProfileBinner(object):
    '''
    Base class for all profile binners
    '''
    __metaclass__ = abc.ABCMeta # Abstract class
    def __init__(self, field, profile_func, bin_bounds):  #, divide_by_counts=True):
        self.field = field
        self.profile_func = profile_func
        self.bin_bounds = bin_bounds
        # self.divide_by_counts = divide_by_counts

    @abc.abstractmethod
    def bin_func(self, point_dset):
        return

    def process(self, source, ncell_per_dim):
        """Compute the profile of the specified data source
        """

        if not isinstance(source, SerenSource):
            raise Exception("Must supply SerenSource object")
        # Prepare full profile histogram
        # profile = np.zeros(len(self.bin_bounds) - 1)
        profile = 0.

        dest = None
        divide_by_counts = False
        if source.family.family == "amr":
            divide_by_counts = True
            dset = source.amr_sample_points(ncell_per_dim=ncell_per_dim, reshape=False)
        elif source.family.family in ["dm", "star"]:
            dset = source.flatten()

        bin_coords = self.bin_func(dset)

        # Compute profile for this batch
        dprofile = np.histogram(
            bin_coords,
            weights=self.profile_func(dset),
            bins=self.bin_bounds,
            normed=False)[0]

        if divide_by_counts:
            print "Dividing by bin counts"
            # Divide by counts
            counts = np.histogram(
                bin_coords,
                bins=self.bin_bounds,
                normed=False)[0]
            counts[counts == 0] = 1
            dprofile = dprofile / counts

        profile += SimArray(dprofile, dset[self.field].units)

        # if average_over_dsets:
        #     # Average out over number of dsets
        #     return profile / float(len(source))
        # else:
        #     return profile
        return profile

class SphericalProfileBinner(ProfileBinner):
    '''
    Spherical profile binner class
    '''
    def __init__(self, field, center, profile_func, bin_bounds, divide_by_counts=False):
        self.center = np.asarray(center)
        super(SphericalProfileBinner, self).__init__(field, profile_func, bin_bounds, divide_by_counts)

    def bin_func(self, point_dset):
        '''
        Returns array of distances from 'point_dset["pos"]' to 'self.center'
        '''

        # Radial vector from center to pos
        rad = point_dset["pos"] - self.center[np.newaxis, :]

        # The bin is determined by the norm of rad
        return np.sqrt(np.sum(rad * rad, axis=1))

class CylindricalProfileBinner(ProfileBinner):
    """
    Cylindrical profile binner class

    """

    def __init__(self, field, center, axis_vect, profile_func, bin_bounds, divide_by_counts=False):
        self.center = np.asarray(center)
        self.axis_vect = np.asarray(axis_vect) / np.linalg.norm(axis_vect, 2)

        super(CylindricalProfileBinner, self).__init__(field, profile_func, bin_bounds, divide_by_counts)

    def bin_func(self, point_dset):
        """Returns the array of distances from `point_dset["pos"]` to
        the cylinder axis for :class:`PointDataset` objects.
        """
        # Decompose the vector from self.center to point_dset["pos"] into its
        # component along the cylinder axis, and its component in the normal
        # plane
        rad = point_dset["pos"] - self.center[np.newaxis, :]
        along = np.dot(rad, self.axis_vect)[:, np.newaxis] * self.axis_vect
        ortho = rad - along

        # Bin is determined by the radial component only
        return np.sqrt(np.sum(ortho * ortho, axis=1))
