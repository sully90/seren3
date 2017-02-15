import numpy as np
from pymses.filters import RegionFilter

def find_galaxy_axis(subsnap, camera=None, nbSample=2000):
    '''
    Provides data access to _find_galaxy_axis function

    Parameters
    ----------
    subsnap : :ref:'PymsesSnapshot'
            A PymsesSnapshot/Subsnapshot/Halo object
    nbSample : int (default=2000)
            number of max massive points to use to compute the axis through cross product
    '''

    point_dset_source = subsnap.g["rho"].source
    camera = subsnap.camera() if camera is None else camera
    return _find_galaxy_axis(point_dset_source, camera, nbSample)

def _find_galaxy_axis(points_dset_source, camera, nbSample):
    '''
    If a galaxy disk is centered in the camera box, this function should
    return a galaxy disk othogonal axis.

    from seren3.utils.camera_utils import find_galaxy_axis

    Parameters
    ----------
    points_dset_source : :ref:`PointDataSource`
            fields "rho" and "size" needed

    camera : pymses camera, the galaxy's center has to fit the camera's center

    nbSample : int
            number of max massive points to use to compute the axis through cross product
    '''
    filtered_points_dset_source = RegionFilter(camera.get_bounding_box(), points_dset_source)
    filtered_points_dset = filtered_points_dset_source.flatten() # multiprocessing data reading and filtering
    region_filtered_mesh_mass = filtered_points_dset.fields["rho"]*(filtered_points_dset.fields["size"]**3)
    argsort = np.argsort(region_filtered_mesh_mass)
    center=camera.center
    nbSample = min(nbSample, argsort.size/2-1)
    result_vect = np.array([0.,0.,0.])
    for i in range (nbSample):
            index1 = argsort[-2*i]
            index2 = argsort[-2*i-1]
            vect1 = filtered_points_dset.points[index1] - center
            vect2 = filtered_points_dset.points[index2] - center
            vect = np.cross(vect1, vect2)
            sign = np.dot(vect, [0.,0.,1.])
            if sign < 0 :
                    vect = - vect
            result_vect = result_vect + vect * (region_filtered_mesh_mass[index1] +
                                                region_filtered_mesh_mass[index2]) * \
                                            np.linalg.norm((vect1-vect2),2)
    result_vect = result_vect/np.linalg.norm(result_vect,2)
    return result_vect

def find_center_of_mass(subsnap, camera=None, nbSample=2000):
    '''
    Provides data access to _find_center_of_mass function

    Parameters
    ----------
    subsnap : :ref:'PymsesSnapshot'
            A PymsesSnapshot/Subsnapshot/Halo object
    nbSample : int (default=2000)
            number of max massive points to use to compute the axis through cross product
    '''

    point_dset_source = subsnap.g["rho"].source
    camera = subsnap.camera() if camera is None else camera
    return _find_center_of_mass(point_dset_source, camera, nbSample)

def _find_center_of_mass(points_dset_source, camera, nbSample):
    r"""
    Find the center of mass in the camera box

    Parameters
    ----------
    points_dset_source : :ref:`PointDataSource`
            fields "rho" and "size" needed

    camera : pymses camera box definition restriction

    nbSample : int (default=2000)
            not working yet : may speed up if random sampling ?
    """
    filtered_points_dset_source = RegionFilter(camera.get_bounding_box(), points_dset_source)
    filtered_points_dset = filtered_points_dset_source.flatten() # multiprocessing data reading and filtering
    d = filtered_points_dset.fields["rho"]*(filtered_points_dset.fields["size"]**3)
    mass=np.sum(d)
    cm=np.sum(d[:,np.newaxis]*filtered_points_dset.points,axis=0)/mass
    return cm

def find_los(subsnap, camera=None, nbSample=2000):
    '''
    Provides data access to _find_los function

    Parameters
    ----------
    subsnap : :ref:'PymsesSnapshot'
            A PymsesSnapshot/Subsnapshot/Halo object
    nbSample : int (default=2000)
            number of max massive points to use to compute the axis through cross product
    '''

    point_dset_source = subsnap.g[["rho", "vel"]].source
    camera = subsnap.camera() if camera is None else camera
    return _find_los(point_dset_source, camera, nbSample)

def _find_los(points_dset_source, camera, nbSample):
    r"""
    Find the line of sight axis which is along the angular momentum of the gas inside the camera box

    Parameters
    ----------
    points_dset_source : :ref:`PointDataSource`
            fields "vel", "rho" and "size" needed

    camera : pymses camera box definition restriction

    nbSample : int (default=2000)
            not working yet : may speed up if random sampling ?
    """
    filtered_points_dset_source = RegionFilter(camera.get_bounding_box(), points_dset_source)
    filtered_points_dset = filtered_points_dset_source.flatten() # multiprocessing data reading and filtering
    d = filtered_points_dset.fields["rho"]*(filtered_points_dset.fields["size"]**3)
    v=filtered_points_dset["vel"]
    JJ=np.zeros_like(v)
    p=d[:,np.newaxis]*v
    JJ[:]=np.cross((filtered_points_dset.points[:]-camera.center),p[:])
    J=np.sum(JJ,axis=0)
    result_vect = J/sum(J**2)
    result_vect = result_vect/np.linalg.norm(result_vect,2)
    return result_vect