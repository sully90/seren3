from seren3 import config

def _get_fname(family, field):
    _IMAGE_DIR = config.get("data", "movie_img_dir")
    return "%s/_tmp_%s_%s_%05i.png" % (_IMAGE_DIR, family.family, field, family.ioutput)

def make_movie(families, field="rho", camera_func=None, mpi=True, **kwargs):
    '''
        Parameters
        ----------
        families     : :iterable:class:`~seren3.core.snapshot.Family`
            iterable of families to make images from
        field        : ``string`` (default rho)
            field to make projections of.
        camera          : :class:`~pymses.analysis.camera.Camera`
            camera containing all the view params

        Returns
        -------
        TODO
    '''
    from seren3.analysis import visualization

    if camera_func is None:
        camera_func = lambda family: family.camera()
    fraction = kwargs.pop("fraction", 0.01)
    cmap = kwargs.pop("cmap", "Viridis")
    verbose = config.get("general", "verbose")

    try:
        if mpi:
            import pymses
            from seren3.analysis.parallel import mpi

            pymses.utils.misc.NUMBER_OF_PROCESSES_LIMIT = 1  # disable multiprocessing
            dest = {}
            for i, sto in mpi.piter(range(len(families)), storage=dest):
                if verbose:
                    mpi.msg("Image %i/%i" % (i, len(families)))
                family = families[i]
                cam = camera_func(family)

                proj = visualization.Projection(family, field, camera=cam, **kwargs)
                fname = _get_fname(family, field)
                proj.save_PNG(img_fname=fname)
                sto.result = {"fname" : fname}
            dset = mpi.unpack(dset)
            print dset
        else:
            fnames = []
            for i in range(len(families)):
                if verbose:
                    print "Image %i/%i" % (i, len(families))

                family = families[i]
                cam = camera_func(family)
                
                proj = visualization.Projection(family, field, camera=cam, **kwargs)
                fname = _get_fname(family, field)
                proj.save_PNG(img_fname=fname)
                fnames.append(fname)
    except Exception as e:
        return e
    return 0

def test(path, istart, iend):
    import seren3
    sim = seren3.init(path)
    families = [sim[i].g for i in range(istart, iend+1)]
    return make_movie(families)

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    istart = int(sys.argv[2])
    iend = int(sys.argv[3])

    print test(path, istart, iend)