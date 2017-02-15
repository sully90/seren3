from seren3 import config
_IMAGE_DIR = config.get("data", "movie_img_dir")

def _get_fname(family, field):
    return "%s/_tmp_%s_%s_%05i.png" % (_IMAGE_DIR, family.family, field, family.ioutput)

def _cleanup():
    import os
    #os.system("rm %s/_tmp*.png" % _IMAGE_DIR)

def _run_mencoder(out_fname, fps, remove_files=True):
    from seren3.utils import which
    import subprocess, os
    mencoder = which("mencoder")
    args = "-mf w=800:h=600:fps=%i:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy" % fps

    # Make the movie
    os.chdir(_IMAGE_DIR)
    exe = "{mencoder} mf://_tmp*.png {args} -o {out_dir}/{out_fname}".format(mencoder=mencoder, \
             out_dir=_IMAGE_DIR, args=args, out_fname=out_fname)
    p = subprocess.check_output(exe, shell=True)

    # Remove _tmp files
    if remove_files:
        _cleanup()

def make_movie(families, out_fname, field="rho", camera_func=None, mpi=True, **kwargs):
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
    import os, sys
    from seren3.analysis import visualization

    if camera_func is None:
        camera_func = lambda family: family.camera()
    fraction = kwargs.pop("fraction", 0.01)
    cmap = kwargs.pop("cmap", "YlOrRd")
    verbose = kwargs.pop("verbose", config.get("general", "verbose"))
    fps = kwargs.pop("fps", 25)

    try:
        if mpi:
            import pymses
            from seren3.analysis.parallel import mpi

            for i in mpi.piter(range(len(families))):
                if verbose:
                    mpi.msg("Image %i/%i" % (i, len(families)))
                family = families[i]
                cam = camera_func(family)

                proj = visualization.Projection(family, field, camera=cam, multi_processing=False, fraction=True, vol_weighted=True, **kwargs)
                fname = _get_fname(family, field)
                proj.save_PNG(img_fname=fname, fraction=fraction, cmap=cmap)

            mpi.comm.barrier()
            if mpi.host:
                _run_mencoder(out_fname, fps)
        else:
            fnames = []
            for i in range(len(families)):
                if verbose:
                    print "Image %i/%i" % (i, len(families))

                family = families[i]
                cam = camera_func(family)
                
                proj = visualization.Projection(family, field, camera=cam, multi_processing=True, **kwargs)
                fname = _get_fname(family, field)
                proj.save_PNG(img_fname=fname, fraction=fraction, cmap=cmap)
                fnames.append(fname)
            _run_mencoder(out_fname, fps)

    except Exception as e:
        _cleanup()
        return e
    except KeyboardInterrupt:
        _cleanup()
        sys.exit()
    return 0

def test(path, istart, iend):
    import seren3
    import numpy as np

    sim = seren3.init(path)
    families = [sim[i].g for i in range(istart, iend+1)]
    camera_func = lambda family: family.camera(region_size=np.array([.05, .05]), distance=.05, far_cut_depth=.05)
    print camera_func(families[0]).__dict__
    return make_movie(families, "output.avi", camera_func=camera_func)

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    istart = int(sys.argv[2])
    iend = int(sys.argv[3])

    print test(path, istart, iend)