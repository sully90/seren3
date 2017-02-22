from enum import Enum

class EngineMode(Enum):
    SPLATTER = 1
    RAYTRACING = 2

class GalaxyAxis(Enum):
    FACEON = 1
    ANGMOM = 2

def Projection(family, field, camera=None, mode=EngineMode.SPLATTER, gal_axis=None, **kwargs):
    '''
    Performs a basic projection using either the stock raytracing or splatter engine.

    Parameters
    =================================================================================
        family (seren3.core.snapshot.Family) - The family specific dataset for the desired field
        field (string) - the field to visualise
        mode (EngineMode) - projection mode (default Splatter)
        camera (pymses.analysis.Camera) - the camera object specifying the image domain

    TODO
        Insert some logic for intensive/extensive variables and when to use mass-weighting
    '''
    import engines

    if camera is None:
        camera = family.camera()

    if gal_axis is not None:
        if isinstance(gal_axis, GalaxyAxis):
            from seren3.utils import camera_utils
            gal_los = None
            if gal_axis == GalaxyAxis.FACEON:
                gal_los = camera_utils.find_galaxy_axis(family.base, camera)
            elif gal_axis == GalaxyAxis.ANGMOM:
                gal_los = camera_utils.find_los(family.base, camera)
            camera.los_axis = gal_los
        else:
            raise Exception("Must choose from GalaxyAxis enum to set gal_axis")

    engine = None
    if mode == EngineMode.SPLATTER:
        print "Making projection with FFT splatter engine"
        engine = engines.SplatterEngine(family, field)
    elif mode == EngineMode.RAYTRACING:
        print "Making projection with RayTracer engine"
        engine = engines.RayTraceEngine(family, field)
    else:
        raise Exception("Unknown visualization engine mode: %s" % mode)

    return engine.process(camera, **kwargs)