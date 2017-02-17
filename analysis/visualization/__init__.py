from enum import Enum

class EngineMode(Enum):
    SPLATTER = 1
    RAYTRACING = 2

def Projection(family, field, mode=EngineMode.RAYTRACING, camera=None, **kwargs):
    '''
    Performs a basic projection using either the stock raytracing or splatter engine.

    Parameters
    =================================================================================
        family (seren3.core.snapshot.Family) - The family specific dataset for the desired field
        field (string) - the field to visualise
        mode (EngineMode) - projection mode (default RayTracing)
        camera (pymses.analysis.Camera) - the camera object specifying the image domain

    TODO
        Insert some logic for intensive/extensive variables and when to use mass-weighting
    '''
    import engines

    if camera is None:
        camera = family.camera()

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