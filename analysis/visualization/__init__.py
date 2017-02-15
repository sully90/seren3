def Projection(family, field, mode="fft", camera=None, **kwargs):
    '''
    Performs a basic projection using either the stock raytracing or splatter engine
    '''
    import engines

    if camera is None:
        camera = family.camera()

    engine = None
    if mode == "fft":
        print "Making projection with FFT splatter engine"
        engine = engines.SplatterEngine(family, field)
    elif mode == "rt":
        print "Making projection with RayTracer engine"
        engine = engines.RayTraceEngine(family, field)
    else:
        raise Exception("Unknown visualization engine mode: %s" % mode)

    return engine.process(camera, **kwargs)