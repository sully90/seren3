def render(subsnap, s=None, dynamic_range=3, width=None, **kwargs):
    '''
    Render colour image of stars in the halo
    '''
    import pynbody
    from pynbody.plot import stars
    import matplotlib.pylab as plt
    # r = subsnap.r_in_units(subsnap.C.kpc)
    r = subsnap.region.radius
    r = subsnap.array(r, subsnap.info["unit_length"]).in_units("kpc")

    # f, axs = plt.subplots(1,2,figsize=(14,6))

    if s is None:
        s = subsnap.pynbody_snapshot(filt=True)
    s.physical_units()

    if width is None:
        width='%f kpc' % (0.667*r)

    print 'width = ', width

    # Face-on to the disk
    pynbody.analysis.angmom.faceon(s.s)
    stars.render(s.s, width=width, dynamic_range=dynamic_range)#, axes=axs[0], **kwargs)
    plt.figure()

    # Side-on to the disk
    pynbody.analysis.angmom.sideon(s.s)
    stars.render(s.s, width=width, dynamic_range=dynamic_range)#, axes=axs[1], **kwargs)

    plt.show()