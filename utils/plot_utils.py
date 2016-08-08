def ncols(n, cmap='rainbow'):
    import matplotlib.pylab as plt
    import numpy as np
    
    cmap = plt.get_cmap(cmap)
    return cmap(np.linspace(0, 1, n))

def save_eps(fig, fname, dpi=1000):
    fig.savefig(fname, format='eps', dpi=dpi)