#!/usr/bin/python3

import glob
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np

for fname in glob.glob('*1D.dat'):
    #Plot 1D plots using scatter plot
    outname = fname + '.png'

    x, y = np.genfromtxt(fname, unpack=True)

    fig = plt.figure()    
    fig.suptitle(fname, fontsize=14, fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.set_title('Traced on GPU with sandman')

    if "Lambda" in fname:
        ax.set_xlabel('Lambda (A)')

    ax.set_ylabel('Relative intensity')

    ax.scatter(x, y)
    fig.show()
    
    fig.savefig(outname)
    



for fname in glob.glob('*2D.dat'):
    #Plot 1D plots using scatter plot
    outname = fname + '.png'

    x, y, z = np.genfromtxt(fname, unpack=True)

    xmax=np.amax(x)
    xmin=np.amin(x)
    ymax=np.amax(y)
    ymin=np.amin(y)

    xi = np.linspace(xmin,xmax,100)
    yi = np.linspace(ymin,ymax,100)
    zi = griddata(x,y,z, xi, yi, interp='linear')

    fig = plt.figure()    
    fig.suptitle(fname, fontsize=14, fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.set_title('Traced on GPU with sandman')
    ax.set_facecolor((0, 0, 0))


    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)

    #plt.scatter(x, y, c=z, cmap='afmhot', linewidths=0)
    #plt.contourf(xi, yi, zi, 15, cmap=plt.cm.afmhot,vmax=np.amax(zi),vmin=np.amin(zi))
    plt.hist2d(x, y, weights=z, bins=100) 


    plt.colorbar()

    fig.show()
    
    fig.savefig(outname)
    
