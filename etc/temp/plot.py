import matplotlib.pyplot as plt

from matplotlib import cm

import numpy as np

from scipy.interpolate import griddata

from matplotlib.colors import ListedColormap,LinearSegmentedColormap,Normalize

from math import log10

import glob, os 

########################################

# Plot parameters

########################################

fig = plt.figure(figsize=(5,5))

plt.rcParams.update({'font.size': 14})

#plt.rcParams.update({'font.family': 'Times'}) 

tics = np.arange(0,220,20) 

contourlevels=(0.1,0.3,0.5,0.7,0.9) 

# Colorbars

col = {'red':  ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),

         'green': ((0.0, 1.0, 1.0),(1.0, 0.8, 0.8)),

         'blue':  ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'alpha':  ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0))}

my_gold = LinearSegmentedColormap('WhiteGold', col) 

col = {'red':  ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'green': ((0.0, 1.0, 1.0),(1.0, 0.8, 0.8)),

         'blue':  ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'alpha':  ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0))}

my_green = LinearSegmentedColormap('WhiteGreen', col) 

col = {'red':  ((0.0, 1.0, 1.0),(1.0, 0.8, 0.8)),

         'green': ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'blue':  ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'alpha':  ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0))}

my_red = LinearSegmentedColormap('WhiteRed', col) 

col = {'red':  ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'green': ((0.0, 1.0, 1.0),(1.0, 0.0, 0.0)),

         'blue':  ((0.0, 1.0, 1.0),(1.0, 0.9, 0.9)),

         'alpha':  ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0))}

my_blue = LinearSegmentedColormap('WhiteBlue', col) 

ngrid = 200

xi = np.linspace(0., 200., ngrid)

yi = np.linspace(0., 200., ngrid) 

def snapshot(file_name):

    print('Plotting '+file_name+'...')

    fig.clf()

    x,y,c,Eta1,Eta2,Eta3,Eta4 = np.loadtxt(file_name, skiprows=1, unpack=True)

    ci = griddata((x, y), c, (xi[None,:], yi[:,None]))

    eta1i = griddata((x, y), Eta1, (xi[None,:], yi[:,None]))

    eta2i = griddata((x, y), Eta2, (xi[None,:], yi[:,None]))

    eta3i = griddata((x, y), Eta3, (xi[None,:], yi[:,None]))

    eta4i = griddata((x, y), Eta4, (xi[None,:], yi[:,None]))

    ax = fig.add_subplot()

    index = [int(word) for word in str(file_name).split('.') if word.isdigit()]

    ax.set_title("t = "+str(index[0]))

    ax.set_aspect(1.0)

    ax.set_xlabel('x')

    ax.set_xlim([0, 200])

    plt.xticks(tics)

    ax.set_ylabel('y')

    ax.set_ylim([0, 200])

    plt.yticks(tics)

    ax.pcolor(eta1i, vmin=0.0, vmax=1.0, cmap=my_blue)

    ax.pcolor(eta2i, vmin=0.0, vmax=1.0, cmap=my_green)

    ax.pcolor(eta3i, vmin=0.0, vmax=1.0, cmap=my_gold)

    ax.pcolor(eta4i, vmin=0.0, vmax=1.0, cmap=my_red)

    contours = ax.contour(xi, yi, ci, contourlevels, colors='k')

    plt.savefig(file_name+".png",bbox_inches='tight',dpi=300) 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for file in glob.glob("Fields.*.dat"):

    snapshot(file);