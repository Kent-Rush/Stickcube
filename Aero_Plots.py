#Functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_earth(ax, color = 'b', radius = 6378, alpha = .25, resolution = 100):
    # Make data
    u = np.linspace(0,2*np.pi, resolution)
    v = np.linspace(0,np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color=color,alpha = alpha)

def plot_orbit(ax, positions):
    ax.plot(positions[:,0],positions[:,1],positions[:,2])

def scale_plot(ax):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    zlims = ax.get_zlim()
    x_range = xlims[1] - xlims[0]
    y_range = ylims[1] - ylims[0]
    z_range = zlims[1] - zlims[0]
    x_mid = (xlims[1] + xlims[0])/2
    y_mid = (ylims[1] + ylims[0])/2
    z_mid = (zlims[1] + zlims[0])/2
    maxrange = max([x_range, y_range, z_range])
    ax.set_xlim(x_mid - maxrange/2, x_mid + maxrange/2)
    ax.set_ylim(y_mid - maxrange/2, y_mid + maxrange/2)
    ax.set_zlim(z_mid - maxrange/2, z_mid + maxrange/2)
    
def plot_coes(t,coes):
    fig, ((ax5,ax6),(ax7,ax8),(ax9,ax10)) = plt.subplots(3,2,sharex = True,figsize = (8,8))
    fig.suptitle('COEs Plot')
    plt.subplots_adjust(wspace = 0.4,hspace = 0.4)
    ax5.plot(t,coes[:,0])
    ax5.set_title('Inclination')
    ax5.set_ylabel('Degs')
    ax6.plot(t,coes[:,1])
    ax6.set_title('RAAN')
    ax6.set_ylabel('Degs')
    ax7.plot(t,coes[:,2])
    ax7.set_title('Eccentricity')
    ax8.plot(t,coes[:,3])
    ax8.set_title('Argument of Perigee')
    ax8.set_ylabel('Degs')
    #ax8.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax9.plot(t,coes[:,-3])
    ax9.set_title('Energy')
    ax9.set_ylabel('Degs')
    ax9.set_xlabel('Time [Days]')
    ax10.plot(t,coes[:,-2])
    ax10.set_title('Angular Momentum')
    ax10.set_ylabel('Momentum [km^2/s]')
    ax10.set_xlabel('Time [Days]')
    return fig