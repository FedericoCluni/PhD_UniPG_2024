# code adapted from 
# https://mathproblems123.wordpress.com/2012/10/19/finite-difference-method-for-2d-laplace-equation/
#
# see Lecture 4 of 15th March 2024 for more information

import numpy as np
import sys
import numpy.linalg as la
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt


def createmat(n):
     cell1 = np.diag(4*np.ones(n))+np.diag(-1*np.ones(n-1),-1)+np.diag(-1*np.ones(n-1),1)
     tridiag = np.kron(np.eye(n),cell1)
     cell2 = np.diag(-1*np.ones(n**2-n),-n)+np.diag(-1*np.ones(n**2-n),n)
     return tridiag + cell2

def laplacefd2(n):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    h = 1./n
    A = createmat(n-2)
    A = A/h**2
    f = np.ones((n-2)**2)
    u = la.solve(A,f)
    Ui = np.reshape(u, (n-2,n-2))
    U = np.zeros((n,n))
    U[1:n-1,1:n-1] = Ui[:]
    #
    # plot the solution
    #
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    [X,Y] = np.meshgrid(x,y)
    surf = ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap=cm.inferno, 
      linewidth=0, antialiased=False)
    cset = ax.contour(X, Y, U, zdir='z', offset=-0.05, cmap=cm.inferno)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set(xlabel='x', ylabel='y', zlabel='u', zlim=(-0.05,0.1))
    plt.show()
    #
    return U
     
if __name__ == "__main__":
    n = int(sys.argv[1])
    U = laplacefd2(n)