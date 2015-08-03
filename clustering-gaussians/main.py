''' Implement the algorithm described in Differential Entropic Clustering of Multivariate Gaussians. [1] Jason V. Davis and Inderjit S. Dhillon, Differential Entropic Clustering of Multivariate Gaussians. In Bernhard Scholkopf, John Platt, and Thomas Hoffman, editors, Neural Information Processing Systems (NIPS), pages 337-344 MIT Press, 2006."
'''
import numpy as np
import matplotlib.pyplot as plt

class generateMultivariateGaussian():
    def __init__(self,dim=2):
      self.dim = dim
      self.Mu = np.random.standard_normal(dim)*10
      # Generate a random SPD matrix
      A = np.random.standard_normal((dim,dim))
      self.Sigma = np.dot(A,A.T)
    def plotter(self,ax):
      if self.dim >2 :
        print "Can only plot one or two dimensional Gaussians"
      elif self.dim == 1:
      else:

xi = np.linspace(-2.1,2.1,100)
yi = np.linspace(-2.1,2.1,100)
## grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
levels = [0.2, 0.4, 0.6, 0.8, 1.0]
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
#CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
plt.colorbar() # draw colorbar
# plot data points.
# plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('griddata test (%d points)' % npts)
plt.show()


if __name__ == '__main__':
    # Genereate a bunch of random multivariate Gaussians
    dim = 2 # So we can plot them
    M = generateMultivariateGaussian(dim)
    print M.Mu
    print M.Sigma
    fig, ax = plt.subplots()
    M.plotter(ax)
    plt.show()
