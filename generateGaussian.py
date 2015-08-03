import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def generate_gaussians(N,dim):
  ''' Generate N gaussians of dimension dim '''
  Gauss = []
  for _ in xrange(N):
      g = MVGaussian(dim)
      Gauss.append(g)
  return Gauss

def plot_gaussians_interactive(Gauss, cluster_centers):
  # Plot the Gaussians
    init_plot = False
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    for g in Gauss:
      g.plotter(ax)
    for g in cluster_centers:
      g.plotter(ax,cluster=True)
    plt.xlim([-40.,40.])
    plt.ylim([-40.,40.])
    plt.draw()
    sleep(0.1)

def plot_gaussians(Gauss, cluster_centers = None):
  # Plot the Gaussians
    init_plot = False
    fig, ax = plt.subplots()
    for g in Gauss:
      g.plotter(ax)
    if cluster_centers is not None:
      for g in cluster_centers:
        g.plotter(ax,cluster=True)
    plt.xlim([-40.,40.])
    plt.ylim([-40.,40.])
    plt.show()

class MVGaussian():
  def __init__(self,dim=2):
    self.dim = dim
    self.Mu = np.random.standard_normal(dim)*10
    # Generate a random SPD matrix
    A = np.random.standard_normal((dim,dim))
    #self.Sigma = np.dot(A,A.T)
    self.Sigma = np.eye(dim)
    self.group = 0 # for clustering purposes
    self.Sinv  = np.eye(dim) # Holds inv(self.Sigma)
    self.logdet = 1.0 # Holds log(det(Sinv))

  def plotter(self,ax,cluster=False):
    if self.dim >2 :
      print "Can only plot one or two dimensional Gaussians"
    elif self.dim == 1:
      # plot from [-x,x] where x = mean + 5*std with 250 points
      pass
    else:
      [lam,v] = np.linalg.eig(self.Sigma)
      if lam[0] < lam[1]:
        #eigenvalues are not sorted
        lam = lam[::-1]
        v   = v[::-1]
      # Calculate the angle between the x-axis and the largest eigenvector
      angle = np.arctan2(v[0,1], v[0,0])
      # This angle is between -pi and pi.
      # Let's shift it such that the angle is between 0 and 2pi
      if angle < 0:
        angle += 2.0*np.pi;

      # Get the coordinates of the data mean
      avg = self.Mu
      # Get the 95% confidence interval error ellipse
      chisquare_val = 2.4477;
      theta_grid = np.linspace(0.,2.0*np.pi,100)
      phi = angle;
      [X0,Y0] = avg
      [a,b] = chisquare_val*np.sqrt(lam)

      # the ellipse in x and y coordinates
      ellipse_x_r  = a*np.cos( theta_grid )
      ellipse_y_r  = b*np.sin( theta_grid )
      #Define a rotation matrix
      R = np.asarray([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])

      # let's rotate the ellipse to some angle phi
      r_ellipse = np.dot(np.asarray([ellipse_x_r,ellipse_y_r]).T, R)
      # Draw the error ellipse
      if cluster:
        ax.fill(r_ellipse[:,0] + X0,r_ellipse[:,1] + Y0,'k',alpha = 0.3)
        ax.plot(X0,Y0,'gs')
      else:
        ax.fill(r_ellipse[:,0] + X0,r_ellipse[:,1] + Y0,'b',alpha=0.3)
        ax.plot(X0,Y0,'r.')

if __name__ == '__main__':
  # Generate a bunch of random multivariate Gaussians
    dim = 2 # So we can plot them
    Gaussians = []
    N = 100 # Number of Gaussians

    for _ in xrange(N):
      g = MVGaussian(dim)
      Gaussians.append(g)

    # Plot the Gaussians
    fig, ax = plt.subplots()
    for g in Gaussians:
      g.plotter(ax)
    plt.axis([-40,40,-40,40])
    plt.axis_equal
    plt.show()


