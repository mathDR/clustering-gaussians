''' 
    Implement the algorithm described in Differential Entropic Clustering of 
    Multivariate Gaussians. [1] Jason V. Davis and Inderjit S. Dhillon, 
    Differential Entropic Clustering of Multivariate Gaussians. In Bernhard 
    Scholkopf, John Platt, and Thomas Hoffman, editors, Neural Information 
    Processing Systems (NIPS), pages 337-344 MIT Press, 2006."
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
      ax.plot(r_ellipse[:,0] + X0,r_ellipse[:,1] + Y0,'b-')
      ax.plot(X0,Y0,'r.')

if __name__ == '__main__':
    # Genereate a bunch of random multivariate Gaussians
    dim = 2 # So we can plot them
    Gaussians = []
    N = 100 # Number of Gaussians
    K = 4 # number of clusters

    mus  = []
    sigs = []
    for _ in xrange(N):
      g = generateMultivariateGaussian(dim)
      Gaussians.append(g)
      mus.append(g.Mu)
      sigs.append(g.Sigma)
    mus = np.asarray(mus)
    sigs = np.asarray(sigs)

    # Plot the Gaussians
    init_plot = False
    if init_plot:
      fig, ax = plt.subplots()
      for g in Gaussians:
        g.plotter(ax)
      plt.show()

    # Initial cluster ordering
    pi = np.random.randint(0,K,N) # Doesn't include K, so [0,1,...,K-1]
    converged = False # flag for convergence
    tot_count = 0
    while tot_count < 10:
      old_pi = pi[:]
      # Update cluster centers
      m = [0]*K
      s = [np.zeros_like(sigs[0])]*K
      count = [0]*K
      for i in range(N):
        j = pi[i]
        m[j] += mus[i]
        count[j] += 1
      for j in range(K):
        if count[j] > 0:
          m[j] /= 1.*count[j]
      for i in range(N):
        j = pi[i]
        v = (mus[i] - m[j])
        s[j] += sigs[i] + np.dot(v,v.T)
      for j in range(K):
        if count[j] > 0:
          s[j] /= 1.*count[j]
      # Update cluster membership
      mins = np.inf * np.ones(N) # The minimum values
      for j in range(K):
        Sinv = np.linalg.inv(s[j]) # Compute it once
        for i in range(N):
          B = np.trace(np.dot(sigs[i],Sinv))-np.log(np.linalg.det(Sinv))
          M = np.sqrt(np.dot((mus[i]-m[j]).T,np.dot(Sinv,(mus[i]-m[j]))))
          val = B + M
          if val < mins[i]:
            pi[i] = j
            mins[i] = val
      if init_plot:
        fig, ax = plt.subplots()
        temp_Gaussians = []
        for j in range(K):
          g = generateMultivariateGaussian(dim)
          g.Mu = m[j]
          g.Sigma = s[j]
          g.plotter(ax)
        plt.show()          
      print np.sum(pi-old_pi)
      if np.sum(pi-old_pi)==0:
        converged = True
      tot_count += 1
      print tot_count
