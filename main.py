'''
    Implement the algorithm described in Differential Entropic Clustering of
    Multivariate Gaussians. [1] Jason V. Davis and Inderjit S. Dhillon,
    Differential Entropic Clustering of Multivariate Gaussians. In Bernhard
    Scholkopf, John Platt, and Thomas Hoffman, editors, Neural Information
    Processing Systems (NIPS), pages 337-344 MIT Press, 2006."
'''
import numpy as np
import matplotlib.pyplot as plt

from generateGaussian import MVGaussian
from DiffEntClust import clusterGaussians

def generate_gaussians(N,dim):
  ''' Generate N gaussians of dimension dim '''
  Gauss = []
  for _ in xrange(N):
      g = MVGaussian(dim)
      Gauss.append(g)
  return Gauss

def plot_gaussians(Gauss):
  # Plot the Gaussians
    init_plot = False
    fig, ax = plt.subplots()
    for g in Gauss:
      g.plotter(ax)
    plt.xlim([-40.,40.])
    plt.ylim([-40.,40.])
    plt.show()

if __name__ == '__main__':
  # Generate a bunch of random multivariate Gaussians
  ngaussians = 100 # Number of Gaussians
  nclusters = 4 # number of clusters
  dim = 2 # So we can plot them
  gaussians = generate_gaussians(ngaussians, dim)
  plot_gaussians(gaussians)

  #cluster_centers = clusterGaussians(nclusters, gaussians)



