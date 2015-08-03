'''
    Implement the algorithm described in Differential Entropic Clustering of
    Multivariate Gaussians. [1] Jason V. Davis and Inderjit S. Dhillon,
    Differential Entropic Clustering of Multivariate Gaussians. In Bernhard
    Scholkopf, John Platt, and Thomas Hoffman, editors, Neural Information
    Processing Systems (NIPS), pages 337-344 MIT Press, 2006."
'''
import numpy as np
import matplotlib.pyplot as plt

from generateGaussian import MVGaussian, generate_gaussians, plot_gaussians
from DiffEntClust import clusterGaussians

if __name__ == '__main__':
  # Generate a bunch of random multivariate Gaussians
  ngaussians = 1000 # Number of Gaussians
  nclusters = 6 # number of clusters
  dim = 2 # So we can plot them
  gaussians = generate_gaussians(ngaussians, dim)
  #plot_gaussians(gaussians)

  cluster_centers = clusterGaussians(nclusters, gaussians)
  plot_gaussians(gaussians,cluster_centers)



