import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from generateGaussian import MVGaussian
import random

FLOAT_MAX = 1e100

def kpp(gaussians, cluster_centers):
  cluster_centers[0] = deepcopy(random.choice(gaussians))
  d = [0.0 for _ in xrange(len(gaussians))]

  for i in xrange(1, len(cluster_centers)):
    sum = 0
    for j, p in enumerate(gaussians):
      d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
      sum += d[j]

    sum *= random.random()

    for j, di in enumerate(d):
      sum -= di
      if sum > 0:
        continue
      cluster_centers[i] = deepcopy(gaussians[j])
      break

  for p in gaussians:
    p.group = nearest_cluster_center(p, cluster_centers)[0]

def nearest_cluster_center(gaussian, cluster_centers):
  """Distance and index of the closest cluster center"""
  def bregmanDistance(cc,g):
    B = np.trace(np.dot(g.Sigma,cc.Sinv))-cc.logdet
    M = np.sqrt(np.dot((g.Mu-cc.Mu).T,np.dot(cc.Sinv,(g.Mu-cc.Mu))))
    return B + M

  min_index = gaussian.group
  min_dist = FLOAT_MAX

  for i, cc in enumerate(cluster_centers):
    d = bregmanDistance(cc, gaussian)
    if min_dist > d:
      min_dist  = d
      min_index = i

  return (min_index, min_dist)

def clusterGaussians(nclusters, gaussians):
  ''' Input:
       nclusters - number of clusters
       gaussians - list of elements of type MVGaussian
      Output:
       list of K MVGaussians denoting cluster means and variances
  '''
  dim = gaussians[0].dim

  N = len(gaussians)
  plot_flag = False

  # Initial cluster ordering
  cluster_centers = [MVGaussian(dim) for _ in xrange(nclusters)]
  # call k++ init
  kpp(gaussians, cluster_centers)

  lenpts10 = len(gaussians) >> 10

  changed = 0
  while True:
    # group element for centroids are used as counters
    for cc in cluster_centers:
      cc.Mu = np.zeros(dim)
      cc.Sigma = np.eye(dim)
      cc.group = 0

    for p in gaussians:
      cluster_centers[p.group].group += 1
      cluster_centers[p.group].Mu += p.Mu
    for cc in cluster_centers:
      cc.Mu    /= cc.group
    for p in gaussians:
      v = np.atleast_2d(p.Mu - cluster_centers[p.group].Mu).T
      cluster_centers[p.group].Sigma += p.Sigma + np.dot(v,v.T)

    for cc in cluster_centers:
      cc.Sigma /= cc.group
      cc.Sinv = np.linalg.inv(cc.Sigma) # compute inverse once
      cc.logdet = np.linalg.slogdet(cc.Sinv)[1]

    # find closest centroid of each GaussianPtr
    changed = 0
    for p in gaussians:
      min_i = nearest_cluster_center(p, cluster_centers)[0]
      if min_i != p.group:
        changed += 1
        p.group = min_i

    # stop when 99.9% of gaussians are good
    if changed <= lenpts10:
      break

  for i, cc in enumerate(cluster_centers):
    cc.group = i

  return cluster_centers
