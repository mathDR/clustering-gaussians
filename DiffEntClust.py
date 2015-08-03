import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from generateGaussian import MVGaussian

FLOAT_MAX = 1e100

'''def updateClusterCenters(K,mus,sigs,indices):
  # Update cluster centers
  N = len(mus)
  dim = mus[0].shape[0]
  mu  = np.zeros((K,dim))
  sig = np.zeros((K,dim,dim))
  counts = [0]*K

  for i in range(N):
    j = indices[i] # The cluster the ith gaussian belongs to
    mu[j,:] += mus[i]
    counts[j] += 1

  for j in range(K):
    if counts[j] > 0:
      mu[j,:] /= 1.*counts[j]
  for i in range(N):
    j = indices[i]
    v = np.atleast_2d(mus[i] - mu[j,:]).T
    sig[j,:,:] += sigs[i] + np.dot(v,v.T)
  for j in range(K):
    if counts[j] > 0:
      sig[j,:,:] /= 1.*counts[j]
  return mu, sig
'''

def updateClusterMembership(m,s,mus,sigs):
  N = len(mus)
  K = len(m)
  mins = np.inf * np.ones(N) # The minimum values
  clusterIndices = np.zeros(N)
  for j in range(K):
    Sinv = np.linalg.inv(s[j,:,:]) # Compute it once
    sgn, log_det = np.linalg.slogdet(Sinv)
    for i in range(N):
      B = np.trace(np.dot(sigs[i],Sinv))-log_det
      M = np.sqrt(np.dot((mus[i]-m[j,:]).T,np.dot(Sinv,(mus[i]-m[j,:]))))
      val = B + M
      if val < mins[i]:
        clusterIndices[i] = j
        mins[i] = val
  return np.asarray(clusterIndices,dtype=np.int)

def nearest_cluster_center(gaussian, cluster_centers):
  """Distance and index of the closest cluster center"""
  def bregmanDistance(cc, g):
    return (a.x - b.x) ** 2  +  (a.y - b.y) ** 2

  min_index = point.group
  min_dist = FLOAT_MAX

  for i, cc in enumerate(cluster_centers):
    d = bregmanDistance(cc, gaussian)
    if min_dist > d:
      min_dist  = d
      min_index = i

  return (min_index, min_dist)

def plot_clusters(m,s,MVGaussians):
  K = m.shape[0]
  fig, ax = plt.subplots()
  temp_Gaussians = []
  for g in MVGaussians:
    g.plotter(ax)
  for j in range(K):
    g = MVGaussian(m[0].shape[0])
    g.Mu = m[j,:]
    g.Sigma = s[j,:,:]
    g.plotter(ax,cluster=True)
  plt.xlim([-40.,40.])
  plt.ylim([-40.,40.])
  #plt.axis('equal')
  plt.show()

def generateClusterIndices(K,N):
  # Use an extension of k-means++ (Not implemented yet)
  z = [0]*((N/K))
  for i in range(1,K):
    z.extend([i]*((N/K)))
  for i in range(len(z),N):
    z.append(K-1)
  return z[:N]



def clusterGaussians(nclusters, gaussians):
  ''' Input:
       nclusters - number of clusters
       gaussians - list of elements of type MVGaussian
      Output:
       list of K MVGaussians denoting cluster means and variances
  '''
  dim = gaussians[0].dim
  # Breakout MVGaussians into arrays of means and covariances
  #mus  = []; sigs = []
  #for g in MVGaussians:
  #  mus.append(g.Mu); sigs.append(g.Sigma)
  #mus = np.asarray(mus); sigs = np.asarray(sigs)
  #print mus.shape, mus[0].shape

  N = len(MVGaussians)
  plot_flag = False

  # Initial cluster ordering
  cluster_centers = [MVGaussian(dim) for _ in xrange(nclusters)]
  # call k++ init
  kpp(gaussians, cluster_centers)
  #clusterIndices = generateClusterIndices(K,N)

  lenpts10 = len(points) >> 10

  changed = 0
  while True:
    # group element for centroids are used as counters
    for cc in cluster_centers:
      cc.Mu = np.zeros(dim)
      cc.y = np.eye(dim)
      cc.group = 0

    for p in gaussians:
      cluster_centers[p.group].group += 1
      cluster_centers[p.group].Mu += p.Mu
    for p in gaussians:
      v = (p.Mu - cluster_centers[p.group].Mu)
      cluster_centers[p.group].Sigma += p.Sigma + np.dot(v,v.T)

    for cc in cluster_centers:
      cc.Mu    /= cc.group
      cc.Sigma /= cc.group

    # find closest centroid of each PointPtr
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


  converged = False # flag for convergence
  tot_count = 0
  while (not converged) and (tot_count < 1000): # tot_count is there to prevent infinite loops
    old_clusterIndices = copy(clusterIndices)
    # Update cluster centers
    m,s = updateClusterCenters(K,mus,sigs,clusterIndices)

    # Update cluster membership
    clusterIndices = updateClusterMembership(m,s,mus,sigs)

    if plot_flag:
      plot_clusters(m,s,MVGaussians)

    if np.all(np.isclose(clusterIndices,old_clusterIndices)):
      converged = True
    tot_count += 1
    break
