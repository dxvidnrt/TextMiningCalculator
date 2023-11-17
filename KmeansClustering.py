import numpy as np


def k_means_clustering(index : int, *arrays):

  temp_points = []
  temp_centroids = []

  result  = {}
  for i in range(index):
    temp_points.append(arrays[i])

  for j in range(len(arrays)-index):
    temp_centroids.append(arrays[index + j])


  for centroid in temp_centroids:
    distance = float('inf')
    max_point = [0,0,0]
    for point in temp_points:
      temp_distance = distance
      distance = min(np.linalg.norm(centroid-point), distance)
      if temp_distance != distance:
        max_point = point
    print('Cluster ', centroid, 'is assigned to point: ', max_point, ' with the distance of ', distance)

