# necessary imports
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean
points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    '''
    dist = np.zeros(([points.shape[0],centroids.shape[0]])
    for i in range(len(dist)):
        for j in range(len(i)):
            dist[i,j] = euclidean(points[i], centroids[j])
    '''
    dists = np.sqrt(((points - centroids[:,np.newaxis])**2).sum(axis=2))

    return np.argmin(dists,axis=0)


def move_centroids(points, closest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def graph(centroids,points):
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    ax = plt.gca()
    plt.show()

def main(points):
    num_iterations = 100
    k = 3
    # у нас есть кординаты множества точек мы определеляем центроиды (k точек) случайным образом
    # столько классов у нас будет, по которым мы должны разбить остальные точки.
    # сначала ищем все точки, пренадлежащие каждой центроиде. После этого ищем среднее среди их координат и смещаем центроиды туда
    # спустя некоторое время центроиды перестанут двигаться или их сдвиг будет меньше некоторого E
    # Initialize centroids
    centroids = initialize_centroids(points, k)
    graph(centroids,points)


    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)

    return centroids


centroids = main(points)

graph(centroids, points)