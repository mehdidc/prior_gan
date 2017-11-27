import numpy as np

from scipy.spatial.distance import cdist
from lapjv import lapjv

class Invert:
    def __call__(self, x):
        return 1 - x

class Gray:
    def __call__(self, x):
        return x[0:1]

def grid_embedding(h):
    assert int(np.sqrt(h.shape[0])) ** 2 == h.shape[0], 'Nb of examples must be a square number'
    size = np.sqrt(h.shape[0])
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))).reshape(-1, 2)
    cost_matrix = cdist(grid, h, "sqeuclidean").astype('float32')
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    _, rows, cols = lapjv(cost_matrix)
    return rows


