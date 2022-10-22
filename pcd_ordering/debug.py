import numpy as np
import PyKDTree
import time

if __name__ == '__main__':
    points1 = np.random.rand(2**20, 3).astype(np.double)
    points2 = np.random.rand(2**20, 3).astype(np.double)
    # pdb.set_trace()
    a = time.time()
    tree = PyKDTree.PyKDTree(points1, points2)
    tree.build(1, 1, 2**20, 0)
    result_indx1, result_indx2 = tree.output_index()
    b = time.time()
    print(b-a)
    print(result_indx1.shape)
    print(result_indx1[:20])