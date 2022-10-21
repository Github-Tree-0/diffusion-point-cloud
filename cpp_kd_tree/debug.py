import numpy as np
import PyKDTree
import time

if __name__ == '__main__':
    points = np.random.rand(2**20, 3).astype(np.double)
    # pdb.set_trace()
    a = time.time()
    tree = PyKDTree.PyKDTree(points)
    tree.build(1, 1, 2**20, 0)
    result_indx = tree.output_index()
    b = time.time()
    print(b-a)
    print(result_indx.shape)
    print(result_indx[:20])
    