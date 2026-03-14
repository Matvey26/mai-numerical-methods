import numpy as np
from scipy.sparse import csr_matrix


def transpose_permutation(p):
    p_new = [0] * len(p)
    for i, k in enumerate(p):
        p_new[k] = i
    return p_new

def count_permutation(p):
    p = p[:]
    result = 0
    for i in range(len(p)):
        if p[i] == -1: continue

        cnt = 0
        while p[i] != -1:
            i_new = p[i]
            p[i] = -1

            cnt += 1
            i = i_new

        result += cnt - 1

    return result


def csr_row_norm(A: csr_matrix):
    rows_sum = np.add.reduceat(
        np.abs(A.data),
        A.indptr[:-1]
    )
    return np.max(rows_sum)


def csr_col_norm(A: csr_matrix):
    cols_sum = np.zeros(A.shape[1])
    np.add.at(cols_sum, A.indices, np.abs(A.data))
    return np.max(cols_sum)