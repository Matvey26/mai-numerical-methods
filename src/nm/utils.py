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
