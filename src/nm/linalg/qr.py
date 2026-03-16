import numpy as np


def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]

    Q = np.eye(n)
    R = A.copy()
    for i in range(n - 1):
        v = np.zeros(n)
        part = R[i:, i].squeeze()
        norm_part = np.linalg.norm(part)
        if norm_part < 1e-15:
            continue

        v[i:] = part
        v[i] += np.sign(R[i, i]) * norm_part
        H = np.eye(n) - 2 / v.dot(v) * np.outer(v, v)
        R = H.dot(R)
        Q = Q.dot(H)

    return Q, R


def qr_algorithm(A: np.ndarray, precision_epsilon: float = 1e-3) -> np.ndarray:
    A = A.copy()
    n = A.shape[0]

    prev_eigenvalues = np.zeros(n, dtype=complex)

    k = 0
    while (k := k + 1):
        Q, R = qr_decomposition(A)
        A = R.dot(Q)

        current_eigenvalues = []
        converged_mask = []

        i = 0
        while i < n:
            subdiag_norm = np.linalg.norm(A[i+1:, i]) if i < n - 1 else 0

            if subdiag_norm <= precision_epsilon:
                val = A[i, i]
                current_eigenvalues.append(val)
                converged_mask.append(True)
                i += 1
            else:
                a11, a12 = A[i, i], A[i, i+1]
                a21, a22 = A[i+1, i], A[i+1, i+1]

                tr = a11 + a22
                det = a11 * a22 - a12 * a21

                disc = tr**2 - 4*det
                sqrt = np.lib.scimath.sqrt if disc < 0 else np.sqrt

                l1 = (tr + sqrt(disc)) / 2
                l2 = (tr - sqrt(disc)) / 2

                current_eigenvalues.extend([l1, l2])

                if disc >= 0:
                    is_converged = False
                else:
                    is_converged = (abs(l1 - prev_eigenvalues[i]) <= precision_epsilon and 
                                    abs(l2 - prev_eigenvalues[i+1]) <= precision_epsilon)
                converged_mask.extend([is_converged, is_converged])

                i += 2

        if all(converged_mask):
            break

        prev_eigenvalues = np.array(current_eigenvalues)

    return np.array(current_eigenvalues)
