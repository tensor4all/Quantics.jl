import numpy as np
import scipy.interpolate


def decompose_int(A, n, tol=None):
    A = np.asarray(A)
    Ushape = A.shape[:n]
    VTshape = A.shape[n:]

    Aflat = A.reshape(np.prod(Ushape), np.prod(VTshape))
    Uflat, VTflat = decompose_int_matrix(Aflat, tol)
    U = Uflat.reshape(*Ushape, -1)
    VT = VTflat.reshape(-1, *VTshape)
    return U, VT


def decompose_int_matrix(A, tol=None):
    A = np.asarray(A)
    if np.issubdtype(A.dtype, np.integer):
        A = A.astype(np.float32)
    if tol is None:
        tol = max(2, np.sqrt(A.shape[0])) * np.finfo(A.dtype).eps

    # Truncated SVD
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s_mask = (s > tol * s[0])
    U = U[:, s_mask]
    s = s[s_mask]
    VT = VT[s_mask, :]

    # Truncate small elements
    U_mask = np.abs(U) > tol
    VT_mask = np.abs(VT) > tol
    U[~U_mask] = 0
    VT[~VT_mask] = 0

    # Undo scaling
    U_count = U_mask.sum(0)
    VT_count = VT_mask.sum(1)
    U *= np.sqrt(U_count)
    VT *= np.sqrt(VT_count)[:, None]
    s /= np.sqrt(U_count * VT_count)
    np.testing.assert_allclose(s, 1, atol=tol, rtol=0)

    # Round stuff
    U_int = np.round(U).astype(int)
    VT_int = np.round(VT).astype(int)
    np.testing.assert_allclose(U_int, U, atol=tol, rtol=0)
    np.testing.assert_allclose(VT_int, VT, atol=tol, rtol=0)

    # Regauge values
    U_sgn = np.sign(U_int.sum(0))
    np.testing.assert_equal(np.abs(U_sgn), 1)
    U_int *= U_sgn
    VT_int *= U_sgn[:, None]

    # We want a bit matrix
    np.testing.assert_equal(U_int, U_int.astype(bool))
    np.testing.assert_equal(VT_int, VT_int.astype(bool))

    # Return
    return U_int, VT_int


def interleave_bits(A, K):
    A = np.asarray(A)
    R = A.ndim // K
    if A.shape != (2,) * (K*R):
        raise ValueError("not of proper quantics shape")

    order = np.hstack([np.arange(i, K*R, R) for i in range(R)])
    return A.transpose(*order)


def quantize(A, interleave=False):
    A = np.asarray(A)
    bits = np.log2(A.shape)
    if not (bits == bits.astype(int)).all():
        raise ValueError("invalid shape: " + str(A.shape))
    bits = bits.astype(int)
    A = A.reshape((2,) * bits.sum())

    if interleave:
        if not (bits == bits[0]).all():
            raise ValueError("unable to interleave")
        A = interleave_bits(A, len(bits))
    return A


def print_nonzeros(A, headers=None):
    A = np.asarray(A)
    N = A.ndim
    if headers:
        headers = tuple(headers)
        print(("%2s " * N) % headers)
    fmt = " ".join(("%2i",) * N)
    for x in zip(*A.nonzero()):
        print(fmt % x)
