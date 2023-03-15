
import numpy as np
import scipy.fftpack as fftpack

def dct2(a):
    return fftpack.dct( fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return fftpack.idct( fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

# D - dictionary of size M x N (where M < N), i.e. N columns each of size M
# u - input signal of size M x 1
# goal is to find: u = D*alpha where alpha of size N x 1
#  nonneg: enforce non-negative coefficients
def OMP_solve_basic(D, X, eps=1e-10, maxit=200, nonneg=False):

    # returns a vector of norms of size N (along each atom or column)
    norms = np.linalg.norm(D, ord=2, axis=0)
    D = np.matrix(D / norms[None,:]) # make each atom have unit norm

    if D.shape[0] != len(X):
        print('D and X must have same number of rows (samples)')
        return 0

    u = np.array(X, dtype=D.dtype, ndmin=2).T # Mx1
    unorm = np.linalg.norm(u, ord=2)
    H = D.getH()  # conjugate transpose of D of size N x M

    residual = u
    active = []
    resVec = np.zeros(D.shape[1], dtype=D.dtype)  # solution vector

    for it in range(maxit):
        # H: NxM, rcov: 1xN, residual: Mx1
        rcov = np.dot(H, residual)
        if nonneg:
            i = np.argmax(rcov)
            rc = rcov[i]
        else:
            i = np.argmax(np.abs(rcov))
            rc = np.abs(rcov[i])

        if i not in active: # update active set
            active.append(i)

        if nonneg:
            coefi, _ = sp.optimize.nnls(D[:, active], u)
        else:
            coefi = np.dot(np.linalg.pinv(D[:, active]),u) #
            #coefi, _, _, _ = np.linalg.lstsq(D[:, active], u)
        resVec[active] = coefi.flatten()  # update solution:

        # update residual vector and error
        # u of size Mx1, D[:,active] of size: M x len(active)
        residual = u - np.dot(D[:, active], coefi)
        err = np.linalg.norm(residual, ord=2)
        print(f"iter: {it}, i: {i}, rc: {rc}, active: {len(active)}, err: {err}")

        err /= unorm
        if err < eps:
            break

    return resVec

# D - dictionary of size M x N (where M < N), i.e. N columns each of size M
# u - input signal of size M x 1
# goal is to find: u = D*alpha where alpha of size N x 1
#  nonneg: enforce non-negative coefficients
def OMP_solve_fast(D, X, eps=1e-10, maxit=200, nonneg=False):

    # returns a vector of norms of size N (along each atom or column)
    norms = np.linalg.norm(D, ord=2, axis=0)
    D = np.array(D / norms[None,:]) # make each atom have unit norm

    if D.shape[0] != len(X):
        print('D and X must have same number of rows (samples)')
        return 0
    u = np.array(X, dtype=D.dtype, ndmin=2).T # Mx1

    unorm = np.linalg.norm(u, ord=2)
    H = np.conjugate(D.T)         # conjugate transpose of D of size N x M
    G = np.matmul(H, D)  # compute D^H * D -> shall result in identity matrix..

    residual = u
    active,a,B = ([],[],[])

    resVec = np.zeros(D.shape[1], dtype=D.dtype)  # solution vector
    pvec = np.dot(H, residual).flatten()
    pvecNorm2 = np.linalg.norm(residual, ord=2)
    pvecNorm2 *= pvecNorm2

    ak = 0
    for k in range(maxit):

        if k > 0:
            pvec = pvec - B[:,k-1]*ak
            print(f"{i} pvec norm: {np.linalg.norm(pvec)}")

        if nonneg:
            i = np.argmax(pvec)
        else:
            i = np.argmax(np.abs(pvec))

        if i not in active: # update active set
            active.append(i)

        if k == 0:
            gammaK = 1.0 / math.sqrt(G[i,i])
            bk =  gammaK * G[:,i]
            B = np.array(bk, dtype=D.dtype, ndmin=2).T
            F = np.array(1.0/gammaK, dtype=D.dtype, ndmin=2).T
        else:
            ck = B[i,:] # i-th row of B
            gammaK = 1.0 / math.sqrt(G[i, i] - np.dot(np.conjugate(ck.T), ck))
            bk = gammaK * (G[:,i] - np.dot(B,ck))
            B = np.c_[B, bk]
            F = np.c_[F, -gammaK*np.dot(F,ck)]
            uuu = np.array([0] * F.shape[0] + [gammaK], dtype=F.dtype)
            F = np.r_[F, [uuu]]

        ak = gammaK * pvec[i]
        a.append(ak)
        pvecNorm2 -= ak*ak # this could get negative ..

        err = math.sqrt(math.fabs(pvecNorm2)) / unorm
        print(f"iter: {k}, active: {len(active)}, err: {err}; ak: {ak}")

        if err < eps:
            break

    resVec[active] = np.dot(F,a)
    return resVec