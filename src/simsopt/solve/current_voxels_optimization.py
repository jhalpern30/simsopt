import sys
import numpy as np
from simsoptpp import acc_prox_grad_descent, matmult, matmult_sparseA
import time
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.sparse import csc_matrix, spdiags, vstack, hstack, eye
from scipy.sparse.linalg import LinearOperator, splu, utils
from numpy import inner, zeros, inf, finfo, isclose
from numpy.linalg import norm
from math import sqrt
from functools import partial

__all__ = ['relax_and_split_increasingl0', 'ras_minres', 'ras_preconditioned_minres']


def minres(A, b, x0=None, shift=0.0, tol=1e-5, maxiter=None,
           M=None, callback=None, show=False, check=False, print_iter=10):
    """
    Use MINimum RESidual iteration to solve Ax=b. 

    NOTE: This is a direct copy of
    the minres function implemented in scipy.sparse.linalg, except 
    for commenting out two lines relating to the istop = 1 termination 
    condition for the algorithm, which allows the algorithm
    to progress further. Also, only calls the callback(x) function
    every print_iter iterations, to avoid excessive computations. 

    MINRES minimizes norm(Ax - b) for a real symmetric matrix A.  Unlike
    the Conjugate Gradient method, A can be indefinite or singular.

    If shift != 0 then the method solves (A - shift*I)x = b

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real symmetric N-by-N matrix of the linear system
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    Other Parameters
    ----------------
    x0 : ndarray
        Starting guess for the solution.
    shift : float
        Value to apply to the system ``(A - shift * I)x = b``. Default is 0.
    tol : float
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below `tol`.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    show : bool
        If ``True``, print out a summary and metrics related to the solution
        during iterations. Default is ``False``.
    check : bool
        If ``True``, run additional input validation to check that `A` and
        `M` (if specified) are symmetric. Default is ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import minres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> A = A + A.T
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = minres(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    References
    ----------
    Solution of sparse indefinite systems of linear equations,
        C. C. Paige and M. A. Saunders (1975),
        SIAM J. Numer. Anal. 12(4), pp. 617-629.
        https://web.stanford.edu/group/SOL/software/minres/

    This file is a translation of the following MATLAB implementation:
        https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip

    """
    A, M, x, b, postprocess = utils.make_system(A, M, x0, b)

    matvec = A.matvec
    psolve = M.matvec

    first = 'Enter minres.   '
    last = 'Exit  minres.   '

    n = A.shape[0]

    if maxiter is None:
        maxiter = 5 * n

    msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ',   # -1
           ' beta1 = 0.  The exact solution is x0          ',   # 0
           ' A solution to Ax = b was found, given rtol        ',   # 1
           ' A least-squares solution was found, given rtol    ',   # 2
           ' Reasonable accuracy achieved, given eps           ',   # 3
           ' x has converged to an eigenvector                 ',   # 4
           ' acond has exceeded 0.1/eps                        ',   # 5
           ' The iteration limit was reached                   ',   # 6
           ' A  does not define a symmetric matrix             ',   # 7
           ' M  does not define a symmetric matrix             ',   # 8
           ' M  does not define a pos-def preconditioner       ']   # 9

    if show:
        print(first + 'Solution of symmetric Ax = b')
        print(first + 'n      =  %3g     shift  =  %23.14e' % (n, shift))
        print(first + 'itnlim =  %3g     rtol   =  %11.2e' % (maxiter, tol))
        print()

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = x.dtype

    eps = finfo(xtype).eps

    # Set up y and v for the first Lanczos vector v1.
    # y  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.

    if x0 is None:
        r1 = b.copy()
    else:
        r1 = b - A@x
    y = psolve(r1)

    beta1 = inner(r1, y)

    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return (postprocess(x), 0)

    bnorm = norm(b)
    if bnorm == 0:
        x = b
        return (postprocess(x), 0)

    beta1 = sqrt(beta1)

    if check:
        # are these too strict?

        # see if A is symmetric
        w = matvec(y)
        r2 = matvec(w)
        s = inner(w, w)
        t = inner(y, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            raise ValueError('non-symmetric matrix')

        # see if M is symmetric
        r2 = psolve(y)
        s = inner(y, y)
        t = inner(r1, r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            raise ValueError('non-symmetric preconditioner')

    # Initialize other quantities
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = finfo(xtype).max
    cs = -1
    sn = 0
    w = zeros(n, dtype=xtype)
    w2 = zeros(n, dtype=xtype)
    r2 = r1

    if show:
        print()
        print()
        print('   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')

    while itn < maxiter:
        itn += 1

        s = 1.0/beta
        v = s*y

        y = matvec(v)
        y = y - shift * v

        if itn >= 2:
            y = y - (beta/oldb)*r1

        alfa = inner(v, y)
        y = y - (alfa/beta)*r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = inner(r2, y)
        if beta < 0:
            raise ValueError('non-symmetric matrix')
        beta = sqrt(beta)
        tnorm2 += alfa**2 + oldb**2 + beta**2

        if itn == 1:
            if beta/beta1 <= 10*eps:
                istop = -1  # Terminate later

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
        epsln = sn * beta     # epsln2 = 0         epslnk+1
        dbar = - cs * beta   # dbar 2 = beta2     dbar k+1
        root = norm([gbar, dbar])
        Arnorm = phibar * root

        # Compute the next plane rotation Qk

        gamma = norm([gbar, beta])       # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma             # ck
        sn = beta / gamma             # sk
        phi = cs * phibar              # phik
        phibar = sn * phibar              # phibark+1

        # Update  x.

        denom = 1.0/gamma
        w1 = w2
        w2 = w
        w = (v - oldeps*w1 - delta*w2) * denom
        x = x + phi*w

        # Go round again.

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta*z
        rhs2 = - epsln*z

        # Estimate various norms and test for convergence.

        Anorm = sqrt(tnorm2)
        ynorm = norm(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * tol
        diag = gbar

        if diag == 0:
            diag = epsa

        if itn == 1:
            test0 = 0
        else:
            test0 = test1

        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = inf
        else:
            test1 = rnorm / (Anorm*ynorm)    # ||r||  / (||A|| ||x||)
        if Anorm == 0:
            test2 = inf
        else:
            test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q @ H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax/gmin

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        if istop == 0:
            t1 = 1 + test1      # These tests work if tol < eps
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            # if t1 <= 1:
            #     istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1/eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            # if rnorm <= epsx   : istop = 2
            # if rnorm <= epsr   : istop = 1
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1

            # Custom condition if it slows down too much
            # if isclose(abs(test1 - test0), tol, atol=1e-24, rtol=1e-5):
            #     istop = 5

        # See if it is time to print something.

        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= maxiter-10:
            prnt = True
        if itn % 10 == 0:
            prnt = True
        if qrnorm <= 10*epsx:
            prnt = True
        if qrnorm <= 10*epsr:
            prnt = True
        if Acond <= 1e-2/eps:
            prnt = True
        if istop != 0:
            prnt = True

        if show and prnt:
            str1 = '%6g %12.5e %10.3e' % (itn, x[0], test1)
            str2 = ' %10.3e' % (test2,)
            str3 = ' %8.1e %8.1e %8.1e' % (Anorm, Acond, gbar/Anorm)

            print(str1 + str2 + str3)

            if itn % 10 == 0:
                print()

        if (itn % print_iter) == 0:
            if callback is not None:
                callback(x)

        if istop != 0:
            break  # TODO check this

    if show:
        print()
        print(last + ' istop   =  %3g               itn   =%5g' % (istop, itn))
        print(last + ' Anorm   =  %12.4e      Acond =  %12.4e' % (Anorm, Acond))
        print(last + ' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm, ynorm))
        print(last + ' Arnorm  =  %12.4e' % (Arnorm,))
        print(last + msg[istop+1])

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return (postprocess(x), info)


def compute_J(current_voxels, alphas, ws):
    """
    Compute J at each quadrature point from the alphas
    determined through optimization. 

    Args:
        current_voxels: CurrentVoxelsGrid class object
          representing the geometry of the voxel grid.
        alphas: 2D numpy array, shape (num_voxels, num_basis_functions)
          representing the optimized voxel degrees of freedom.
        ws: 2D numpy array, shape (num_voxels, num_basis_functions) 
          representing the proxy voxel degrees of freedom.
    """
    num_basis = current_voxels.n_functions
    n_interp = current_voxels.Phi.shape[2]
    current_voxels.J = np.sum(np.multiply(alphas.T[:, :, np.newaxis, np.newaxis], current_voxels.Phi), axis=0)
    current_voxels.J_sparse = np.sum(np.multiply(ws.T[:, :, np.newaxis, np.newaxis], current_voxels.Phi), axis=0)


def prox_group_l0(alpha, threshold, n, num_basis):
    """
    Computes the non-overlapping group l0 proximal operator,
    which is a very fancy way to say threshold the subgroups to 
    zero if the subgroup has norm < threshold.

    Args:
        alpha: 1D numpy array, shape (num_voxels, num_basis_functions)
          representing the optimization variables to threshold.
        threshold: float
          Value such that if a subgroup in alpha has norm < threshold,
          we set that whole subgroup to zeros.
        n: int
          Number of voxels in the unique part of the voxel grid.
        num_basis: int
          Number of basis functions in each voxel, hard-coded to 5 elsewhere.
    """
    alpha_mat = alpha.reshape(n, num_basis)
    alpha_mat[np.linalg.norm(alpha_mat.reshape(n, num_basis), axis=-1) < threshold, :] = 0.0
    return alpha_mat.reshape(n * num_basis, 1)


def relax_and_split_increasingl0(
        current_voxels, lam=0.0, nu=1e20, max_iter=5000, sigma=1.0,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=10,
        alpha0=None):
    """
    This function performs a relax-and-split algorithm for solving the 
    current voxel problem. The convex part of the solve is done with
    accelerated projected gradient descent and the nonconvex part 
    can be solved with the prox of the group l0 norm... hard 
    thresholding on the whole set of alphas in a grid cell. 

    Args:
        P : projection matrix onto the linear equality constraints representing
            the flux jump constraints. If C is the flux jump constraint matrix,
            P = I - C.T @ (C * C.T)^{-1} @ C. P is very large but very sparse
            so is assumed to be stored as a scipy csc_matrix. 
    """
    t1 = time.time()
    P = current_voxels.P
    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N, 1) - 0.5) * 1e4  

    if P is not None:
        alpha_opt = P.dot(alpha_opt)

    B = current_voxels.B_matrix
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b = b.reshape(len(b), 1)
    ITbI = contig(IT * b_I)
    BTb = contig(BTb.reshape(len(BTb), 1))   
    alpha_opt = contig(alpha_opt.reshape(len(alpha_opt), 1))
    IT = contig(IT)
    BT = contig(BT)
    I = contig(I)
    B = contig(B)
    #optimization_dict = {'A': B, 'b': b, 'C': current_voxels.C.todense(), 'A_I': I, 'b_I': b_I}
    #np.save('optimization_matrices.npy', optimization_dict)
    t2 = time.time()

    lam_nu = (lam + 1.0 / nu)

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    L = np.linalg.svd(BT @ B + sigma * IT @ I + np.eye(N) * lam_nu, compute_uv=False, hermitian=True)[0]
    L0 = np.linalg.svd(BT @ B + sigma * IT @ I + np.eye(N) * lam, compute_uv=False, hermitian=True)[0]
    #if True:
    # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam_nu, axis=0), axis=-1)
    # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + np.eye(N) * lam_nu, axis=0), axis=-1)
    #else:
    #    L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size) 
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    f_B = []
    f_I = []
    f_K = []
    f_RS = []
    f_Bw = []
    f_Iw = []
    f_Kw = []
    f_0 = []
    f_0w = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    t1 = time.time()
    set_point = True
    #w_opt = alpha_opt  # np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    w_opt = np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        if j > (2.0 * len(l0_thresholds) / 3.0) and set_point:
            rs_max_iter = 3 * rs_max_iter
            set_point = False
        for k in range(rs_max_iter):
            if j == 0 and k == 0 and len(l0_thresholds) > 1:
                nu_algo = 1e100
                #nu_algo = nu
                step_size_i = 1.0 / L0
                max_it = 5000
                #max_it = max_iter 
            else:
                nu_algo = nu
                step_size_i = step_size
                max_it = max_iter
            alpha_opt_prev = alpha_opt 
            BTb_ITbI_nuw = BTb + sigma * ITbI + w_opt / nu_algo
            lam_nu = (lam + 1.0 / nu_algo)
            # max_it = max_iter + 1
            # print(w_opt, alpha_opt)
            for i in range(max_it + 1):
                vi = contig(alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev))
                alpha_opt_prev = alpha_opt
                alpha_opt = P.dot(
                    vi + step_size_i * (BTb_ITbI_nuw - matmult(BT, matmult(B, vi)) - sigma * matmult(IT, matmult(I, vi)) - lam_nu * vi)
                )
                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0

                # print metrics
                if ((i % print_iter) == 0.0):
                    f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                    f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                    f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                    f_RS.append(np.linalg.norm(np.ravel(alpha_opt - w_opt) ** 2))
                    f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) > threshold))
                    # f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
                    # ind = (j * (max_iter + 1) * rs_max_iter + k * (max_iter + 1) + i) // print_iter
                    print(j, k, i, 
                          f_B[-1], 
                          sigma * f_I[-1], 
                          lam * f_K[-1], 
                          f_RS[-1] / nu, 
                          f_0[-1], ' / ', n
                          )  
                    #if np.allclose(alpha_opt, alpha_opt_prev, rtol=1e-5):
                    #    break

            # now do the prox step
            w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)
            if threshold > 0:
                f_Bw.append(np.linalg.norm(np.ravel(B @ w_opt - b), ord=2) ** 2)
                f_Iw.append(np.linalg.norm(np.ravel(I @ w_opt - b_I)) ** 2)
                f_Kw.append(np.linalg.norm(np.ravel(w_opt)) ** 2) 
                # f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
                f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1)))
                # ind = (j * rs_max_iter + k)
                current_voxels.w_history.append(w_opt)
                current_voxels.alpha_history.append(alpha_opt)
                print('w : ', j, k, i, 
                      f_Bw[-1], 
                      f_Iw[-1] * sigma, 
                      lam * f_Kw[-1], 
                      f_0w[-1], ' / ', n, ', ',
                      )

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    current_voxels.alphas = alpha_opt
    current_voxels.w = np.ravel(w_opt)
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    current_voxels.J_sparse = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    ws = current_voxels.w.reshape(n, num_basis)
    compute_J(current_voxels, alphas, ws)
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(f_B) * nfp, 
            0.5 * np.array(f_K), 
            0.5 * np.array(f_I), 
            0.5 * np.array(f_RS),
            f_0,
            0.5 * 2 * np.array(f_Bw) * nfp, 
            0.5 * np.array(f_Kw), 
            0.5 * np.array(f_Iw)
            )


def ras_minres(
        current_voxels, kappa=0.0, nu=1e20, max_iter=5000, sigma=1.0,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=10,
        alpha0=None, OUT_DIR=''):
    """
        This function performs a relax-and-split algorithm for solving the 
        current voxel problem. The convex part of the solve is done with
        accelerated projected gradient descent and the nonconvex part 
        can be solved with the prox of the group l0 norm... hard 
        thresholding on the whole set of alphas in a grid cell. 
    """
    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = np.zeros((N, 1))  # (np.random.rand(N, 1) - 0.5) * 1e4  

    B = current_voxels.B_matrix
    #S = np.linalg.svd(B, compute_uv=False)
    #plt.figure()
    #plt.semilogy(S, 'ro')
    #plt.show()
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    BTb = (BT @ b).reshape(-1, 1)
    IT = I.T
    ITbI = (IT * b_I).reshape(-1, 1) 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b_B = b.reshape(len(b), 1)
    alpha_opt = alpha_opt.reshape(len(alpha_opt), 1)
    C = current_voxels.C
    K = C.shape[0]
    CT = C.T
    kappa_nu = (kappa + 1.0 / nu)
    scale = 1e6
    A = scale * np.vstack((B, np.sqrt(sigma) * I))
    b = scale * np.vstack((b_B, np.sqrt(sigma) * b_I))
    AT = A.T
    ATb = AT @ b
    #opt_dict = {"A": B, "b": b_B, "A_I": I, "b_I": b_I, "C": C}
    #savemat('optimization_matrices_new.mat', opt_dict)
    # C = csc_matrix(C.shape)
    # exit()

    def A_fun(x):
        alpha = x[:N]
        Ax = np.vstack(((AT @ (A @ alpha) + kappa_nu * alpha + CT @ x[N:])[:, np.newaxis], (C @ alpha)[:, np.newaxis]))
        return Ax

    def A_fun_nu0(x):
        alpha = x[:N]
        Ax = np.vstack(((AT @ (A @ alpha) + kappa * alpha + CT @ x[N:])[:, np.newaxis], (C @ alpha)[:, np.newaxis]))
        return Ax

    A_operator = LinearOperator((N + K, N + K), matvec=A_fun)
    A_operator_nu0 = LinearOperator((N + K, N + K), matvec=A_fun_nu0)

    def callback(xk, w_opt, rhs, A_func, nu_algo):
        alpha = xk[:N, np.newaxis]
        fB = np.linalg.norm(B @ alpha - b_B) ** 2
        fI = sigma * np.linalg.norm(I @ alpha - b_I) ** 2
        fK = kappa * np.linalg.norm(alpha) ** 2 / scale ** 2
        fRS = np.linalg.norm(alpha - w_opt) ** 2 / nu_algo / scale ** 2
        f = fB + fI + fK + fRS
        print(f, fB, fI, fK, fRS,
              np.linalg.norm(C.dot(alpha)),  # / scale ** 2,
              np.linalg.norm(A_func(xk) - rhs))  # / scale ** 2)

    lam_opt = np.zeros((K, 1))
    tol = 1e-22
    f_Bw = []
    f_Iw = []
    f_Kw = []
    f_0w = []
    f_RSw = []
    f_Cw = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    t1 = time.time()
    set_point = True
    w_opt = np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    sys.stdout = open(OUT_DIR + 'minres_output.txt', 'w')
    print('Total ', 'fB ', 'sigma * fI ', 'kappa * fK ', 'fRS/nu ', 'C*alpha ', 'MINRES residual')
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        if j > (2.0 * len(l0_thresholds) / 3.0) and set_point:
            rs_max_iter = 2 * rs_max_iter
            set_point = False
        for k in range(rs_max_iter):
            if (j == 0) and (k == 0) and (len(l0_thresholds) > 1):
                A_op = A_operator_nu0
                A_func = A_fun_nu0 
                nu_algo = 1e100
                max_it = 50000
            else:
                A_op = A_operator
                A_func = A_fun 
                max_it = max_iter
                nu_algo = nu
            rhs = ATb + w_opt / nu_algo
            rhs = np.vstack((rhs.reshape(len(rhs), 1), np.zeros((K, 1))))

            # now MINRES
            #t1 = time.time()
            x0 = np.vstack((alpha_opt, lam_opt)) 
            sys.stdout = open(OUT_DIR + 'minres_output.txt', 'a')
            #alpha_opt, _ = minres(A_op, rhs, x0=x0, tol=tol, maxiter=max_it, callback=callback, print_iter=print_iter)
            alpha_opt, _ = minres(A_op, rhs, x0=x0, tol=tol, maxiter=max_it, callback=partial(callback, w_opt=w_opt, rhs=rhs, A_func=A_func, nu_algo=nu_algo), print_iter=print_iter)
            #t2 = time.time()
            sys.stdout.close()
            sys.stdout = open("/dev/stdout", "w")
            alpha_opt = alpha_opt.reshape(-1, 1)
            lam_opt = alpha_opt[N:]
            alpha_opt = alpha_opt[:N]

            if (threshold > 0) and (k % print_iter == 0):
                f_Bw.append(np.linalg.norm(np.ravel(B @ w_opt - b_B), ord=2) ** 2)
                f_Iw.append(np.linalg.norm(np.ravel(I @ w_opt - b_I)) ** 2 * sigma)
                f_Kw.append(kappa * np.linalg.norm(np.ravel(w_opt)) ** 2 / scale ** 2)
                f_RSw.append(np.linalg.norm(alpha_opt - w_opt) ** 2 / nu_algo / scale ** 2)
                f_Cw.append(np.linalg.norm(C.dot(w_opt)) / scale ** 2)
                # f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
                f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1)))
                current_voxels.w_history.append(w_opt)
                current_voxels.alpha_history.append(alpha_opt)
                print('w : ', j, k, 
                      f_Bw[-1], 
                      f_Iw[-1],
                      f_Kw[-1], 
                      f_RSw[-1],  # / nu,
                      f_Cw[-1],
                      f_0w[-1], ' / ', n, ', ',
                      )

            # now do the prox step
            w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    f0, fB, fI, fK, fRS, fC, fminres = np.loadtxt(OUT_DIR + 'minres_output.txt', skiprows=1, unpack=True)
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    current_voxels.alphas = alpha_opt
    current_voxels.w = np.ravel(w_opt)
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    current_voxels.J_sparse = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    ws = current_voxels.w.reshape(n, num_basis)
    compute_J(current_voxels, alphas, ws)
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(fB) * nfp, 
            0.5 * np.array(fK), 
            0.5 * np.array(fI), 
            0.5 * np.array(fRS),  # / nu,
            f0,
            fC,
            fminres,
            0.5 * 2 * np.array(f_Bw) * nfp, 
            0.5 * np.array(f_Kw), 
            0.5 * np.array(f_Iw),
            0.5 * np.array(f_RSw),
            0.5 * np.array(f_Cw)
            )


def ras_preconditioned_minres(
        current_voxels, kappa=0.0, nu=1e20, max_iter=5000, sigma=1.0,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=10,
        alpha0=None, OUT_DIR=''):
    """
        This function performs a relax-and-split algorithm for solving the 
        current voxel problem. The convex part of the solve is done with
        accelerated projected gradient descent and the nonconvex part 
        can be solved with the prox of the group l0 norm... hard 
        thresholding on the whole set of alphas in a grid cell. 
    """
    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = np.zeros((N, 1))  # (np.random.rand(N, 1) - 0.5) * 1e4  

    B = current_voxels.B_matrix
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    IT = I.T
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b_B = b.reshape(len(b), 1)
    alpha_opt = alpha_opt.reshape(len(alpha_opt), 1)
    kappa_nu = (kappa + 1.0 / nu)
    scale = 1e6
    C = current_voxels.C   # * scale ** 2
    CT = C.T
    K = C.shape[0]
    A = scale * np.vstack((B, np.sqrt(sigma) * I))
    b = scale * np.vstack((b_B, np.sqrt(sigma) * b_I))
    AT = A.T
    ATb = AT @ b

    def A_fun(x):
        alpha = x[:N]
        Ax = np.vstack(((AT @ (A @ alpha) + kappa_nu * alpha + CT @ x[N:])[:, np.newaxis], (C @ alpha)[:, np.newaxis]))
        return Ax

    def A_fun_nu0(x):
        alpha = x[:N]
        Ax = np.vstack(((AT @ (A @ alpha) + kappa * alpha + CT @ x[N:])[:, np.newaxis], (C @ alpha)[:, np.newaxis]))
        return Ax

    A_operator = LinearOperator((N + K, N + K), matvec=A_fun)
    A_operator_nu0 = LinearOperator((N + K, N + K), matvec=A_fun_nu0)

    # setup matrices for the preconditioner
    AAdiag = np.sum(A ** 2, axis=0) + kappa_nu  # diagonal of A'*A + 1/nu*I
    AAdiag_inv_diag = (1 / AAdiag).reshape(-1, 1)
    AAdiag_inv = spdiags(1 / AAdiag.T, 0, N, N)

    # Schur complement is C*inv(A.'*A + (1/mu)*I)*C.'. 
    # Let's use the inverse diagonal of A:
    S = C @ AAdiag_inv @ CT
    perturbation_factor = abs(S).max() * 1e-7
    S = S + perturbation_factor * spdiags(np.ones(K), 0, K, K)  # regularized Schur complement
    S_inv_op = splu(S)

    def M_inv(x):
        # apply inverse of P given by
        # P_minres = [spdiags((AAdiag'),0,n,n), 0;
        #            sparse(k,n),      S ];
        return np.vstack((x[:N, np.newaxis] * AAdiag_inv_diag, S_inv_op.solve(x[N:, np.newaxis])))

    M_operator = LinearOperator((N + K, N + K), matvec=M_inv)

    # repeat with nu -> infinity for first iteration 
    AAdiag = np.sum(A ** 2, axis=0) + kappa
    AAdiag_inv_diag = (1 / AAdiag).reshape(-1, 1)
    AAdiag_inv = spdiags(1 / AAdiag.T, 0, N, N)
    S = C @ AAdiag_inv @ CT
    perturbation_factor = abs(S).max() * 1e-7
    S = S + perturbation_factor * spdiags(np.ones(K), 0, K, K)  # regularized Schur complement
    S_inv_op = splu(S)

    def M_inv_nu0(x):
        return np.vstack((x[:N, np.newaxis] * AAdiag_inv_diag, S_inv_op.solve(x[N:, np.newaxis])))

    M_operator_nu0 = LinearOperator((N + K, N + K), matvec=M_inv_nu0)

    def callback(xk, w_opt, rhs, A_func, nu_algo):
        alpha = xk[:N, np.newaxis]
        fB = np.linalg.norm(B @ alpha - b_B) ** 2
        fI = sigma * np.linalg.norm(I @ alpha - b_I) ** 2
        fK = kappa * np.linalg.norm(alpha) ** 2 / scale ** 2
        fRS = np.linalg.norm(alpha - w_opt) ** 2 / nu_algo / scale ** 2
        f = fB + fI + fK + fRS
        print(f, fB, fI, fK, fRS,
              np.linalg.norm(C.dot(alpha)),  # / scale ** 2,
              np.linalg.norm(A_func(xk) - rhs))  # / scale ** 2)

    # Now prepare to record optimization progress
    lam_opt = np.zeros((K, 1))
    tol = 1e-20
    f_Bw = []
    f_Iw = []
    f_Kw = []
    f_0w = []
    f_RSw = []
    f_Cw = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    t1 = time.time()
    set_point = True
    w_opt = np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    sys.stdout = open(OUT_DIR + 'minres_output.txt', 'w')
    print('Total ', 'fB ', 'sigma * fI ', 'kappa * fK ', 'fRS/nu ', 'C*alpha ', 'MINRES residual')
    sys.stdout.close()
    sys.stdout = open("/dev/stdout", "w")
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        if j > (2.0 * len(l0_thresholds) / 3.0) and set_point:
            # rs_max_iter = 2 * rs_max_iter
            set_point = False
        for k in range(rs_max_iter):
            if (j == 0) and (k == 0) and (len(l0_thresholds) > 1):
                A_op = A_operator_nu0
                A_func = A_fun_nu0 
                M_op = M_operator_nu0
                nu_algo = 1e100
                max_it = 50000
            else:
                A_op = A_operator
                A_func = A_fun 
                M_op = M_operator
                max_it = max_iter
                nu_algo = nu
            rhs = ATb + w_opt / nu_algo
            rhs = np.vstack((rhs.reshape(len(rhs), 1), np.zeros((K, 1))))

            # now MINRES
            t1 = time.time()
            x0 = np.vstack((alpha_opt, lam_opt))
            sys.stdout = open(OUT_DIR + 'minres_output.txt', 'a')
            alpha_opt, _ = minres(A_op, rhs, x0=x0, tol=tol, maxiter=max_it, callback=partial(callback, w_opt=w_opt, rhs=rhs, A_func=A_func, nu_algo=nu_algo), M=M_op, print_iter=print_iter)
            t2 = time.time()
            sys.stdout.close()
            sys.stdout = open("/dev/stdout", "w")
            t3 = t2 - t1
            #print('MINRES took = ', t3, ' s')
            alpha_opt = alpha_opt.reshape(-1, 1)
            lam_opt = alpha_opt[N:]
            alpha_opt = alpha_opt[:N]

            if (threshold > 0) and (k % print_iter == 0):
                f_Bw.append(np.linalg.norm(np.ravel(B @ w_opt - b_B), ord=2) ** 2)
                f_Iw.append(np.linalg.norm(np.ravel(I @ w_opt - b_I)) ** 2 * sigma)
                f_Kw.append(kappa * np.linalg.norm(np.ravel(w_opt)) ** 2 / scale ** 2)
                f_RSw.append(np.linalg.norm(alpha_opt - w_opt) ** 2 / nu_algo / scale ** 2)
                f_Cw.append(np.linalg.norm(C.dot(w_opt)) / scale ** 2)
                # f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
                f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1)))
                current_voxels.w_history.append(w_opt)
                current_voxels.alpha_history.append(alpha_opt)
                print('w : ', j, k, 
                      f_Bw[-1], 
                      f_Iw[-1],
                      f_Kw[-1], 
                      f_RSw[-1],  # / nu,
                      f_Cw[-1],
                      f_0w[-1], ' / ', n, ', ',
                      )

            # now do the prox step
            w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)

    t2 = time.time()
    algo_time = t2 - t1
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    f0, fB, fI, fK, fRS, fC, fminres = np.loadtxt(OUT_DIR + 'minres_output.txt', skiprows=1, unpack=True)
    current_voxels.alphas = alpha_opt
    current_voxels.w = np.ravel(w_opt)
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    current_voxels.J_sparse = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    ws = current_voxels.w.reshape(n, num_basis)
    compute_J(current_voxels, alphas, ws)
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(fB) * nfp, 
            0.5 * np.array(fK), 
            0.5 * np.array(fI), 
            0.5 * np.array(fRS),  # / nu,
            f0,
            fC,
            fminres,
            0.5 * 2 * np.array(f_Bw) * nfp, 
            0.5 * np.array(f_Kw), 
            0.5 * np.array(f_Iw),
            0.5 * np.array(f_RSw),
            0.5 * np.array(f_Cw),
            algo_time
            )
