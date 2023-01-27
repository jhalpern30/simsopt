import numpy as np
import simsoptpp as sopp
import time

__all__ = ['projected_gradient_descent_Tikhonov']


def projected_gradient_descent_Tikhonov(winding_volume, lam=0.0, alpha0=None, max_iter=5000, P=None, acceleration=True):
    """
        This function performed projected gradient descent for the Tikhonov-regularized
        winding volume optimization problem (if the flux jump constraints are not used,
        then it reduces to simple gradient descent). Without constraints, flux jumps
        across cells are not constrained and therefore J is not globally
        divergence-free (although J is locally divergence-free in each cell). 
        This function is primarily for making sure this sub-problem of the
        winding volume optimization is working correctly. Note that this function
        uses gradient descent even for the unconstrained problem
        because the matrices are too big for np.linalg.solve. Also, by default it 
        uses Nesterov acceleration. 

        Args:
            P : projection matrix onto the linear equality constraints representing
                the flux jump constraints. If C is the flux jump constraint matrix,
                P = I - C.T @ (C * C.T)^{-1} @ C. P is very large but very sparse
                so is assumed to be stored as a scipy csc_matrix. 
    """
    t1 = time.time()
    B = winding_volume.B_matrix
    I = winding_volume.Itarget_matrix
    BT = B.T
    # BTB = BT @ B
    IT = I.T
    # ITI = IT @ I
    #L = np.linalg.svd(BTB + ITI + lam * np.eye(B.shape[1]), compute_uv=False)[0]
    L = np.linalg.svd(BT @ B + IT @ I + lam * np.eye(B.shape[1]), compute_uv=False)[0]
    #cond_num = np.linalg.cond(BTB + ITI + lam * np.eye(B.shape[1]))
    # cond_num = np.linalg.cond(BT @ B + IT @ I + lam * np.eye(B.shape[1]))
    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size, condition number = ', L, step_size)  # , cond_num)
    b = winding_volume.b_rhs
    b_I = winding_volume.Itarget_rhs
    BTb = BT @ b
    ITbI = IT * b_I 
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    alpha_opt = (np.random.rand(n * num_basis) - 0.5) * 1e3  # set some initial guess

    # Initial guess with projected gradient descent must start inside the
    # feasible region (the area allowed by the equality constraints)
    if P is not None:
        alpha_opt = P.dot(alpha_opt)
    t2 = time.time()
    print('Time to setup the algo = ', t2 - t1, ' s')
    contig = np.ascontiguousarray
    #t1 = time.time()
    #Pdense = P.todense()
    #print(Pdense.shape, B.shape, I.shape, BTb.shape, ITbI.shape)
    #alpha_opt_cpp = sopp.acc_prox_grad_descent(contig(Pdense), contig(B), contig(I), contig(BTb), contig(ITbI), lam, step_size, max_iter)
    #t2 = time.time()
    #print('Time to run algo in C++ = ', t2 - t1, ' s')
    #print('f_B from c++ calculation = ', 0.5 * nfp * np.linalg.norm(B @ alpha_opt_cpp - b, ord=2) ** 2)
    f_B = []
    f_I = []
    f_K = []
    print_iter = 200
    t1 = time.time()
    if P is None:
        if acceleration:  # Nesterov acceleration 
            # first iteration do regular GD 
            alpha_opt_prev = alpha_opt
            step_size_i = step_size
            alpha_opt = alpha_opt + step_size_i * (
                BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt
            )
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_I.append((I @ alpha_opt - b_I) ** 2)
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
            for i in range(1, max_iter):
                vi = alpha_opt + (i - 1) / (i + 2) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                alpha_opt = vi + step_size_i * (BTb + ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi)
                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
                f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
                f_I.append((I @ alpha_opt - b_I) ** 2)
                f_K.append(np.linalg.norm(alpha_opt, ord=2))
                if (i % print_iter) == 0.0:
                    print(i, f_B[i] ** 2 * nfp * 0.5, f_I[i], f_K[i])
        else:
            for i in range(max_iter):
                alpha_opt = alpha_opt + step_size * (BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt)
                f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
                f_I.append((I @ alpha_opt - b_I) ** 2)
                f_K.append(np.linalg.norm(alpha_opt, ord=2))
                if (i % print_iter) == 0.0:
                    print(i, f_B[i] ** 2 * nfp * 0.5, f_I[i], f_K[i])
    else:
        if acceleration:  # Nesterov acceleration 
            # first iteration do regular GD 
            alpha_opt_prev = alpha_opt
            step_size_i = step_size
            alpha_opt = P.dot(alpha_opt + step_size_i * (
                BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt
            )   
            )
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_I.append((I @ alpha_opt - b_I) ** 2)
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
            print('0', f_B[0] ** 2 * nfp * 0.5, f_I[0], f_K[0])
            for i in range(1, max_iter):
                vi = alpha_opt + (i - 1) / (i + 2) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                #alpha_opt = P.dot(vi + step_size_i * (BTb + ITbI - (BTB + ITI + lam * ident) @ vi))
                alpha_opt = P.dot(vi + step_size_i * (BTb + ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi))

                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
                if (i % print_iter) == 0.0:
                    f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
                    f_I.append((I @ alpha_opt - b_I) ** 2)
                    f_K.append(np.linalg.norm(alpha_opt, ord=2))
                    print(i, f_B[i // print_iter] ** 2 * nfp * 0.5, f_I[i // print_iter], f_K[i // print_iter])
        else:
            for i in range(max_iter):
                alpha_opt = P.dot(alpha_opt + step_size * (BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt))
                if (i % print_iter) == 0.0:
                    f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
                    f_I.append((I @ alpha_opt - b_I) ** 2)
                    f_K.append(np.linalg.norm(alpha_opt, ord=2))
                    print(i, f_B[i // print_iter] ** 2 * nfp * 0.5, f_I[i // print_iter], f_K[i // print_iter])
    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    winding_volume.alphas = alpha_opt 
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = alpha_opt.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return alpha_opt, 0.5 * np.array(f_B) ** 2 * nfp, 0.5 * np.array(f_K) ** 2, 0.5 * np.array(f_I)
