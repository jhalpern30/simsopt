import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from sklearn.linear_model import Lasso
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.field.currentpotential import CurrentPotentialFourier
from scipy.io import netcdf_file
from simsopt.field.magneticfieldclasses import WindingSurfaceField
import warnings

__all__ = ["CurrentPotentialSolve"]


class CurrentPotentialSolve:
    """
    Current Potential Solve object is designed for performing
    the winding surface coil optimization. We provide functionality
    for the REGCOIL (tikhonov regularization) and L1 norm (Lasso)
    variants of the problem.

    Args:
        cp: CurrentPotential class object containing the winding surface.
        plasma_surface: The plasma surface to optimize Bnormal over.
        Bnormal_plasma: Bnormal coming from plasma currents.
        B_GI: Bnormal coming from the net coil currents.
    """

    def __init__(self, cp, plasma_surface, Bnormal_plasma, B_GI):
        self.current_potential = cp
        self.winding_surface = self.current_potential.winding_surface
        self.ntheta_coil = len(self.current_potential.quadpoints_theta)
        self.nphi_coil = len(self.current_potential.quadpoints_phi)
        self.ndofs = self.current_potential.num_dofs()
        self.plasma_surface = plasma_surface
        self.Bnormal_plasma = Bnormal_plasma
        self.B_GI = B_GI
        self.ilambdas_l2 = []
        self.dofs_l2 = []
        self.phis_l2 = []
        self.K2s_l2 = []
        self.fBs_l2 = []
        self.fKs_l2 = []
        self.ilambdas_l1 = []
        self.dofs_l1 = []
        self.phis_l1 = []
        self.K2s_l1 = []
        self.fBs_l1 = []
        self.fKs_l1 = []
        warnings.warn(
            "Beware: the f_B (also called chi^2_B) computed from the "
            "CurrentPotentialSolve class will be slightly different than "
            "the f_B computed using SquaredFlux with the BiotSavart law "
            "implemented in WindingSurfaceField. This is because the "
            "optimization formulation and the full BiotSavart calculation "
            "are discretized in different ways. This disagreement will "
            "worsen at low regularization, but improve with higher "
            "resolution on the plasma and coil surfaces. "
        )

    @classmethod
    def from_netcdf(cls, filename: str):
        """
        Initialize a CurrentPotentialSolve using a CurrentPotentialFourier
        from a regcoil netcdf output file. The single_valued_current_potential_mn
        are set to zero.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
        """
        f = netcdf_file(filename, 'r')
        nfp = f.variables['nfp'][()]
        Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
        rmnc_plasma = f.variables['rmnc_plasma'][()]
        zmns_plasma = f.variables['zmns_plasma'][()]
        xm_plasma = f.variables['xm_plasma'][()]
        xn_plasma = f.variables['xn_plasma'][()]
        mpol_plasma = int(np.max(xm_plasma))
        ntor_plasma = int(np.max(xn_plasma)/nfp)
        ntheta_plasma = f.variables['ntheta_plasma'][()]
        nzeta_plasma = f.variables['nzeta_plasma'][()]
        if ('rmns_plasma' in f.variables and 'zmnc_plasma' in f.variables):
            rmns_plasma = f.variables['rmns_plasma'][()]
            zmnc_plasma = f.variables['zmnc_plasma'][()]
            if np.all(zmnc_plasma == 0) and np.all(rmns_plasma == 0):
                stellsym_plasma_surf = True
            else:
                stellsym_plasma_surf = False
        else:
            rmns_plasma = np.zeros_like(rmnc_plasma)
            zmnc_plasma = np.zeros_like(zmns_plasma)
            stellsym_plasma_surf = True
        f.close()

        cp = CurrentPotentialFourier.from_netcdf(filename)

        s_plasma = SurfaceRZFourier(
            nfp=nfp,
            mpol=mpol_plasma,
            ntor=ntor_plasma,
            stellsym=stellsym_plasma_surf
        )
        s_plasma = s_plasma.from_nphi_ntheta(
            nfp=nfp, ntheta=ntheta_plasma,
            nphi=nzeta_plasma,
            mpol=mpol_plasma, ntor=ntor_plasma,
            stellsym=stellsym_plasma_surf, range="field period"
        )
        s_plasma.set_dofs(0 * s_plasma.get_dofs())
        for im in range(len(xm_plasma)):
            s_plasma.set_rc(xm_plasma[im], int(xn_plasma[im] / nfp), rmnc_plasma[im])
            s_plasma.set_zs(xm_plasma[im], int(xn_plasma[im] / nfp), zmns_plasma[im])
            if not stellsym_plasma_surf:
                s_plasma.set_rs(xm_plasma[im], int(xn_plasma[im] / nfp), rmns_plasma[im])
                s_plasma.set_zc(xm_plasma[im], int(xn_plasma[im] / nfp), zmnc_plasma[im])

        cp_copy = CurrentPotentialFourier.from_netcdf(filename)
        Bfield = WindingSurfaceField(cp_copy)
        points = s_plasma.gamma().reshape(-1, 3)
        Bfield.set_points(points)
        B = Bfield.B()
        normal = s_plasma.unitnormal().reshape(-1, 3)
        B_GI_winding_surface = np.sum(B*normal, axis=1)
        return cls(cp, s_plasma, np.ravel(Bnormal_from_plasma_current), B_GI_winding_surface)

    def write_current_potential_to_regcoil(self, filename: str, opt_type='L2'):
        """
        Set phic and phis based on a regcoil netcdf file.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
            ilambda: 0-based index for the lambda array, indicating which current
                potential solution to use
        """
        if opt_type not in ['L2', 'L1']:
            raise ValueError('opt_type parameter must be L2 or L1')

        f = netcdf_file(filename + '_' + opt_type, 'w')
        f.variables['nfp'][()] = self.plasma_surface.nfp
        f.variables['mpol_plasma'][()] = self.plasma_surface.mpol
        f.variables['ntor_plasma'][()] = self.plasma_surface.ntor
        f.variables['xm_plasma'][()] = self.plasma_surface.m
        f.variables['xn_plasma'][()] = self.plasma_surface.n
        f.variables['mpol_potential'][()] = self.winding_surface.mpol
        f.variables['ntor_potential'][()] = self.winding_surface.ntor
        f.variables['xm_potential'][()] = self.winding_surface.m
        f.variables['xn_potential'][()] = self.winding_surface.n
        f.variables['symmetry_option'][()] = self.plasma_surface.stellsym
        f.variables['net_poloidal_current_Amperes'][()] = self.current_potential.net_poloidal_current_amperes
        f.variables['net_toroidal_current_Amperes'][()] = self.current_potential.net_toroidal_current_amperes

        f.variables['ntheta_plasma'][()] = self.plasma_surface.ntheta
        f.variables['nzeta_plasma'][()] = self.plasma_surface.nzeta 
        f.variables['ntheta_coil'][()] = self.winding_surface.ntheta
        f.variables['nzeta_coil'][()] = self.winding_surface.nzeta
        f.variables['Bnormal_from_plasma_current'][()] = self.Bnormal_plasma
        f.variables['Bnormal_from_net_coil_currents'][()] = self.B_GI
        f.variables['norm_normal_plasma'][()] = np.linal.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1) / (2 * np.pi * 2 * np.pi)
        f.variables['norm_normal_coil'][()] = np.linal.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1) / (2 * np.pi * 2 * np.pi)

        rmnc = np.zeros(self.plasma_surface.mpol)
        zmns = np.zeros(self.plasma_surface.mpol)
        rmns = np.zeros(self.plasma_surface.mpol)
        zmnc = np.zeros(self.plasma_surface.mpol)
        xn_coil = self.plasma_surface.n
        xm_coil = self.plasma_surface.m
        nfp = self.plasma_surface.nfp
        for im in range(self.plasma_surface.mpol):
            rmnc[im] = self.plasma_surface.get_rc(xm_coil[im], int(xn_coil[im]/nfp))
            zmns[im] = self.plasma_surface.get_zs(xm_coil[im], int(xn_coil[im]/nfp))
            if not self.plasma_surface.stellsym:
                rmns[im] = self.plasma_surface.get_rs(xm_coil[im], int(xn_coil[im]/nfp))
                zmnc[im] = self.plasma_surface.get_zc(xm_coil[im], int(xn_coil[im]/nfp))
        f.variables['rmnc_plasma'][()] = rmnc
        f.variables['zmns_plasma'][()] = zmns
        f.variables['rmns_plasma'][()] = rmns
        f.variables['zmnc_plasma'][()] = zmnc

        rmnc = np.zeros(self.winding_surface.mpol)
        zmns = np.zeros(self.winding_surface.mpol)
        rmns = np.zeros(self.winding_surface.mpol)
        zmnc = np.zeros(self.winding_surface.mpol)
        xn_coil = self.winding_surface.n
        xm_coil = self.winding_surface.m
        nfp = self.winding_surface.nfp
        for im in range(self.winding_surface.mpol):
            rmnc[im] = self.winding_surface.get_rc(xm_coil[im], int(xn_coil[im]/nfp))
            zmns[im] = self.winding_surface.get_zs(xm_coil[im], int(xn_coil[im]/nfp))
            if not self.winding_surface.stellsym:
                rmns[im] = self.winding_surface.get_rs(xm_coil[im], int(xn_coil[im]/nfp))
                zmnc[im] = self.winding_surface.get_zc(xm_coil[im], int(xn_coil[im]/nfp))
        f.variables['rmnc_coil'][()] = rmnc
        f.variables['zmns_coil'][()] = zmns
        f.variables['rmns_coil'][()] = rmns
        f.variables['zmnc_coil'][()] = zmnc

        f.variables['RHS_B'][()], _ = self.B_matrix_and_rhs()
        f.variables['RHS_regularization'][()] = self.K_rhs()

        points = self.plasma_surface.gamma().reshape(-1, 3)
        normal = self.plasma_surface.normal().reshape(-1, 3)
        ws_points = self.winding_surface.gamma().reshape(-1, 3)
        ws_normal = self.winding_surface.normal().reshape(-1, 3)
        dtheta_coil = self.winding_surface.quadpoints_theta[1]
        dzeta_coil = self.winding_surface.quadpoints_phi[1]

        if opt_type == 'L2' and len(self.ilambdas_l2) > 0:
            for i, ilambda in enumerate(self.ilambdas_l2):
                f.variables['single_valued_current_potential_mn'][()][i, :] = self.dofs_l2[i]
                f.variables['single_valued_current_potential_thetazeta'][()][i, :, :] = self.phis_l2[i]
                f.variables['K2'][()][i, :] = self.K2s_l2[i]
                f.variables['lambda'][()][i] = ilambda
                f.variables['chi2_B'][()][i] = 2 * self.fBs_l2[i]
                f.variables['chi2_k'][()][i] = 2 * self.fKs_l2[i]
                Bnormal_regcoil_sv = sopp.WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, self.phis_l2[i], normal) * dtheta_coil * dzeta_coil
                Bnormal_regcoil_total = Bnormal_regcoil_sv + self.B_GI + self.Bnormal_plasma
                f.variables['Bnormal_total'][()][i, :, :] = Bnormal_regcoil_total
        if opt_type == 'L1' and len(self.ilambdas_l1) > 0:
            for i, ilambda in enumerate(self.ilambdas_l1):
                f.variables['single_valued_current_potential_mn'][()][ilambda, :] = self.dofs_l1[ilambda]
                f.variables['single_valued_current_potential_thetazeta'][()][i, :, :] = self.phis_l1[i]
                f.variables['K2'][()][ilambda, :, :] = self.K2s_l1[i]
                f.variables['lambda'][()][i] = ilambda
                f.variables['chi2_B'][()][i] = 2 * self.fBs_l1[i]
                f.variables['chi2_k'][()][i] = 2 * self.fKs_l1[i]
                Bnormal_regcoil_sv = sopp.WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, self.phis_l1[i], normal) * dtheta_coil * dzeta_coil
                Bnormal_regcoil_total = Bnormal_regcoil_sv + self.B_GI + self.Bnormal_plasma
                f.variables['Bnormal_total'][()][i, :, :] = Bnormal_regcoil_total
        f.close()

    def K_rhs_impl(self, K_rhs):
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, normal)
        K_rhs *= self.winding_surface.quadpoints_theta[1] * self.winding_surface.quadpoints_phi[1] / self.winding_surface.nfp

    def K_rhs(self):
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        self.K_rhs_impl(K_rhs)
        return K_rhs

    def K_matrix_impl(self, K_matrix):
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, normal)
        K_matrix *= self.winding_surface.quadpoints_theta[1] * self.winding_surface.quadpoints_phi[1] / self.winding_surface.nfp

    def K_matrix(self):
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        self.K_matrix_impl(K_matrix)
        return K_matrix

    def B_matrix_and_rhs(self):
        """
            Compute the matrices and right-hand-side corresponding the Bnormal part of
            the optimization, for both the Tikhonov and Lasso optimizations.
        """
        plasma_surface = self.plasma_surface
        normal = self.winding_surface.normal().reshape(-1, 3)
        Bnormal_plasma = self.Bnormal_plasma
        normal_plasma = plasma_surface.normal().reshape(-1, 3)
        points_plasma = plasma_surface.gamma().reshape(-1, 3)
        points_coil = self.winding_surface.gamma().reshape(-1, 3)
        theta = self.winding_surface.quadpoints_theta
        phi_mesh, theta_mesh = np.meshgrid(self.winding_surface.quadpoints_phi, theta, indexing='ij')
        zeta_coil = np.ravel(phi_mesh)
        theta_coil = np.ravel(theta_mesh)

        # Compute terms for the REGCOIL (L2) problem
        gj, B_matrix = sopp.winding_surface_field_Bn(
            points_plasma,
            points_coil,
            normal_plasma,
            normal,
            self.current_potential.stellsym,
            zeta_coil,
            theta_coil,
            self.current_potential.num_dofs(),
            self.current_potential.m,
            self.current_potential.n,
            self.winding_surface.nfp
        )
        B_GI = self.B_GI

        # set up RHS of optimization
        b_rhs = - np.ravel(B_GI + Bnormal_plasma) @ gj
        dzeta_plasma = (plasma_surface.quadpoints_phi[1] - plasma_surface.quadpoints_phi[0])
        dtheta_plasma = (plasma_surface.quadpoints_theta[1] - plasma_surface.quadpoints_theta[0])
        dzeta_coil = (self.winding_surface.quadpoints_phi[1] - self.winding_surface.quadpoints_phi[0])
        dtheta_coil = (self.winding_surface.quadpoints_theta[1] - self.winding_surface.quadpoints_theta[0])

        # scale bmatrix and b_rhs by factors of the grid spacing
        b_rhs = b_rhs * dzeta_plasma * dtheta_plasma * dzeta_coil * dtheta_coil
        B_matrix = B_matrix * dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2
        normN = np.linalg.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1)
        self.gj = gj * np.sqrt(dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2)
        self.b_e = - np.sqrt(normN * dzeta_plasma * dtheta_plasma) * (B_GI + Bnormal_plasma)

        normN = np.linalg.norm(self.winding_surface.normal().reshape(-1, 3), axis=-1)
        dr_dzeta = self.winding_surface.gammadash1().reshape(-1, 3)
        dr_dtheta = self.winding_surface.gammadash2().reshape(-1, 3)
        G = self.current_potential.net_poloidal_current_amperes
        I = self.current_potential.net_toroidal_current_amperes

        normal_coil = self.winding_surface.normal().reshape(-1, 3)
        m = self.current_potential.m
        n = self.current_potential.n
        nfp = self.winding_surface.nfp

        contig = np.ascontiguousarray

        # Compute terms for the Lasso (L1) problem
        d, fj = sopp.winding_surface_field_K2_matrices(
            contig(dr_dzeta), contig(dr_dtheta), contig(normal_coil), self.current_potential.stellsym,
            contig(zeta_coil), contig(theta_coil), self.ndofs, contig(m), contig(n), nfp, G, I
        )
        self.fj = fj * 2 * np.pi * np.sqrt(dzeta_coil * dtheta_coil)
        self.d = d * 2 * np.pi * np.sqrt(dzeta_coil * dtheta_coil)
        return b_rhs, B_matrix

    def solve_tikhonov(self, lam=0):
        """
            Solve the REGCOIL problem -- winding surface optimization with
            the L2 norm. This is tested against REGCOIL runs extensively in
            tests/field/test_regcoil.py.
        """
        K_matrix = self.K_matrix()
        K_rhs = self.K_rhs()
        b_rhs, B_matrix = self.B_matrix_and_rhs()

        # least-squares solve
        phi_mn_opt = np.linalg.solve(B_matrix + lam * K_matrix, b_rhs + lam * K_rhs)
        self.current_potential.set_dofs(phi_mn_opt)

        # Get other matrices for direct computation of fB and fK loss terms
        nfp = self.plasma_surface.nfp
        normN = np.linalg.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1)
        A_times_phi = self.gj @ phi_mn_opt / np.sqrt(normN)
        b_e = self.b_e
        Ak_times_phi = self.fj @ phi_mn_opt
        f_B = 0.5 * np.linalg.norm(A_times_phi - b_e) ** 2 * nfp
        f_K = 0.5 * np.linalg.norm(Ak_times_phi - self.d) ** 2
        self.ilambdas_l2.append(lam)
        self.dofs_l2.append(phi_mn_opt)
        self.phis_l2.append(self.current_potential.Phi())
        self.fBs_l2.append(f_B)
        self.fKs_l2.append(f_K)
        K2 = np.sum(self.current_potential.K() ** 2, axis=2)
        self.K2s_l2.append(K2)
        return phi_mn_opt, f_B, f_K

    def solve_lasso(self, lam=0, max_iter=1000, acceleration=True):
        """
            Solve the Lasso problem -- winding surface optimization with
            the L1 norm, which should tend to allow stronger current
            filaments to form than the L2. There are a couple changes to make:

            1. Need to define new optimization variable z = A_k * phi_mn - b_k
               so that optimization becomes
               ||AA_k^{-1} * z - (b - A * A_k^{-1} * b_k)||_2^2 + alpha * ||z||_1
               which is the form required to use the Lasso pre-built optimizer
               from sklearn (which actually works poorly) or the optimizer used
               here (proximal gradient descent for LASSO, also called ISTA or FISTA).
            2. The alpha term should be similar amount of regularization as the L2
               so we rescale lam -> sqrt(lam) since lam is used for the (L2 norm)^2
               loss term used for Tikhonov regularization.


            We use the FISTA algorithm but you could use scikit-learn's Lasso
            optimizer too. Like any gradient-based algorithm, both FISTA
            and Lasso will work very poorly at low regularization since convergence
            goes like the condition number of the fB matrix. In both cases,
            this can be addressed by using the exact Tikhonov solution (obtained
            with a matrix inverse instead of a gradient-based optimization) as
            an initial guess to the optimizers.
        """
        # Set up some matrices
        _, _ = self.B_matrix_and_rhs()
        normN = np.linalg.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1)
        A_matrix = self.gj
        for i in range(self.gj.shape[0]):
            A_matrix[i, :] *= (1.0 / np.sqrt(normN[i]))
        b_e = self.b_e
        Ak_matrix = self.fj.reshape(self.fj.shape[0] * 3, self.fj.shape[-1])
        d = np.ravel(self.d)
        nfp = self.plasma_surface.nfp

        # Ak is non-square so pinv required. Careful with rcond parameter
        Ak_inv = np.linalg.pinv(Ak_matrix, rcond=1e-30)
        A_new = A_matrix @ Ak_inv
        b_new = b_e - A_new @ d

        # rescale the l1 regularization
        l1_reg = lam
        # l1_reg = np.sqrt(lam)

        # if alpha << 1, want to use initial guess from the Tikhonov solve,
        # which is exact since it comes from a matrix inverse.
        phi0, _, _, = self.solve_tikhonov(lam=lam)
        z0 = Ak_matrix @ phi0 - d
        z_opt, z_history = self._FISTA(A=A_new, b=b_new, alpha=l1_reg, max_iter=max_iter, acceleration=acceleration, xi0=z0)

        # Compute the history of values from the optimizer
        phi_history = []
        fB_history = []
        fK_history = []
        for i in range(len(z_history)):
            phi_history.append(Ak_inv @ (z_history[i] + d))
            fB_history.append(0.5 * np.linalg.norm(A_matrix @ phi_history[i] - b_e) ** 2 * nfp)
            fK_history.append(np.linalg.norm(z_history[i], ord=1))
            # fK_history.append(np.linalg.norm(Ak_matrix @ phi_history[i] - d, ord=1))

        # Remember, Lasso solved for z = A_k * phi_mn - b_k so need to convert back
        phi_mn_opt = Ak_inv @ (z_opt + d)

        self.current_potential.set_dofs(phi_mn_opt)
        f_B = 0.5 * np.linalg.norm(A_matrix @ phi_mn_opt - b_e) ** 2 * nfp
        f_K = np.linalg.norm(Ak_matrix @ phi_mn_opt - d, ord=1)
        self.ilambdas_l1.append(lam)
        self.dofs_l1.append(phi_mn_opt)
        self.phis_l1.append(self.current_potential.Phi())
        self.fBs_l1.append(f_B)
        self.fKs_l1.append(f_K)
        K2 = np.sum(self.current_potential.K() ** 2, axis=2)
        self.K2s_l1.append(K2)
        return phi_mn_opt, f_B, f_K, fB_history, fK_history

    def _FISTA(self, A, b, alpha=0.0, max_iter=1000, acceleration=True, xi0=None):
        """
            This function uses Nesterov's accelerated proximal
            gradient descent algorithm to solve the Lasso
            (L1-regularized) winding surface problem. This is
            usually called fast iterative soft-thresholding algorithm
            (FISTA). If acceleration = False, it will use the ISTA
            algorithm, which tends to converge much slower than FISTA
            but is a true descent algorithm, unlike FISTA.
        """

        # if abs(current_potential_{k+1} - current_potential_{k}) < tol
        # for every element, then algorithm quits in this k-th iteration.
        tol = 1e1

        # pre-compute/load some stuff so for loops (below) are faster
        ATA = A.T @ A
        ATb = A.T @ b
        prox = self._prox_l1

        # An upper bound on the
        # Lipshitz constant L of the least-squares loss term
        # can be computed easily as largest eigenvalue of ATA
        L = np.linalg.svd(ATA, compute_uv=False)[0]

        # initial step size should be just smaller than 1 / L
        # which for most of these problems L ~ 1e-13 or smaller
        # so the step size is enormous
        ti = 1.0 / L

        # initialize current potential to random values in [5e-4, 5e4]
        if xi0 is None:
            xi0 = (np.random.rand(A.shape[1]) - 0.5) * 1e5
        x_history = [xi0]
        if acceleration:  # FISTA algorithm
            # first iteration do ISTA
            x_history.append(prox(xi0 + ti * (ATb - ATA @ xi0), ti * alpha))
            for i in range(1, max_iter):
                vi = x_history[i] + (i - 1) / (i + 2) * (x_history[i] - x_history[i - 1])
                # note l1 'threshold' is rescaled here
                x_history.append(prox(vi + ti * (ATb - ATA @ vi), ti * alpha))
                ti = (1 + np.sqrt(1 + 4 * ti ** 2)) / 2.0
                if (i % 100) == 0:
                    if np.all(abs(x_history[i + 1] - x_history[i]) < tol):
                        break
        else:  # ISTA algorithm
            alpha = ti * alpha  # ti does not vary in ISTA algorithm
            ATA = ti * ATA
            I_ATA = np.eye(ATA.shape[0]) - ATA
            ATb = ti * ATb
            for i in range(max_iter):
                x_history.append(prox(ATb + I_ATA @ x_history[i], alpha))
                if (i % 100) == 0:
                    if np.all(abs(x_history[i + 1] - x_history[i]) < tol):
                        break
        xi = x_history[-1]
        return xi, x_history

    def _prox_l1(self, x, threshold):
        """
            Proximal operator for L1 regularization,
            which is often called soft-thresholding.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
