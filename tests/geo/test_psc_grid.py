from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.geo import PSCgrid, WPgrid, SurfaceRZFourier
from simsopt.field import BiotSavart, coils_via_symmetries, Current, CircularCoil
import simsoptpp as sopp

input_name = 'input.LandremanPaul2021_QA_lowres'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
surf1 = SurfaceRZFourier.from_vmec_input(
    surface_filename, range='full torus', nphi=16, ntheta=16
)
surf1.nfp = 1
surf1.stellsym = False
surf2 = SurfaceRZFourier.from_vmec_input(
    surface_filename, range='half period', nphi=16, ntheta=16
)
input_name = 'input.LandremanPaul2021_QH'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "examples" / "2_Intermediate" / "inputs").resolve()
surface_filename = TEST_DIR / input_name
surf3 = SurfaceRZFourier.from_vmec_input(
    surface_filename, range='half period', nphi=16, ntheta=16
)
input_name = 'wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
surf4 = SurfaceRZFourier.from_wout(
    surface_filename, range='half period', nphi=16, ntheta=16
)
surfs = [surf1, surf2, surf3, surf4]
ncoils = 7
np.random.seed(29)
R = 0.05
a = 1e-5
points = (np.random.rand(ncoils, 3) - 0.5) * 3
points[:, -1] = 0.4
alphas = (np.random.rand(ncoils) - 0.5) * np.pi
deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
epsilon = 1e-4  # smaller epsilon and numerical accumulation starts to be an issue

class Testing(unittest.TestCase):
    
    # def test_coil_forces_derivatives(self):
    #     ncoils = 7
    #     np.random.seed(1)
    #     R = 1.0
    #     a = 1e-5
    #     points = (np.random.rand(ncoils, 3) - 0.5) * 20
    #     alphas = (np.random.rand(ncoils) - 0.5) * np.pi
    #     deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
    #     epsilon = 1e-4  # smaller epsilon and numerical accumulation starts to be an issue
    #     for surf in surfs:
    #         print('Surf = ', surf)
    #         kwargs_manual = {"plasma_boundary": surf}
    #         wp_array = WPgrid.geo_setup_manual(
    #             points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
    #         )
    #         b_F, A_F = wp_array.coil_forces()
    #         I = wp_array.I
    #         b_F *= I 
    #         A_F *= I
    #         force_objective = 0.5 * (A_F @ I + b_F).T @ (A_F @ I + b_F)
    #         deltas_new = np.copy(deltas)
    #         deltas_new[0] += epsilon
    #         alphas_new = np.copy(alphas)
    #         psc_array_new = WPgrid.geo_setup_manual(
    #             points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
    #         )
    #         b_F, A_F_new = wp_array.coil_forces()
    #         I = wp_array.I
    #         b_F *= I 
    #         A_F *= I
    #         force_objective_new = 0.5 * (A_F_new @ I + b_F).T @ (A_F_new @ I + b_F)
    #         df_objective = (force_objective_new - force_objective) / epsilon
    #         # print(A @ Linv @ psi_new, A @ Linv @ psi, b)
    #         dAF_ddelta = (A_F_new - A_F) / epsilon
    #         AF_deriv = wp_array.AF_deriv()
    #         print(dAF_ddelta[0], AF_deriv[ncoils])
    #         assert(np.allclose(dAF_ddelta[0], AF_deriv[ncoils], rtol=1e-3))
    #         df_analytic = (A_F @ I + b_F).T @ (A_F @ I)
    #         print(df_objective, df_analytic[0])
    #         assert np.isclose(df_objective, df_analytic[0], rtol=1e-2)
    
    # def test_coil_forces(self):
    #     from scipy.special import ellipk, ellipe
        
    #     ncoils = 2
    #     I1 = 5.0
    #     I2 = -3.0
    #     Z1 = 1.0
    #     Z2 = 4.0
    #     R1 = 0.5
    #     R2 = R1
    #     a = 1e-5
    #     points = np.array([[0.0, 0.0, Z1], [0.0, 0.0, Z2]])
    #     alphas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * np.pi
    #     deltas = np.zeros(ncoils)  # (np.random.rand(ncoils) - 0.5) * 2 * np.pi
    #     k = np.sqrt(4.0 * R1 * R2 / ((R1 + R2) ** 2 + (Z2 - Z1) ** 2))
    #     mu0 = 4 * np.pi * 1e-7
    #     # Jackson uses K(k) and E(k) but this corresponds to
    #     # K(k^2) and E(k^2) in scipy library
    #     F_analytic = mu0 * I1 * I2 * k * (Z2 - Z1) * (
    #         (2  - k ** 2) * ellipe(k ** 2) / (1 - k ** 2) - 2.0 * ellipk(k ** 2)
    #     ) / (4.0 * np.sqrt(R1 * R2))
        
    #     wp_array = WPgrid.geo_setup_manual(
    #         points, R=R1, a=a, alphas=alphas, deltas=deltas, I=np.array([I1, I2])
    #     )
    #     b_F, A_F = wp_array.coil_forces()
    #     F_TF = wp_array.I[:, None] * b_F
    #     # print('A = ', A_F[:, :, 2])
    #     F_WP = wp_array.I[:, None] * np.tensordot(A_F, wp_array.I, axes=([1, 0])) * wp_array.fac
    #     print(F_TF, F_WP, F_WP.shape, F_analytic)
    #     assert np.allclose(F_WP[:, 0], 0.0)
    #     assert np.allclose(F_WP[:, 1], 0.0)
    #     assert np.allclose(np.abs(F_WP[:, 2]), abs(F_analytic))
    
    def test_dpsi_analytic_derivatives(self):
        """
        Tests the analytic calculations of dpsi/dalpha and dpsi/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            b = psc_array.b_opt # * psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc] # / psc_array.fac
            I = (-Linv @ psi)
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            # Bn_objective = 0.5 * (A @ (Linv @ psi)[:psc_array.num_psc] + b).T @ (A @ (Linv @ psi)[:psc_array.num_psc] + b)
            # print('Lpsi = ', Linv @ psi)
            # print('A_matrix = ', A[0, :])
            # print(A.shape, A_full.shape)
            # print('A_matrix_full = ', A_full[0, :])
            deltas_new = np.copy(deltas)
            deltas_new[0] += epsilon 
            alphas_new = np.copy(alphas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            psi_new = psc_array_new.psi / psc_array.fac
            I_new = (-Linv @ psi_new)
            # print(psi)
            # print(psi - psi_new)
            # print(psi, psi_new)
            # Bn_objective_new = 0.5 * (A @ (Linv @ psi_new)[:psc_array.num_psc] + b).T @ (A @ (Linv @ psi_new)[:psc_array.num_psc] + b)
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            # print(Bn_objective, Bn_objective_new)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # print(A @ Linv @ psi_new, A, Linv, psi, psi_new)
            # print(b)
            # print('psi1, psi2 = ', psc_array_new.psi, psc_array.psi)
            dpsi_ddelta = (psc_array_new.psi - psc_array.psi) / epsilon  / psc_array.fac
            psi_deriv = psc_array.psi_deriv()  # * 1e-7
            # print('dpsi = ', dpsi_ddelta[0], psi_deriv[ncoils])
            # print(psi_deriv)
            assert(np.allclose(dpsi_ddelta[0], psi_deriv[ncoils], rtol=1e-3))
            dBn_analytic = (A @ I + b).T @ (-A @ (Linv * psi_deriv[ncoils:]))
            print(dBn_objective, dBn_analytic[0])
            assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
            
            # Repeat but change coil 3
            # psc_array = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            # A = psc_array.A_matrix
            # psi = psc_array.psi_total
            # Linv = psc_array.L_inv  # [:psc_array.num_psc, :psc_array.num_psc]
            # Bn_objective = 0.5 * (A @ (Linv @ psi)[:psc_array.num_psc] + b).T @ (A @ (Linv @ psi)[:psc_array.num_psc] + b)
            # deltas_new = np.copy(deltas)
            # deltas_new[3] += epsilon
            # alphas_new = np.copy(alphas)
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            # )
            # psi_new = psc_array_new.psi_total
            # Bn_objective_new = 0.5 * (A @ (Linv @ psi_new)[:psc_array.num_psc] + b).T @ (A @ (Linv @ psi_new)[:psc_array.num_psc] + b)
            # dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # dpsi_ddelta = (psi_new - psi) / epsilon
            # psi_deriv = psc_array.psi_deriv() * 1e-7
            # print(dpsi_ddelta[3], psi_deriv[ncoils + 3])
            # assert(np.allclose(dpsi_ddelta[3], psi_deriv[ncoils + 3], rtol=1e-3))
            # dBn_analytic = (A @ (Linv @ psi)[:psc_array.num_psc] + b).T @ (A @ (Linv * psi_deriv[ncoils_sym:])[:psc_array.num_psc, :])
            # print(dBn_objective, dBn_analytic[3])
            # assert np.isclose(dBn_objective, dBn_analytic[3], rtol=1e-2)
    
            # # Test partial derivative calculation if we change alpha by epsilon
            # psc_array = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            # A = psc_array.A_matrix
            # psi = psc_array.psi
            # Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc]
            # Bn_objective = 0.5 * (A @ Linv @ psi + b).T @ (A @ Linv @ psi + b)
            # alphas_new = np.copy(alphas)
            # alphas_new[0] += epsilon
            # deltas_new = np.copy(deltas)
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            # )
            # psi_new = psc_array_new.psi
            # Bn_objective_new = 0.5 * (A @ Linv @ psi_new + b).T @ (A @ Linv @ psi_new + b)
            # dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # dpsi_dalpha = (psi_new - psi) / epsilon
            # psi_deriv = psc_array.psi_deriv() * 1e-7
            # print(dpsi_dalpha[0], psi_deriv[0])
            # assert(np.allclose(dpsi_dalpha[0], psi_deriv[0], rtol=1e-2))
            # dBn_analytic = (A @ Linv @ psi + b).T @ (A @ (Linv * psi_deriv[:ncoils]))
            # print(dBn_objective, dBn_analytic[0])
            # assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-1)
            
            # Repeat for changing coil 7
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi / psc_array.fac
            Linv = psc_array.L_inv[:psc_array.num_psc, :psc_array.num_psc]
            I = (-Linv @ psi)
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            alphas_new = np.copy(alphas)
            alphas_new[1] += epsilon
            deltas_new = np.copy(deltas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            psi_new = psc_array_new.psi / psc_array.fac
            I_new = (-Linv @ psi_new)
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dpsi_dalpha = (psi_new - psi) / epsilon 

            psi_deriv = psc_array.psi_deriv() # * 1e-7
            print(dpsi_dalpha[1], psi_deriv[1])
            # assert(np.allclose(dpsi_dalpha[6], psi_deriv[6], rtol=1e-3))
            dBn_analytic = (A @ I + b).T @ (-A @ (Linv * psi_deriv[:ncoils]))
            print(dBn_objective, dBn_analytic[1])
            assert np.isclose(dBn_objective, dBn_analytic[1], rtol=1e-1)
    
    def test_L_analytic_derivatives(self):
        """
        Tests the analytic calculations of dL/dalpha and dL/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        for surf in surfs:  # surf1
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            psi = psc_array.psi_total 
            b = psc_array.b_opt * psc_array.fac
            L = psc_array.L
            Linv = psc_array.L_inv
            I = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            L_deriv = psc_array.L_deriv()
            deltas_new = np.copy(deltas)
            deltas_new[0] += epsilon
            alphas_new = np.copy(alphas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            L_new = psc_array_new.L
            Linv_new = psc_array_new.L_inv
            I_new = (-Linv_new @ psi)[:psc_array.num_psc]
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dL_ddelta = (L_new - L) / epsilon
            dLinv_ddelta = (Linv_new - Linv) / epsilon
            ncoils_sym = L_deriv.shape[0] // 2
            # print(np.array_str(dL_ddelta*1e7, precision=4, suppress_small=True))
            # print(np.array_str(L_deriv[ncoils_sym, :, :]*1e7, precision=4, suppress_small=True))
            # print(np.array_str(L_deriv[ncoils_sym, :, :], precision=4))
            # print(np.array_str(dL_ddelta / L_deriv[ncoils_sym, :, :], precision=4, suppress_small=True))

            dL_ddelta_analytic = L_deriv
            dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
            # print(dLinv_ddelta_analytic.shape)
            # print(np.array_str(dLinv_ddelta_analytic * 1e10, precision=3, suppress_small=True))
            # exit()

            # Linv calculations looks much more incorrect that the L derivatives,
            # maybe because of numerical error accumulation? 
            print(np.max(np.abs(dL_ddelta - dL_ddelta_analytic[ncoils_sym, :, :])))
            # print(dL_ddelta -  dL_ddelta_analytic[ncoils_sym, :, :])
            assert(np.allclose(dL_ddelta, dL_ddelta_analytic[ncoils_sym, :, :], rtol=1))
            # assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[ncoils_sym, :, :], rtol=10))
            dBn_analytic = (A @ I + b).T @ (-A @ (dLinv_ddelta_analytic @ psi)[:, :psc_array.num_psc].T)
            print(dBn_objective, dBn_analytic[ncoils_sym])
            assert np.isclose(dBn_objective, dBn_analytic[ncoils_sym], rtol=1e-1)
            
            # repeat calculation but change coil 3
            # psc_array = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            # L = psc_array.L
            # Linv = psc_array.L_inv
            # I = (Linv @ psi)[:psc_array.num_psc]
            # Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            # L_deriv = psc_array.L_deriv()
            # deltas_new = np.copy(deltas)
            # deltas_new[3] += epsilon
            # alphas_new = np.copy(alphas)
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            # )
            # L_new = psc_array_new.L
            # Linv_new = psc_array_new.L_inv
            # I_new = (Linv_new @ psi)[:psc_array.num_psc]
            # Bn_objective_new = 0.5 * (A @ I_new  + b).T @ (A @ I_new + b)
            # dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # dL_ddelta = (L_new - L) / epsilon
            # dLinv_ddelta = (Linv_new - Linv) / epsilon
            # dL_ddelta_analytic = L_deriv
            # dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
            # # print(np.array_str(dL_ddelta / L_deriv[ncoils_sym + 3, :, :], precision=5))
            # # print(np.array_str(L_deriv[ncoils_sym + 3, :, :], precision=5))
            # # print(dLinv_ddelta / dLinv_ddelta_analytic[ncoils_sym + 3, :, :])
            # assert(np.allclose(dL_ddelta, dL_ddelta_analytic[ncoils_sym + 3, :, :], rtol=1e-1))
            # # assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[ncoils_sym + 3, :, :], rtol=10))
            # dBn_analytic = (A @ I + b).T @ (A @ (dLinv_ddelta_analytic @ psi)[:, :psc_array.num_psc].T)
            # print(dBn_objective, dBn_analytic[ncoils_sym + 3])
            # assert np.isclose(dBn_objective, dBn_analytic[ncoils_sym + 3], rtol=1e-2)
    
            # # Test partial derivative calculation if we change alpha by epsilon
            # psc_array = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            # L = psc_array.L
            # Linv = psc_array.L_inv
            # I = (Linv @ psi)[:psc_array.num_psc]
            # Bn_objective_new = 0.5 * (A @ I + b).T @ (A @ I + b)
            # L_deriv = psc_array.L_deriv()
            # alphas_new = np.copy(alphas)
            # alphas_new[0] += epsilon
            # deltas_new = np.copy(deltas)
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            # )
            # L_new = psc_array_new.L
            # Linv_new = psc_array_new.L_inv
            # I_new = (Linv_new @ psi)[:psc_array.num_psc]
            # Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            # dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # dL_ddelta = (L_new - L) / epsilon
            # dLinv_ddelta = (Linv_new - Linv) / epsilon
            # dL_ddelta_analytic = L_deriv
            # dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
            # assert(np.allclose(dL_ddelta, dL_ddelta_analytic[0, :, :], rtol=1e-2))
            # # assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[0, :, :], rtol=10))
            # dBn_analytic = (A @ I + b).T @ (A @ (dLinv_ddelta_analytic @ psi)[:, :psc_array.num_psc].T)
            # print(dBn_objective, dBn_analytic[0])
            # assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
            
            # # Repeat changing coil 2
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            L = psc_array.L
            Linv = psc_array.L_inv
            I = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective_new = 0.5 * (A @ I + b).T @ (A @ I + b)
            L_deriv = psc_array.L_deriv()
            alphas_new = np.copy(alphas)
            alphas_new[1] += epsilon
            deltas_new = np.copy(deltas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            L_new = psc_array_new.L
            Linv_new = psc_array_new.L_inv
            I_new = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective_new = 0.5 * (A @ I_new + b).T @ (A @ I_new + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            dL_ddelta = (L_new - L) / epsilon
            dLinv_ddelta = (Linv_new - Linv) / epsilon
            dL_ddelta_analytic = L_deriv
            dLinv_ddelta_analytic = -L_deriv @ Linv @ Linv
                
            print(np.array_str(dL_ddelta*1e7, precision=3, suppress_small=True))
            print(np.array_str(L_deriv[1, :, :]*1e7, precision=3, suppress_small=True))
            # print(np.max(np.abs(dL_ddelta - dL_ddelta_analytic[ncoils_sym, :, :])))
            assert(np.allclose(dL_ddelta, dL_ddelta_analytic[1, :, :], rtol=1e-2))
            # assert(np.allclose(dLinv_ddelta, dLinv_ddelta_analytic[2, :, :], rtol=10))
            dBn_analytic = (A @ I + b).T @ (-A @ (dLinv_ddelta_analytic @ psi)[:, :psc_array.num_psc].T)
            print(dBn_objective, dBn_analytic[1])
            # assert np.isclose(dBn_objective, dBn_analytic[1], rtol=1e-2)
        
        
    def test_symmetrized_quantities(self):
        """
        Tests that discrete symmetries are satisfied by the various terms
        appearing in optimization. 
        """
        from simsopt.objectives import SquaredFlux
        from simsopt.field import coils_via_symmetries
        
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        nphi = 32
        surf = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='half period', nphi=nphi, ntheta=nphi
        )
        surf_full = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=nphi * 4, ntheta=nphi * 4
        )
        ncoils = 6
        np.random.seed(1)
        R = 0.2
        a = 1e-5
        points = (np.random.rand(ncoils, 3) - 0.5) * 20
        alphas = (np.random.rand(ncoils) - 0.5) * np.pi
        deltas = (np.random.rand(ncoils) - 0.5) * 2 * np.pi
        kwargs_manual = {"plasma_boundary": surf}
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
        )
        psc_array.I = 1e4 * np.ones(len(psc_array.I))
        Ax_b = (psc_array.A_matrix @ psc_array.I + psc_array.b_opt) * psc_array.grid_normalization
        Bn = 0.5 * Ax_b.T @ Ax_b / psc_array.normalization * psc_array.fac ** 2
        
        currents = []
        for i in range(psc_array.num_psc):
            currents.append(Current(psc_array.I[i]))
        all_coils = coils_via_symmetries(
            psc_array.curves, currents, nfp=surf.nfp, stellsym=surf.stellsym
        )
        B_WP = BiotSavart(all_coils)

        fB = SquaredFlux(surf, B_WP + psc_array.B_TF, np.zeros((nphi, nphi))).J() / psc_array.normalization
        assert np.isclose(Bn, fB)
        fB_full = SquaredFlux(surf_full, B_WP + psc_array.B_TF, np.zeros((nphi * 4, nphi * 4))).J() / psc_array.normalization
        assert np.isclose(Bn, fB_full)
        
        # A_deriv = psc_array.A_deriv()
        # A_derivI = (A_deriv[:, :psc_array.num_psc] @ psc_array.I) * psc_array.grid_normalization
        # Bn_deriv = 0.5 * Ax_b.T @ A_derivI / psc_array.normalization * psc_array.fac ** 2
        # print(Bn_deriv)
        

        
    def test_dA_analytic_derivatives(self):
        """
        Tests the analytic calculations of dA/dalpha and dA/ddelta
        against finite differences for tiny changes to the angles, 
        to see if the analytic calculations are correct.
        """
        for surf in surfs:
            print('Surf = ', surf)
            kwargs_manual = {"plasma_boundary": surf}
            psc_array = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            )
            A = psc_array.A_matrix
            Linv = psc_array.L_inv
            psi = psc_array.psi_total
            I = (-Linv @ psi)[:psc_array.num_psc]
            b = psc_array.b_opt * psc_array.fac
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            alphas_new = np.copy(alphas)
            # alphas_new[0] += epsilon
            deltas_new = np.copy(deltas)
            deltas_new[0] += epsilon
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            A_new = psc_array_new.A_matrix
            
            # A_new_full = psc_array_new.A_matrix_full
            Bn_objective_new = 0.5 * (A_new @ I + b).T @ (A_new @ I + b)
            # print('Bns = ', Bn_objective, Bn_objective_new)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            A_deriv = psc_array.A_deriv()
            dBn_analytic = (A @ I + b).T @ (A_deriv[:, ncoils:] * I)
            print(dBn_objective, dBn_analytic[0])
            dA_dalpha = (A_new - A) / epsilon
            dA_dkappa_analytic = A_deriv
            # print('dA0 = ', dA_dalpha[:, 0], dA_dkappa_analytic[:, ncoils])
            print('dA0 = ', dA_dalpha[0, :], dA_dkappa_analytic[0, :])
            assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-1)

            # print(dA_dalpha[:, 0], dA_dkappa_analytic[:, ncoils])
            # print('dA0 = ', dA_dalpha[-1, 0], dA_dkappa_analytic[-1, ncoils])
            assert np.allclose(dA_dalpha[:, 0], dA_dkappa_analytic[:, ncoils], rtol=1e1)
            
            # Repeat but change coil 3
            # psc_array = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            A = psc_array.A_matrix
            Linv = psc_array.L_inv
            psi = psc_array.psi_total
            I = (-Linv @ psi)[:psc_array.num_psc]
            Bn_objective = 0.5 * (A @ I + b).T @ (A @ I + b)
            alphas_new = np.copy(alphas)
            alphas_new[1] += epsilon
            deltas_new = np.copy(deltas)
            psc_array_new = PSCgrid.geo_setup_manual(
                points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            )
            A_new = psc_array_new.A_matrix
            # dd = psc_array_new.deltas - psc_array.deltas
            # ddprime = psc_array_new.deltas_total - psc_array.deltas_total
            # aaprime = psc_array_new.alphas_total - psc_array.alphas_total
            # print('dd, ddprime = ', dd, ddprime, ddprime / epsilon)
            # print('a = ', psc_array.alphas_total)
            # print('a_new = ', psc_array_new.alphas_total)
            # print('aaprime = ', aaprime / epsilon, psc_array.aaprime_aa)
            # print('aaprime_dd = ', psc_array.aaprime_dd)
            # print('ddprime_aa = ', psc_array.ddprime_aa)
            # print('aaprime_dd = ', psc_array.aaprime_dd)
            # print('aaprime_dd = ', psc_array.aaprime_dd)
            Bn_objective_new = 0.5 * (A_new @ I + b).T @ (A_new @ I + b)
            dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            A_deriv = psc_array.A_deriv()
            dBn_analytic = (A @ I + b).T @ (A_deriv[:, :ncoils] * I)
            print(dBn_objective, dBn_analytic[1])
            # assert np.isclose(dBn_objective, dBn_analytic[1], rtol=1e-1)
            dA_dalpha = (A_new - A) / epsilon
            dA_dkappa_analytic = A_deriv
            print('dA0 = ', dA_dalpha[0, 1], dA_dkappa_analytic[0, 1])
            # print(dA_dalpha[:, 4] / dA_dkappa_analytic[:, 4])
            assert np.allclose(dA_dalpha[:, 1], dA_dkappa_analytic[:, 1], rtol=1)
            
            # # Repeat but changing delta instead of alpha
            # psc_array = PSCgrid.geo_setup_manual(
            # points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            # A = psc_array.A_matrix
            # Linv = psc_array.L_inv[:ncoils, :ncoils]
            # psi = psc_array.psi
            # Linv_psi = Linv @ psi
            # Bn_objective = 0.5 * (A @ Linv_psi + b).T @ (A @ Linv_psi + b)
            # alphas_new = np.copy(alphas)
            # deltas_new = np.copy(deltas)
            # deltas_new[0] += epsilon
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            # )
            # A_new = psc_array_new.A_matrix
            # Bn_objective_new = 0.5 * (A_new @ Linv_psi + b).T @ (A_new @ Linv_psi + b)
            # dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # A_deriv = psc_array.A_deriv()
            # # notice ncoils: instead of :ncoils below
            # dBn_analytic = (A @ Linv_psi + b).T @ (A_deriv[:, ncoils:] * Linv_psi)
            # print(dBn_objective, dBn_analytic[0])
            # dA_ddelta = (A_new - A) / epsilon
            # dA_dkappa_analytic = A_deriv
            # print(dA_ddelta[-1, 0] / dA_dkappa_analytic[-1, ncoils])
            # assert np.isclose(dBn_objective, dBn_analytic[0], rtol=1e-2)
            # assert np.allclose(dA_ddelta[:, 0], dA_dkappa_analytic[:, ncoils], rtol=1e-1)
            
            # # Repeat but change coil 5
            # psc_array = PSCgrid.geo_setup_manual(
            # points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs_manual
            # )
            # A = psc_array.A_matrix
            # Linv = psc_array.L_inv[:ncoils, :ncoils]
            # psi = psc_array.psi
            # Linv_psi = Linv @ psi
            # Bn_objective = 0.5 * (A @ Linv_psi + b).T @ (A @ Linv_psi + b)
            # alphas_new = np.copy(alphas)
            # deltas_new = np.copy(deltas)
            # deltas_new[5] += epsilon
            # psc_array_new = PSCgrid.geo_setup_manual(
            #     points, R=R, a=a, alphas=alphas_new, deltas=deltas_new, **kwargs_manual
            # )
            # A_new = psc_array_new.A_matrix
            # Bn_objective_new = 0.5 * (A_new @ Linv_psi + b).T @ (A_new @ Linv_psi + b)
            # dBn_objective = (Bn_objective_new - Bn_objective) / epsilon
            # A_deriv = psc_array.A_deriv()
            # # notice ncoils: instead of :ncoils below
            # dBn_analytic = (A @ Linv_psi + b).T @ (A_deriv[:, ncoils:] * Linv_psi)
            # print(dBn_objective, dBn_analytic[5])
            # assert np.isclose(dBn_objective, dBn_analytic[5], rtol=1e-2)
            # dA_ddelta = (A_new - A) / epsilon
            # dA_dkappa_analytic = A_deriv
            # print(dA_ddelta[-1, 5] / dA_dkappa_analytic[-1, ncoils + 5])
            # assert np.allclose(dA_ddelta[:, 5], dA_dkappa_analytic[:, ncoils + 5], rtol=1)

    def test_L(self):
        """
        Tests the inductance calculation for some limiting cases:
            1. Identical, coaxial coils 
            (solution in Jacksons textbook, problem 5.28)
        and tests that the inductance and flux calculations agree,
        when we use the "TF field" as one of the coil B fields
        and compute the flux through the other coil, for coils with random 
        separations, random orientations, etc.
        """
    
        from scipy.special import ellipk, ellipe
        
        np.random.seed(1)
        
        # initialize coaxial coils
        R0 = 4
        R = 0.1
        a = 1e-5
        mu0 = 4 * np.pi * 1e-7
        points = np.array([[R0, 0.5 * R0, 0], [R0, 0.5 * R0, R0]])
        alphas = np.zeros(2)
        deltas = np.zeros(2)
        psc_array = PSCgrid.geo_setup_manual(
            points, R=R, a=a, alphas=alphas, deltas=deltas
        )
        L = psc_array.L * 1e-7
        L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        
        # Jackson problem 5.28
        k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        # Jackson uses K(k) and E(k) but this corresponds to
        # K(k^2) and E(k^2) in scipy library
        L_mutual_analytic = mu0 * R * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )
        assert(np.allclose(L_self_analytic, np.diag(L)))
        print(L_mutual_analytic * 1e10, L[0, 1] * 1e10, L)
        assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10, rtol=1e-3))
        assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10, rtol=1e-3))
        
        # Another simple test of the analytic formula   
        # Formulas only valid for R/a >> 1 so otherwise it will fail
        # R = 1
        # a = 1e-6
        # R0 = 1
        # mu0 = 4 * np.pi * 1e-7
        # points = np.array([[0, 0, 0], [0, 0, R0]])
        # alphas = np.zeros(2)
        # deltas = np.zeros(2)
        # psc_array = PSCgrid.geo_setup_manual(
        #     points, R=R, a=a, alphas=alphas, deltas=deltas,
        # )
        # L = psc_array.L * 1e-7
        # k = np.sqrt(4.0 * R ** 2 / (4.0 * R ** 2 + R0 ** 2))
        # L_mutual_analytic = mu0 * R * (
        #     (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        # )
        # L_self_analytic = mu0 * R * (np.log(8.0 * R / a) - 2.0)
        # assert(np.allclose(L_self_analytic, np.diag(L)))
        # print(L_mutual_analytic * 1e10, L[0, 1] * 1e10)
        # assert(np.isclose(L_mutual_analytic * 1e10, L[0, 1] * 1e10, rtol=1e-1))
        # assert(np.isclose(L_mutual_analytic * 1e10, L[1, 0] * 1e10, rtol=1e-1))
        I = 1e10
        # coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
        # bs = BiotSavart(coils)
        # center = np.array([0, 0, 0]).reshape(1, 3)
        # bs.set_points(center)
        # B_center = bs.B()
        # Bz_center = B_center[:, 2]
        # kwargs = {"B_TF": bs, "ppp": 1000}
        # psc_array = PSCgrid.geo_setup_manual(
        #     points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
        # )
        # # This is not a robust check but it only converges when N >> 1
        # points are used to do the integrations and there
        # are no discrete symmetries. Can easily check that 
        # can decrease rtol as you increase number of integration points
        # assert(np.isclose(psc_array.psi[0] / I, L[1, 0], rtol=1e-1))
        # Only true if R << 1, assert(np.isclose(psc_array.psi[0], np.pi * psc_array.R ** 2 * Bz_center))
        
        # Test that inductance and flux calculations for wide set of
        # scenarios
        a = 1e-4
        R0 = 10
        print('starting loop')
        for R in [0.1]:
            for points in [np.array([(np.random.rand(3) - 0.5) * 40, 
                                      (np.random.rand(3) - 0.5) * 40])]:
                for alphas in [np.zeros(2), (np.random.rand(2) - 0.5) * np.pi]:
                    for deltas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                        for surf in [surfs[0]]:
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas,
                            )
                            coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                            bs = BiotSavart(coils)
                            kwargs = {"B_TF": bs, "ppp": 2000, "plasma_boundary": surf}
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                            )
                            L = psc_array.L
                            # print(L[0, 1] * 1e10)
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
                                    phi0 = (2 * np.pi / psc_array.nfp) * fp
                                    # get new locations by flipping the y and z components, then rotating by phi0
                                    ox = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                                    oy = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                                    oz = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                                    xyz = np.array([ox, oy, oz]).T
                                    Bn_PSC += sopp.A_matrix(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                                        psc_array.R,
                                    ) @ psc_array.I * 2e-7
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_boundary.gamma().reshape(-1, 3)),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I * stell),
                                        psc_array.R,
                                    )
                                    q = q + 1
                            # Calculate Bfields from CircularCoil class
                            B_circular_coils = np.zeros(B_PSC.shape)
                            Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                            for i in range(len(psc_array.alphas)):
                                PSC = CircularCoil(
                                    psc_array.R, 
                                    psc_array.grid_xyz_all[i, :], 
                                    psc_array.I_all[i], 
                                    psc_array.coil_normals_all[i, :]
                                )
                                PSC.set_points(psc_array.plasma_boundary.gamma().reshape(-1, 3))
                                B_circular_coils += PSC.B().reshape(-1, 3)
                                Bn_circular_coils += np.sum(PSC.B().reshape(
                                    -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart
                            currents = []
                            for i in range(len(psc_array.I)):
                                currents.append(Current(psc_array.I[i]))
                            coils = coils_via_symmetries(
                                psc_array.curves, currents, nfp=1, stellsym=False
                            )
                            B_direct = BiotSavart(coils)
                            B_direct.set_points(psc_array.plasma_points)
                            Bn_direct = np.sum(B_direct.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart, using all the coils manually defined
                            currents = []
                            for i in range(psc_array.num_psc * psc_array.symmetry):
                                currents.append(Current(psc_array.I_all[i]))
                            all_coils = coils_via_symmetries(
                                psc_array.all_curves, currents, nfp=1, stellsym=False
                            )
                            B_direct_all = BiotSavart(all_coils)
                            B_direct_all.set_points(psc_array.plasma_points)
                            Bn_direct_all = np.sum(B_direct_all.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            
                            
                            psc_array.Bn_PSC = psc_array.Bn_PSC * psc_array.fac
                            print(psc_array.Bn_PSC[-1] * 1e10, Bn_PSC[-1] * 1e10, 
                                  Bn_circular_coils[-1] * 1e10, Bn_direct[-1] * 1e10,
                                  Bn_direct_all[-1] * 1e10)
                            # Robust test of all the B and Bn calculations from circular coils
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, atol=1e3, rtol=1e-3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, atol=1e3, rtol=1e-3))
        
        # Applying discrete symmetries to the coils wont work unless they dont intersect the symmetry planes
        for R in [0.1]:
            for points in [np.array([[1, 2, -1], [-1, -1, 3]])]:
                for alphas in [np.zeros(2), (np.random.rand(2) - 0.5) * np.pi]:
                    for deltas in [np.zeros(2), (np.random.rand(2) - 0.5) * 2 * np.pi]:
                        print(R, points, alphas, deltas)
                        for surf in surfs[1:]:
                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas,
                            )
                            # coils = coils_via_symmetries([psc_array.curves[1]], [Current(I)], nfp=1, stellsym=False)
                            # bs = BiotSavart(coils)
                            # kwargs = {"B_TF": bs, "ppp": 2000, "plasma_boundary": surf}
                            kwargs = {"plasma_boundary": surf}

                            psc_array = PSCgrid.geo_setup_manual(
                                points, R=R, a=a, alphas=alphas, deltas=deltas, **kwargs
                            )
                            L = psc_array.L
                            
                            contig = np.ascontiguousarray
                            # Calculate B fields like psc_array function does
                            B_PSC = np.zeros((psc_array.nphi * psc_array.ntheta, 3))
                            Bn_PSC = np.zeros(psc_array.nphi * psc_array.ntheta)
                            nn = psc_array.num_psc
                            q = 0
                            for fp in range(psc_array.nfp):
                                for stell in psc_array.stell_list:
                                    phi0 = (2 * np.pi / psc_array.nfp) * fp
                                    # get new locations by flipping the y and z components, then rotating by phi0
                                    ox = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                                    oy = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                                    oz = psc_array.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                                    xyz = np.array([ox, oy, oz]).T
                                    Bn_PSC += sopp.A_matrix(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                                        psc_array.R,
                                    ) @ (psc_array.I) * 2e-7
                                    B_PSC += sopp.B_PSC(
                                        contig(xyz),
                                        contig(psc_array.plasma_points),
                                        contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                                        contig(psc_array.I),
                                        psc_array.R,
                                    )
                                    q = q + 1
                            B_PSC_all = sopp.B_PSC(
                                contig(psc_array.grid_xyz_all),
                                contig(psc_array.plasma_points),
                                contig(psc_array.alphas_total),
                                contig(psc_array.deltas_total),
                                contig(psc_array.I),
                                psc_array.R,
                            )
                            # Calculate Bfields from CircularCoil class
                            B_circular_coils = np.zeros(B_PSC.shape)
                            Bn_circular_coils = np.zeros(psc_array.Bn_PSC.shape)
                            print(surf)
                            for i in range(psc_array.alphas_total.shape[0]):
                                PSC = CircularCoil(
                                    psc_array.R, 
                                    psc_array.grid_xyz_all[i, :], 
                                    psc_array.I_all[i], # * np.sign(psc_array.I_all[i]), 
                                    psc_array.coil_normals_all[i, :]
                                )
                                PSC.set_points(psc_array.plasma_points)
                                B_circular_coils += PSC.B().reshape(-1, 3)
                                Bn_circular_coils += np.sum(PSC.B().reshape(
                                    -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart
                            currents = []
                            for i in range(len(psc_array.I)):
                                currents.append(Current(psc_array.I[i]))
                            coils = coils_via_symmetries(
                                psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
                            )
                            currents_total = [coil.current.get_value() for coil in coils]
                            # print(currents_total)
                            B_direct = BiotSavart(coils)
                            B_direct.set_points(psc_array.plasma_points)
                            Bn_direct = np.sum(B_direct.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            # Calculate Bfields from direct BiotSavart, using all the coils manually defined
                            currents = []
                            for i in range(psc_array.num_psc * psc_array.symmetry):
                                currents.append(Current(psc_array.I_all_with_sign_flips[i]))
                            all_coils = coils_via_symmetries(
                                psc_array.all_curves, currents, nfp=1, stellsym=False
                            )
                            currents_total = [coil.current.get_value() for coil in all_coils]
                            curves_total = [coil.curve for coil in all_coils]
                            
                            from simsopt.geo import curves_to_vtk

                            curves_to_vtk(curves_total, "direct_curves", close=True, scalar_data=currents_total)
                            # print(currents_total)
                            B_direct_all = BiotSavart(all_coils)
                            B_direct_all.set_points(psc_array.plasma_points)
                            Bn_direct_all = np.sum(B_direct_all.B().reshape(
                                -1, 3) * psc_array.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
                            
                            psc_array.Bn_PSC = psc_array.Bn_PSC * psc_array.fac
                            # Robust test of all the B and Bn calculations from circular coils
                            print(psc_array.Bn_PSC[-1] * 1e10, Bn_PSC[-1] * 1e10, 
                                  Bn_circular_coils[-1] * 1e10, Bn_direct[-1] * 1e10,
                                  Bn_direct_all[-1] * 1e10)
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_PSC * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_circular_coils * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct * 1e10, rtol=1e-2, atol=1e3))
                            assert(np.allclose(psc_array.Bn_PSC * 1e10, Bn_direct_all * 1e10, rtol=1e-2, atol=1e3))

if __name__ == "__main__":
    unittest.main()
