import unittest
import numpy as np
from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.biotsavart import BiotSavart
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.geo.surfaceobjectives import Area
from simsopt.geo.surfaceobjectives import boozer_surface_residual
from simsopt.geo.surfaceobjectives import boozer_surface_dexactresidual_dcoils_vjp
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from .surface_test_helpers import get_ncsx_data, get_surface, get_exact_surface
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()



surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


class BoozerSurfaceTests(unittest.TestCase):
    def test_compare_with_booz_xform(self):
        """
        Load three surfaces precomputed with booz_xform and compare with the surfaces computed
        here.
        """
        coils, currents, ma = get_ncsx_data()
        stellarator = CoilCollection(coils, currents, 3, True)
        mpol = 10  # try increasing this to 8 or 10 for smoother surfaces
        ntor = 10  # try increasing this to 8 or 10 for smoother surfaces
        stellsym = True
        nfp = 3
        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.10, flip_theta=True)

        bs = BiotSavart(stellarator.coils, stellarator.currents)
        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
        tf = ToroidalFlux(s, bs_tf, stellarator)
        
        iota0 = -0.4
        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        
        bxf_iota = [0.407090160846074, 0.414192217748192, 0.41946449378259]
        bxf_G = [2.20518990460331, 2.20501527027576, 2.20502566123929]
        bxf_tf = [0.05, 0.10, 0.15]
        
        from scipy.io import netcdf
        f1 = netcdf.NetCDFFile(TEST_DIR / 'NCSX_test_data' / 'boozmn_flux0.05_mpol9_ntor9_ns201.nc', 'r')
        f2 = netcdf.NetCDFFile(TEST_DIR / 'NCSX_test_data' / 'boozmn_flux0.10_mpol9_ntor9_ns201.nc', 'r')
        f3 = netcdf.NetCDFFile(TEST_DIR / 'NCSX_test_data' / 'boozmn_flux0.15_mpol9_ntor9_ns201.nc', 'r')
        
        files = [f1, f2, f3]
        for idx, (tf_target, f) in enumerate(zip(bxf_tf, files)):
            boozer_surface = BoozerSurface(bs, s, tf, tf_target)
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=300,iota=iota0,G=G0)
            print(res["success"], np.abs(bxf_iota[idx]-(-res["iota"]))/bxf_iota[idx], np.abs(bxf_G[idx]-res["G"]/(2.*np.pi))/bxf_G[idx])
            
            in_bmn = f.variables['bmnc_b']
            in_rmn = f.variables['rmnc_b']
            in_zmn = f.variables['zmns_b']
            in_ixm = f.variables['ixm_b']
            in_ixn = f.variables['ixn_b']
            
            bmn = np.zeros(in_bmn.shape)
            rmn = np.zeros(in_rmn.shape)
            zmn = np.zeros(in_zmn.shape)
            ixm = np.zeros(in_ixm.shape)
            ixn = np.zeros(in_ixn.shape)
            
            bmn[:,:] = in_bmn[:,:]
            rmn[:,:] = in_rmn[:,:]
            zmn[:,:] = in_zmn[:,:]
            ixm[:] = in_ixm[:]
            ixn[:] = in_ixn[:]
            
            bmn = bmn.reshape( (-1,) )
            rmn = rmn.reshape( (-1,) )
            zmn = zmn.reshape( (-1,) )
            
            phi_grid, theta_grid = np.meshgrid(phis, thetas)
            phi_grid = phi_grid.T * 2 * np.pi
            theta_grid = theta_grid.T * 2 * np.pi
            bxf_B = np.sum(bmn[:,None,None] * np.cos( -ixm[:, None, None] * theta_grid[None, ...] - ixn[:, None, None] * phi_grid[None, ...]), axis=0)
            bxf_R = np.sum(rmn[:,None,None] * np.cos( -ixm[:, None, None] * theta_grid[None, ...] - ixn[:, None, None] * phi_grid[None, ...]), axis=0)
            bxf_Z = np.sum(zmn[:,None,None] * np.sin( -ixm[:, None, None] * theta_grid[None, ...] - ixn[:, None, None] * phi_grid[None, ...]), axis=0)
            
            booz_xyz = s.gamma()
            bs.set_points(booz_xyz.reshape( (-1,3) ) )
            B = bs.B()

            booz_R = np.sqrt(booz_xyz[:,:,0]**2 + booz_xyz[:,:,1]**2)
            booz_Z = booz_xyz[:,:,2]
            booz_B = np.sqrt(B[...,0]**2 + B[...,1]**2 + B[...,2]**2).reshape( (booz_xyz.shape[0], booz_xyz.shape[1]) )
            print( np.max(np.abs(booz_R - bxf_R)/np.abs(bxf_R)), np.max(np.abs(booz_Z - bxf_Z) ), np.max(np.abs(booz_B - bxf_B)/bxf_B) )

        for f in files:
            f.close()
#            import mayavi.mlab as mlab
#            mlab.mesh(bxf_R * np.cos(phi_grid), bxf_R * np.sin(phi_grid), bxf_Z)
#            mlab.mesh(booz_R * np.cos(phi_grid), booz_R * np.sin(phi_grid), booz_Z)
#            mlab.show()
#            import ipdb;ipdb.set_trace()





#     def test_residual(self):
#        """
#        This test loads a SurfaceXYZFourier that interpolates the xyz coordinates
#        of a surface in the NCSX configuration that was computed on a previous
#        branch of pyplasmaopt.  Here, we verify that the  Boozer residual at these 
#        interpolation points is small.
#        """
#
#        s = get_exact_surface()
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#
#        weight = 1.
#        tf = ToroidalFlux(s, bs_tf)
#
#        # these data are obtained from `boozer` branch of pyplamsaopt
#        tf_target = 0.41431152
#        iota = -0.44856192
#
#        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
#        x = np.concatenate((s.get_dofs(), [iota]))
#        r0 = boozer_surface.boozer_penalty_constraints(
#            x, derivatives=0, constraint_weight=weight, optimize_G=False, scalarize=False)
#        # the residual should be close to zero for all entries apart from the y
#        # and z coordinate at phi=0 and theta=0 (and the corresponding rotations)
#        ignores_idxs = np.zeros_like(r0)
#        ignores_idxs[[1, 2, 693, 694, 695, 1386, 1387, 1388, -2, -1]] = 1
#        assert np.max(np.abs(r0[ignores_idxs < 0.5])) < 1e-8
#        assert np.max(np.abs(r0[-2:])) < 1e-6
#
#
#    def test_dresidual_dcoils_vjp(self):
#        """
#        This test verifies that the dresidual_dcoils_vjp calculation is correct.
#        """
#
#        def get_exact_residual(surface, label, target_label, biotsavart,iota=0.,G=None):
#            if not isinstance(s, SurfaceXYZTensorFourier):
#                raise RuntimeError('Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier')
#        
#            m = surface.get_stellsym_mask()
#            mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
#            if surface.stellsym:
#                mask[0, 0, 0] = False
#            mask = mask.flatten()
#        
#            boozer = boozer_surface_residual(surface, iota, G, bs, derivatives=0)
#            r = boozer[0]
#            return r[mask]
#        
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#
#        mpol = 8  # try increasing this to 8 or 10 for smoother surfaces
#        ntor = 8  # try increasing this to 8 or 10 for smoother surfaces
#        stellsym = True
#        nfp = 3
#        
#        phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
#        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
#        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
#        s.fit_to_curve(ma, 0.10, flip_theta=True)
#        iota = -0.4
#
#
#
#
#        G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
#    
#        label = Area(s)
#        target_label = 0.025
#
#        coeffs = stellarator.get_dofs()
#        def f(dofs):
#            stellarator.set_dofs(dofs)
#            bs.clear_cached_properties()
#            return get_exact_residual(s, label, target_label, bs,iota,G0)
#        f0 = f(coeffs)
#
#
#
#
#
#        m = s.get_stellsym_mask()
#        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
#        if s.stellsym:
#            mask[0, 0, 0] = False
#        mask = mask.flatten()
#        lm1 = np.random.uniform(size=mask.size)-0.5
#        lm1[np.logical_not(mask)] = 0.
#
#        lm1_dg_dcoils = boozer_surface_dresidual_dcoils_vjp(lm1, s, iota, G0, bs)
#        lm1_dg_dcoils = stellarator.reduce_coefficient_derivatives(lm1_dg_dcoils)
#        
#        lm2 = np.random.uniform(size=lm1_dg_dcoils.size)-0.5
#        fd_exact = np.dot(lm1_dg_dcoils , lm2)
#        
#        err_old = 1e9
#        epsilons = np.power(2., -np.asarray(range(7, 20)))
#        print("################################################################################")
#        for eps in epsilons:
#            f1 = f(coeffs + eps * lm2)
#            Jfd = (f1-f0)/eps
#            err = np.linalg.norm(np.dot(Jfd, lm1[mask])-fd_exact)/np.linalg.norm(fd_exact)
#            print(err/err_old)
#            assert err < err_old * 0.55
#            err_old = err
#        print("################################################################################")
#
#
#   
#
#    def test_boozer_penalty_constraints_gradient(self):
#        """
#        Taylor test to verify the gradient of the scalarized constrained optimization problem's
#        objective.
#        """
#        for surfacetype in surfacetypes_list:
#            for stellsym in stellsym_list:
#                for optimize_G in [True, False]:
#                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G):
#                        self.subtest_boozer_penalty_constraints_gradient(surfacetype, stellsym, optimize_G)
#
#    def test_boozer_penalty_constraints_hessian(self):
#        """
#        Taylor test to verify the Hessian of the scalarized constrained optimization problem's
#        objective.
#        """
#        for surfacetype in surfacetypes_list:
#            for stellsym in stellsym_list:
#                for optimize_G in [True, False]:
#                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G):
#                        self.subtest_boozer_penalty_constraints_hessian(surfacetype, stellsym, optimize_G)
#
#    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym, optimize_G=False):
#        np.random.seed(1)
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#
#        s = get_surface(surfacetype, stellsym)
#        s.fit_to_curve(ma, 0.1)
#
#        weight = 11.1232
#
#        tf = ToroidalFlux(s, bs_tf)
#
#        tf_target = 0.1
#        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
#
#        iota = -0.3
#        x = np.concatenate((s.get_dofs(), [iota]))
#        if optimize_G:
#            x = np.concatenate((x, [2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))]))
#        f0, J0 = boozer_surface.boozer_penalty_constraints(
#            x, derivatives=1, constraint_weight=weight, optimize_G=optimize_G)
#
#        h = np.random.uniform(size=x.shape)-0.5
#        Jex = J0@h
#
#        err_old = 1e9
#        epsilons = np.power(2., -np.asarray(range(7, 20)))
#        print("################################################################################")
#        for eps in epsilons:
#            f1 = boozer_surface.boozer_penalty_constraints(
#                x + eps*h, derivatives=0, constraint_weight=weight, optimize_G=optimize_G)
#            Jfd = (f1-f0)/eps
#            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
#            print(err/err_old, f0, f1)
#            assert err < err_old * 0.55
#            err_old = err
#        print("################################################################################")
#
#    def subtest_boozer_penalty_constraints_hessian(self, surfacetype, stellsym, optimize_G=False):
#        np.random.seed(1)
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#
#        s = get_surface(surfacetype, stellsym)
#        s.fit_to_curve(ma, 0.1)
#
#        tf = ToroidalFlux(s, bs_tf)
#
#        tf_target = 0.1
#        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
#
#        iota = -0.3
#        x = np.concatenate((s.get_dofs(), [iota]))
#        if optimize_G:
#            x = np.concatenate((x, [2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))]))
#        f0, J0, H0 = boozer_surface.boozer_penalty_constraints(x, derivatives=2, optimize_G=optimize_G)
#
#        h1 = np.random.uniform(size=x.shape)-0.5
#        h2 = np.random.uniform(size=x.shape)-0.5
#        d2f = h1@H0@h2
#
#        err_old = 1e9
#        epsilons = np.power(2., -np.asarray(range(10, 20)))
#        print("################################################################################")
#        for eps in epsilons:
#            fp, Jp = boozer_surface.boozer_penalty_constraints(x + eps*h1, derivatives=1, optimize_G=optimize_G)
#            d2f_fd = (Jp@h2-J0@h2)/eps
#            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
#            print(err/err_old)
#            assert err < err_old * 0.55
#            err_old = err
#
#    def test_boozer_constrained_jacobian(self):
#        """
#        Taylor test to verify the Jacobian of the first order optimality conditions of the exactly
#        constrained optimization problem.
#        """
#        for surfacetype in surfacetypes_list:
#            for stellsym in stellsym_list:
#                for optimize_G in [True, False]:
#                    with self.subTest(surfacetype=surfacetype, stellsym=stellsym, optimize_G=optimize_G):
#                        self.subtest_boozer_constrained_jacobian(surfacetype, stellsym, optimize_G)
#
#    def subtest_boozer_constrained_jacobian(self, surfacetype, stellsym, optimize_G=False):
#        np.random.seed(1)
#        coils, currents, ma = get_ncsx_data()
#        stellarator = CoilCollection(coils, currents, 3, True)
#
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#        bs_tf = BiotSavart(stellarator.coils, stellarator.currents)
#
#        s = get_surface(surfacetype, stellsym)
#        s.fit_to_curve(ma, 0.1)
#
#        tf = ToroidalFlux(s, bs_tf)
#
#        tf_target = 0.1
#        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
#
#        iota = -0.3
#        lm = [0., 0.]
#        x = np.concatenate((s.get_dofs(), [iota]))
#        if optimize_G:
#            x = np.concatenate((x, [2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))]))
#        xl = np.concatenate((x, lm))
#        res0, dres0 = boozer_surface.boozer_exact_constraints(xl, derivatives=1, optimize_G=optimize_G)
#
#        h = np.random.uniform(size=xl.shape)-0.5
#        dres_exact = dres0@h
#
#
#        err_old = 1e9
#        epsilons = np.power(2., -np.asarray(range(7, 20)))
#        print("################################################################################")
#        for eps in epsilons:
#            res1 = boozer_surface.boozer_exact_constraints(xl + eps*h, derivatives=0, optimize_G=optimize_G)
#            dres_fd = (res1-res0)/eps
#            err = np.linalg.norm(dres_fd-dres_exact)
#            print(err/err_old)
#            assert err < err_old * 0.55
#            err_old = err
#        print("################################################################################")
#
#    def test_boozer_surface_optimisation_convergence(self):
#        """
#        Test to verify the various optimization algorithms that compute
#        the Boozer angles on a surface.
#        """
#
#        configs = [
#            ("SurfaceXYZTensorFourier", True,  True,  'residual_exact'),
#            ("SurfaceXYZTensorFourier", True,  True,  'newton_exact'),
#            ("SurfaceXYZTensorFourier", True,  True,  'newton'),
#            ("SurfaceXYZTensorFourier", False, True,  'ls'),
#            ("SurfaceXYZFourier",       True,  False, 'ls'),
#        ]
#        for surfacetype, stellsym, optimize_G, second_stage in configs:
#            with self.subTest(
#                surfacetype=surfacetype, stellsym=stellsym,
#                    optimize_G=optimize_G, second_stage=second_stage):
#                self.subtest_boozer_surface_optimisation_convergence(surfacetype, stellsym, optimize_G, second_stage)
#
#    def subtest_boozer_surface_optimisation_convergence(self, surfacetype, stellsym, optimize_G, second_stage):
#        coils, currents, ma = get_ncsx_data()
#
#        if stellsym:
#            stellarator = CoilCollection(coils, currents, 3, True)
#        else:
#            # Create a stellarator that still has rotational symmetry but
#            # doesn't have stellarator symmetry. We do this by first applying
#            # stellarator symmetry, then breaking this slightly, and then
#            # applying rotational symmetry
#            from simsopt.geo.curve import RotatedCurve
#            coils_flipped = [RotatedCurve(c, 0, True) for c in coils]
#            currents_flipped = [-cur for cur in currents]
#            for c in coils_flipped:
#                c.rotmat += 0.001*np.random.uniform(low=-1., high=1., size=c.rotmat.shape)
#                c.rotmatT = c.rotmat.T
#            stellarator = CoilCollection(coils + coils_flipped, currents + currents_flipped, 3, False)
#
#        bs = BiotSavart(stellarator.coils, stellarator.currents)
#
#        s = get_surface(surfacetype, stellsym)
#        s.fit_to_curve(ma, 0.1)
#        iota = -0.3
#
#        ar = Area(s)
#        ar_target = ar.J()
#        boozer_surface = BoozerSurface(bs, s, ar, ar_target)
#
#        if optimize_G:
#            G = 2.*np.pi*np.sum(np.abs(bs.coil_currents))*(4*np.pi*10**(-7)/(2 * np.pi))
#        else:
#            G = None
#
#        # compute surface first using LBFGS exact and an area constraint
#        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
#            tol=1e-9, maxiter=500, constraint_weight=100., iota=iota, G=G)
#        print('Residual norm after LBFGS', np.sqrt(2*res['fun']))
#        if second_stage == 'ls':
#            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
#                tol=1e-11, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'])
#        elif second_stage == 'newton':
#            res = boozer_surface.minimize_boozer_penalty_constraints_newton(
#                tol=1e-9, maxiter=10, constraint_weight=100., iota=res['iota'], G=res['G'], stab=1e-4)
#        elif second_stage == 'newton_exact':
#            res = boozer_surface.minimize_boozer_exact_constraints_newton(
#                tol=1e-9, maxiter=10, iota=res['iota'], G=res['G'])
#        elif second_stage == 'residual_exact':
#            res = boozer_surface.solve_residual_equation_exactly_newton(
#                tol=1e-12, maxiter=10, iota=res['iota'], G=res['G'])
#
#        print('Residual norm after second stage', np.linalg.norm(res['residual']))
#        assert res['success']
#        # For the stellsym case we have z(0, 0) = y(0, 0) = 0. For the not
#        # stellsym case, we enforce z(0, 0) = 0, but expect y(0, 0) \neq 0
#        gammazero = s.gamma()[0, 0, :]
#        assert np.abs(gammazero[2]) < 1e-10
#        if stellsym:
#            assert np.abs(gammazero[1]) < 1e-10
#        else:
#            assert np.abs(gammazero[1]) > 1e-6
#
#        if surfacetype == 'SurfaceXYZTensorFourier':
#            assert np.linalg.norm(res['residual']) < 1e-9
#
#        print(ar_target, ar.J())
#        print(res['residual'][-10:])
#        if surfacetype == 'SurfaceXYZTensorFourier' or second_stage == 'newton_exact':
#            assert np.abs(ar_target - ar.J()) < 1e-9
#        else:
#            assert np.abs(ar_target - ar.J()) < 1e-4


if __name__ == "__main__":
    unittest.main()
