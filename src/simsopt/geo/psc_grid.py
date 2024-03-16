from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp
from simsopt.field import BiotSavart
from scipy.integrate import simpson, dblquad, IntegrationWarning
from scipy.special import roots_chebyt, roots_legendre, roots_jacobi
import time

__all__ = ['PSCgrid']


class PSCgrid:
    r"""
    ``PSCgrid`` is a class for setting up the grid,
    plasma surface, and other objects needed to perform PSC
    optimization for stellarators. The class
    takes as input two toroidal surfaces specified as SurfaceRZFourier
    objects, and initializes a set of points (in Cartesian coordinates)
    between these surfaces.

    Args:
        plasma_boundary: Surface class object 
            Representing the plasma boundary surface. Gets converted
            into SurfaceRZFourier object for ease of use.
        Bn: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
            Magnetic field (coils and plasma) at the plasma
            boundary. Typically this will be the optimized plasma
            magnetic field from a stage-1 optimization, and the
            optimized coils from a basic stage-2 optimization.
            This variable must be specified to run PSC optimization.
    """

    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7

    def _setup_uniform_grid(self):
        """
        Initializes a uniform grid in cartesian coordinates and sets
        some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the two boundaries
        self.xyz_inner = self.inner_toroidal_surface.gamma().reshape(-1, 3)
        self.xyz_outer = self.outer_toroidal_surface.gamma().reshape(-1, 3)
        x_outer = self.xyz_outer[:, 0]
        y_outer = self.xyz_outer[:, 1]
        z_outer = self.xyz_outer[:, 2]

        x_max = np.max(x_outer)
        x_min = np.min(x_outer)
        y_max = np.max(y_outer)
        y_min = np.min(y_outer)
        z_max = np.max(z_outer)
        z_min = np.min(z_outer)
        z_max = max(z_max, abs(z_min))
        # print(x_min, x_max, y_min, y_max, z_min, z_max)

        # Initialize uniform grid
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.dz = 2 * z_max / (Nz - 1)
        print(Nx, Ny, Nz, self.dx, self.dy, self.dz)
        Nmin = min(self.dx, min(self.dy, self.dz))
        self.R = Nmin / 4.0
        if self.poff is not None:
            self.R = min(self.R, self.poff / 2.0)  # Need to keep coils from touching, so denominator should be > 2

        self.a = self.R / 100.0
        print('Major radius of the coils is R = ', self.R)
        print('Coils are spaced so that every coil of radius R '
              ' is at least 2R away from the next coil'
        )

        # Extra work below so that the stitching with the symmetries is done in
        # such a way that the reflected cells are still dx and dy away from
        # the old cells.
        #### Note that Cartesian cells can only do nfp = 2, 4, 6, ... 
        #### and correctly be rotated to have the right symmetries
        print(self.dx, self.dy, x_max, y_max, x_min, y_min, self.dz, self.R)
        # exit()
        if self.plasma_boundary.nfp > 1:
            # Throw away any points not in the section phi = [0, pi / n_p] and
            # make sure all centers points are at least a distance R from the
            # sector so that all the coil points are reflected correctly. 
            X = np.linspace(
                self.dx / 2.0 + x_min, x_max - self.dx / 2.0, 
                Nx, endpoint=True
            )
            Y = np.linspace(
                self.dy / 2.0 + y_min, y_max - self.dy / 2.0, 
                Ny, endpoint=True
            )
        else:
            X = np.linspace(x_min, x_max, Nx, endpoint=True)
            Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)
        

        # Extra work for nfp > 1 to chop off points outside sector
        if self.nfp > 1:
            inds = []
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        phi = np.arctan2(Y[i, j, k], X[i, j, k])
                        phi2 = np.arctan2(self.R, X[i, j, k])
                        # Safety factor to avoid phi = 45 degrees exactly, for instance
                        if phi >= (np.pi / self.nfp - phi2) or phi < 0.0:
                            inds.append(int(i * Ny * Nz + j * Nz + k))
            good_inds = np.setdiff1d(np.arange(Nx * Ny * Nz), inds)
            self.xyz_uniform = self.xyz_uniform[good_inds, :]
        # else:
        #     # Get (R, Z) coordinates of the outer boundary
        #     rphiz_outer = np.array(
        #         [np.sqrt(self.xyz_outer[:, 0] ** 2 + self.xyz_outer[:, 1] ** 2), 
        #          np.arctan2(self.xyz_outer[:, 1], self.xyz_outer[:, 0]),
        #          self.xyz_outer[:, 2]]
        #     ).T

        #     r_max = np.max(rphiz_outer[:, 0])
        #     r_min = np.min(rphiz_outer[:, 0])
        #     z_max = np.max(rphiz_outer[:, 2])
        #     z_min = np.min(rphiz_outer[:, 2])

        #     # Initialize uniform grid of curved, square bricks
        #     Nr = int((r_max - r_min) / self.dr)
        #     self.Nr = Nr
        #     self.dz = self.dr
        #     Nz = int((z_max - z_min) / self.dz)
        #     self.Nz = Nz
        #     phi = 2 * np.pi * np.copy(self.plasma_boundary.quadpoints_phi)
        #     R = np.linspace(r_min, r_max, Nr)
        #     Z = np.linspace(z_min, z_max, Nz)

        #     # Make 3D mesh
        #     R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
        #     X = R * np.cos(Phi)
        #     Y = R * np.sin(Phi)
        #     self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(-1, 3)

        # Save uniform grid before we start chopping off parts.
        contig = np.ascontiguousarray
        pointsToVTK('uniform_grid', contig(self.xyz_uniform[:, 0]),
                    contig(self.xyz_uniform[:, 1]), contig(self.xyz_uniform[:, 2]))

    @classmethod
    def geo_setup_between_toroidal_surfaces(
        cls, 
        plasma_boundary : Surface,
        Bn,
        B_TF,
        inner_toroidal_surface: Surface, 
        outer_toroidal_surface: Surface,
        **kwargs,
    ):
        """
        Function to initialize a SIMSOPT PermanentMagnetGrid from a 
        volume defined by two toroidal surfaces. These must be specified
        directly. Often a good choice is made by extending the plasma 
        boundary by its normal vectors.

        Args
        ----------
        inner_toroidal_surface: Surface class object 
            Representing the inner toroidal surface of the volume.
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        outer_toroidal_surface: Surface object representing
            the outer toroidal surface of the volume. Typically 
            want this to have same quadrature points as the inner
            surface for a functional grid setup. 
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        kwargs: The following are valid keyword arguments.
            Nx: int
                Number of points in x to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Nx is the x-size of the
                rectangular cubes in the grid.
            Ny: int
                Number of points in y to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Ny is the y-size of the
                rectangular cubes in the grid.
            Nz: int
                Number of points in z to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Nz is the z-size of the
                rectangular cubes in the grid.
            Nt: int
                Number of turns of the coil. 
        Returns
        -------
        psc_grid: An initialized PSCgrid class object.

        """
        from simsopt.field import InterpolatedField
        from simsopt.util import calculate_on_axis_B
                
        psc_grid = cls() 
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
        
        if not isinstance(B_TF, BiotSavart):
            raise ValueError('The magnetic field from the energized coils '
                ' must be passed as a BiotSavart object.'
        )
        # psc_grid.B_TF = B_TF
        # Many big calls to B_TF.B() so make an interpolated object
        psc_grid.plasma_boundary = plasma_boundary.to_RZFourier()
        psc_grid.nfp = plasma_boundary.nfp
        psc_grid.stellsym = plasma_boundary.stellsym
        psc_grid.symmetry = plasma_boundary.nfp * (plasma_boundary.stellsym + 1)
        if psc_grid.stellsym:
            psc_grid.stell_list = [1, -1]
        else:
            psc_grid.stell_list = [1]
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        psc_grid.plasma_unitnormals = psc_grid.plasma_boundary.unitnormal().reshape(-1, 3)
        psc_grid.plasma_points = psc_grid.plasma_boundary.gamma().reshape(-1, 3)
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        # integration over phi for L
        N = 10
        psc_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.dphi = psc_grid.phi[1] - psc_grid.phi[0]
        N = 40
        psc_grid.flux_phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.flux_dphi = psc_grid.flux_phi[1] - psc_grid.flux_phi[0]
        Nx = kwargs.pop("Nx", 10)
        Ny = Nx  # kwargs.pop("Ny", 10)
        Nz = Nx  # kwargs.pop("Nz", 10)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        psc_grid.Nx = Nx
        psc_grid.Ny = Ny
        psc_grid.Nz = Nz
        psc_grid.poff = kwargs.pop("poff", None)
        psc_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        psc_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()    
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a permanent'
            ' magnet grid to be correctly initialized.'
        )        

        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        
        psc_grid.out_dir = kwargs.pop("out_dir", '')

        # Have the uniform grid, now need to loop through and eliminate cells.
        t1 = time.time()
        contig = np.ascontiguousarray
        normal_inner = inner_toroidal_surface.unitnormal().reshape(-1, 3)   
        normal_outer = outer_toroidal_surface.unitnormal().reshape(-1, 3)   
        psc_grid._setup_uniform_grid()
        psc_grid.grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner), 
            contig(normal_outer), 
            contig(psc_grid.xyz_uniform), 
            contig(psc_grid.xyz_inner), 
            contig(psc_grid.xyz_outer))
        inds = np.ravel(np.logical_not(np.all(psc_grid.grid_xyz == 0.0, axis=-1)))
        psc_grid.grid_xyz = np.array(psc_grid.grid_xyz[inds, :], dtype=float)
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        psc_grid.rho = np.linspace(0, psc_grid.R, N, endpoint=False)
        psc_grid.drho = psc_grid.rho[1] - psc_grid.rho[0]

        # psc_grid.pm_phi = np.arctan2(psc_grid.grid_xyz[:, 1], psc_grid.grid_xyz[:, 0])
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))
        
        # Order of the coils. For unit tests, needs > 400
        psc_grid.ppp = kwargs.pop("ppp", 100)

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        
        # Initialize the coil orientations parallel to the B field. 
        # psc_grid.B_TF = B_TF
        n = 40  # tried 20 here and then there are small errors in f_B 
        degree = 3
        R = psc_grid.R
        gamma_outer = outer_toroidal_surface.gamma().reshape(-1, 3)
        rs = np.linalg.norm(gamma_outer[:, :2], axis=-1)
        zs = gamma_outer[:, 2]
        rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
        if psc_grid.nfp > 1:
            phirange = (0, 2*np.pi/psc_grid.nfp, n*2)
        else:
            phirange = (0, 2 * np.pi, n * 2)
        if psc_grid.stellsym:
            zrange = (0, np.max(zs) + R, n // 2)
        else:
            zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
        psc_grid.B_TF = InterpolatedField(
            B_TF, degree, rrange, phirange, zrange, 
            True, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
        )
        B_TF.set_points(psc_grid.grid_xyz)
        B = B_TF.B()
        B_axis = calculate_on_axis_B(B_TF, psc_grid.plasma_boundary, print_out=False)
        psc_grid.normalization = B_axis ** 2 * psc_grid.plasma_boundary.area()

        # Randomly initialize the coil orientations
        random_initialization = kwargs.pop("random_initialization", False)
        if random_initialization:
            psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
            psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
            psc_grid.coil_normals = np.array(
                [np.cos(psc_grid.alphas) * np.sin(psc_grid.deltas),
                  -np.sin(psc_grid.alphas),
                  np.cos(psc_grid.alphas) * np.cos(psc_grid.deltas)]
            ).T
            # deal with -0 terms in the normals, which screw up the arctan2 calculations
            psc_grid.coil_normals[
                np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
                               np.copysign(1.0, psc_grid.coil_normals) < 0)
                ] *= -1.0
        else:
            # determine the alphas and deltas from these normal vectors
            psc_grid.coil_normals = (B.T / np.sqrt(np.sum(B ** 2, axis=-1))).T
            # deal with -0 terms in the normals, which screw up the arctan2 calculations
            psc_grid.coil_normals[
                np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
                               np.copysign(1.0, psc_grid.coil_normals) < 0)
                ] *= -1.0
            deltas = np.arctan2(psc_grid.coil_normals[:, 0], psc_grid.coil_normals[:, 2])
            alphas = -np.arcsin(psc_grid.coil_normals[:, 1])
            
        psc_grid.setup_full_grid()
        t2 = time.time()
        print('Initialize grid time = ', t2 - t1)
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        
        t1 = time.time()
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        t2 = time.time()
        print('Geo setup time = ', t2 - t1)
        psc_grid.setup_curves()
        psc_grid.plot_curves()
        psc_grid.b_vector()
        kappas = np.ravel(np.array([psc_grid.alphas, psc_grid.deltas]))
        psc_grid.kappas = kappas
        psc_grid.A_opt = psc_grid.least_squares(kappas)

        # psc_grid._optimization_setup()
        return psc_grid
    
    @classmethod
    def geo_setup_manual(
        cls, 
        points,
        R,
        a,
        alphas,
        deltas,
        **kwargs,
    ):
        from simsopt.util.permanent_magnet_helper_functions import initialize_coils, calculate_on_axis_B
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        from simsopt.field import InterpolatedField
        
        psc_grid = cls()
        psc_grid.grid_xyz = np.array(points, dtype=float)
        psc_grid.R = R
        psc_grid.a = a
        psc_grid.alphas = alphas
        psc_grid.deltas = deltas
        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        psc_grid.out_dir = kwargs.pop("out_dir", '')
        
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        N = 1000  # decrease if not doing unit tests
        psc_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.dphi = psc_grid.phi[1] - psc_grid.phi[0]
        psc_grid.rho = np.linspace(0, R, N, endpoint=False)
        psc_grid.drho = psc_grid.rho[1] - psc_grid.rho[0]
        psc_grid.flux_phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
        # initialize default plasma boundary
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        default_surf = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=16, ntheta=16
        )
        default_surf.nfp = 1
        default_surf.stellsym = False
        psc_grid.plasma_boundary = kwargs.pop("plasma_boundary", default_surf)
        psc_grid.nfp = psc_grid.plasma_boundary.nfp
        psc_grid.stellsym = psc_grid.plasma_boundary.stellsym
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        psc_grid.plasma_unitnormals = psc_grid.plasma_boundary.unitnormal().reshape(-1, 3)
        psc_grid.plasma_points = psc_grid.plasma_boundary.gamma().reshape(-1, 3)
        psc_grid.symmetry = psc_grid.plasma_boundary.nfp * (psc_grid.plasma_boundary.stellsym + 1)

        if psc_grid.stellsym:
            psc_grid.stell_list = [1, -1]
        else:
            psc_grid.stell_list = [1]
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        Bn = kwargs.pop("Bn", np.zeros((16, 16)))
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
        
        # generate planar TF coils
        ncoils = 2
        R0 = 1.0
        R1 = 0.65
        order = 4
        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        total_current = 1e3
        base_curves = create_equally_spaced_curves(
            ncoils, psc_grid.plasma_boundary.nfp, 
            stellsym=psc_grid.plasma_boundary.stellsym, 
            R0=R0, R1=R1, order=order, numquadpoints=32
        )
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        default_coils = coils_via_symmetries(
            base_curves, base_currents, 
            psc_grid.plasma_boundary.nfp, 
            psc_grid.plasma_boundary.stellsym
        )
        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()
            
        B_TF = kwargs.pop("B_TF", BiotSavart(default_coils))
        
        # Order of the coils. For unit tests, needs > 400
        psc_grid.ppp = kwargs.pop("ppp", 1000)
        
        # Need interpolated region big enough to cover all the coils and the plasma
        n = 20
        degree = 2
        rs = np.linalg.norm(psc_grid.grid_xyz[:, :2] + R ** 2, axis=-1)
        zs = psc_grid.grid_xyz[:, 2]
        rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
        if psc_grid.nfp > 1:
            phirange = (0, 2*np.pi/psc_grid.nfp, n*2)
        else:
            phirange = (0, 2 * np.pi, n * 2)
        if psc_grid.stellsym:
            zrange = (0, np.max(zs) + R, n // 2)
        else:
            zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
        psc_grid.B_TF = InterpolatedField(
            B_TF, degree, rrange, phirange, zrange, 
            True, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
        )
        # psc_grid.B_TF = B_TF
        B_axis = calculate_on_axis_B(B_TF, psc_grid.plasma_boundary, print_out=False)
        psc_grid.normalization = B_axis ** 2 * psc_grid.plasma_boundary.area()
        
        contig = np.ascontiguousarray
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        # psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        # psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        psc_grid.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        psc_grid.coil_normals[
            np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
                           np.copysign(1.0, psc_grid.coil_normals) < 0)
            ] *= -1.0

        # psc_grid.coil_normals = np.array(
        #     [np.cos(deltas) * np.sin(alphas),
        #      np.sin(deltas) * np.sin(alphas),
        #      np.cos(alphas)]
        # ).T
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        psc_grid.setup_full_grid()
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        psc_grid.setup_curves()
        psc_grid.plot_curves()
        psc_grid.b_vector()
        kappas = np.ravel(np.array([psc_grid.alphas, psc_grid.deltas]))
        psc_grid.kappas = kappas
        psc_grid.A_opt = psc_grid.least_squares(kappas)
        
        return psc_grid
    
    def setup_currents_and_fields(self):
        """ """
        # Note L will be well-conditioned unless you accidentally initialized
        # the PSC coils touching/intersecting, in which case it will be
        # VERY ill-conditioned!
        self.L_inv = np.linalg.inv(self.L)
        self.psi_total = np.zeros(self.num_psc * self.symmetry)
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                self.psi_total[q * self.num_psc:(q + 1) * self.num_psc] = self.psi * stell
                q = q + 1
        self.I_all = (-self.L_inv @ self.psi_total)
        self.I = self.I_all[:self.num_psc]
        contig = np.ascontiguousarray
        # A_matrix has shape (num_plasma_points, num_coils)
        A_matrix = np.zeros((self.nphi * self.ntheta, self.num_psc))
        # Need to rotate and flip it
        nn = self.num_psc
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2.0 * np.pi / self.nfp) * fp
                ox = self.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                oy = self.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                oz = self.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                xyz = np.array([ox, oy, oz]).T
                normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                # alphas = self.alphas_total[q * self.num_psc: (q + 1) * self.num_psc]
                # deltas = self.deltas_total[q * self.num_psc: (q + 1) * self.num_psc]
                
                # Not sure this is right. Already flipping and rotating alphas
                # and deltas, and then phi0 and stell are doing it again...
                # So think I should use self.alphas and self.deltas
                # Bn_PSC += sopp.A_matrix(
                #     contig(xyz),
                #     contig(psc_array.plasma_points),
                #     contig(psc_array.alphas_total[q * nn: (q + 1) * nn]),
                #     contig(psc_array.deltas_total[q * nn: (q + 1) * nn]),
                #     contig(psc_array.plasma_boundary.unitnormal().reshape(-1, 3)),
                #     psc_array.R,
                #     0.0,
                #     1.0
                # ) @ (psc_array.I)
                A_matrix += sopp.A_matrix(
                    contig(xyz),
                    contig(self.plasma_points),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.plasma_unitnormals),
                    self.R,
                    0.0,
                    1.0
                )  # accounts for sign change of the currents
                q = q + 1
        self.Bn_PSC = A_matrix @ self.I
        
    def setup_psc_biotsavart(self):
        from simsopt.field import coils_via_symmetries, Current
        # self.B_PSC = CircularCoil(self.R, self.grid_xyz[0, :], self.I[0], self.coil_normals[0, :])
        # self.B_PSC.set_points(self.plasma_boundary.gamma().reshape(-1, 3))
        # for i in range(1, len(self.alphas)):
        #     self.B_PSC += CircularCoil(self.R, self.grid_xyz[i, :], self.I[i], self.coil_normals[i, :])
        #     self.B_PSC.set_points(self.plasma_boundary.gamma().reshape(-1, 3))
        self.setup_curves()
        currents = []
        for i in range(len(self.I)):
            currents.append(Current(self.I[i]))
        
        coils = coils_via_symmetries(
            self.curves, currents, nfp=self.nfp, stellsym=self.stellsym
        )
        self.B_PSC = BiotSavart(coils)
        self.B_PSC.set_points(self.plasma_points)
    
    def setup_curves(self):
        """ """
        from . import CurvePlanarFourier
        from simsopt.field import apply_symmetries_to_curves

        order = 1
        ncoils = self.num_psc
        curves = [CurvePlanarFourier(order*self.ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        for ic in range(ncoils):
            xyz = self.grid_xyz[ic, :]
            dofs = np.zeros(10)
            dofs[0] = self.R
            dofs[1] = 0.0
            dofs[2] = 0.0
            # Conversion from Euler angles in 3-2-1 body sequence to 
            # quaternions: 
            # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            dofs[3] = np.cos(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[4] = np.sin(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[5] = np.cos(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            dofs[6] = -np.sin(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            # Now specify the center 
            dofs[7] = xyz[0]
            dofs[8] = xyz[1]
            dofs[9] = xyz[2] 
            curves[ic].set_dofs(dofs)
        self.curves = curves
        self.all_curves = apply_symmetries_to_curves(curves, self.nfp, self.stellsym)

    def plot_curves(self, filename=''):
        
        from pyevtk.hl import pointsToVTK
        from . import curves_to_vtk
        from simsopt.field import InterpolatedField, Current, coils_via_symmetries
        
        curves_to_vtk(self.curves, self.out_dir + "psc_curves", close=True, scalar_data=self.I)
        contig = np.ascontiguousarray
        if isinstance(self.B_TF, InterpolatedField):
            self.B_TF.set_points(self.grid_xyz)
            B = self.B_TF.B()
            pointsToVTK(self.out_dir + 'curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              "psi": contig(self.psi),
                              "I": contig(self.I),
                              "B_TF": (contig(B[:, 0]), 
                                        contig(B[:, 1]),
                                        contig(B[:, 2])),
                              },
            )
        else:
            pointsToVTK(self.out_dir + 'curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              },
            )
        currents = []
        # for i in range(len(self.I)):
        #     currents.append(Current(self.I[i]))
        # all_coils = coils_via_symmetries(
        #     self.curves, currents, nfp=self.nfp, stellsym=self.stellsym
        # )
        for fp in range(self.nfp):
            for stell in self.stell_list:
                for i in range(self.num_psc):
                    currents.append(Current(self.I[i] * stell))
        all_coils = coils_via_symmetries(
            self.all_curves, currents, nfp=1, stellsym=False
        )
        self.B_PSC_all = BiotSavart(all_coils)

        curves_to_vtk(self.all_curves, self.out_dir + filename + "all_psc_curves", close=True, scalar_data=self.I_all)
        contig = np.ascontiguousarray
        if isinstance(self.B_TF, InterpolatedField):
            self.B_TF.set_points(self.grid_xyz_all)
            B = self.B_TF.B()
            pointsToVTK(self.out_dir + filename + 'all_curve_centers', 
                        contig(self.grid_xyz_all[:, 0]),
                        contig(self.grid_xyz_all[:, 1]), 
                        contig(self.grid_xyz_all[:, 2]),
                        data={"n": (contig(self.coil_normals_all[:, 0]), 
                                    contig(self.coil_normals_all[:, 1]),
                                    contig(self.coil_normals_all[:, 2])),
                              "psi": contig(self.psi_total),
                              "I": contig(self.I_all),
                              "B_TF": (contig(B[:, 0]), 
                                       contig(B[:, 1]),
                                       contig(B[:, 2])),
                              },
            )
        else:
            pointsToVTK(self.out_dir + filename + 'all_curve_centers', 
                        contig(self.grid_xyz_all[:, 0]),
                        contig(self.grid_xyz_all[:, 1]), 
                        contig(self.grid_xyz_all[:, 2]),
                        data={"n": (contig(self.coil_normals_all[:, 0]), 
                                    contig(self.coil_normals_all[:, 1]),
                                    contig(self.coil_normals_all[:, 2])),
                              },
            )
        
    def b_vector(self):
        Bn_target = self.Bn.reshape(-1)
        self.B_TF.set_points(self.plasma_points)
        Bn_TF = np.sum(
            self.B_TF.B().reshape(-1, 3) * self.plasma_unitnormals, axis=-1
        )
        self.b_opt = (Bn_TF + Bn_target) * self.grid_normalization
        
    def least_squares(self, kappas):
        alphas = kappas[:len(kappas) // 2]
        deltas = kappas[len(kappas) // 2:]
        self.setup_orientations(alphas, deltas)
        BdotN2 = 0.5 * np.sum((self.Bn_PSC.reshape(-1) * self.grid_normalization + self.b_opt) ** 2) / self.normalization
        outstr = f"Normalized f_B = {BdotN2:.3e} "
        # for i in range(len(kappas)):
        #     outstr += f"kappas[{i:d}] = {kappas[i]:.2e} "
        # outstr += "\n"
        print(outstr)
        return BdotN2
        
    def setup_orientations(self, alphas, deltas):
        self.alphas = alphas
        self.deltas = deltas
        contig = np.ascontiguousarray
        self.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        self.coil_normals[
            np.logical_and(np.isclose(self.coil_normals, 0.0), 
                           np.copysign(1.0, self.coil_normals) < 0)
            ] *= -1.0
        # rotation_matrix = np.zeros((len(alphas), 3, 3))
        # for i in range(len(alphas)):
        #     rotation_matrix[i, :, :] = np.array(
        #         [[np.cos(deltas[i]), 
        #           np.sin(deltas[i]) * np.sin(alphas[i]),
        #           np.sin(deltas[i]) * np.cos(alphas[i])],
        #           [0.0, np.cos(alphas[i]), -np.sin(alphas[i])],
        #           [-np.sin(deltas[i]), 
        #            np.cos(deltas[i]) * np.sin(alphas[i]),
        #            np.cos(deltas[i]) * np.cos(alphas[i])]])
        # self.rotation_matrix = rotation_matrix
        # t1 = time.time()
        flux_grid = sopp.flux_xyz(
            contig(self.grid_xyz), 
            contig(alphas),
            contig(deltas), 
            contig(self.rho), 
            contig(self.flux_phi), 
            contig(self.coil_normals)
        )
        self.B_TF.set_points(contig(np.array(flux_grid).reshape(-1, 3)))
        # t2 = time.time()
        # print('Flux set points time = ', t2 - t1)
        N = len(self.rho)
        Nflux = len(self.flux_phi)
        # t1 = time.time()
        self.psi = sopp.flux_integration(
            contig(self.B_TF.B().reshape(len(alphas), N, Nflux, 3)),
            contig(self.rho),
            contig(self.coil_normals)
        )
        self.psi *= self.dphi * self.drho
        # t2 = time.time()
        # print('Flux integration time = ', t2 - t1)
        # t1 = time.time()        
        ####### Inductance C++ calculation below
        self.update_alphas_deltas()
        L_total = sopp.L_matrix(
            contig(self.grid_xyz_all), 
            contig(self.alphas_total), 
            contig(self.deltas_total),
            contig(self.phi),
            self.R,
        ) * self.dphi ** 2 / (4 * np.pi)
        L_total = (L_total + L_total.T)
        np.fill_diagonal(L_total, np.log(8.0 * self.R / self.a) - 2.0)
        self.L = L_total * self.mu0 * self.R * self.Nt ** 2
        # print(L_total)
        # exit()
        # print(L_total, L_total.shape)
        # I_total = - np.linalg.solve(L_total, psi_total)
        # L = sopp.L_matrix(
        #     contig(self.grid_xyz), 
        #     contig(alphas), 
        #     contig(deltas),
        #     contig(self.phi),
        #     self.R,
        # ) * self.dphi ** 2 / (4 * np.pi)
        # t2 = time.time()
        # print('Inductances c++ total time = ', t2 - t1)
        # L = (L + L.T)
        # np.fill_diagonal(L, np.log(8.0 * self.R / self.a) - 2.0)
        # self.L = L * self.mu0 * self.R * self.Nt ** 2
        
        # t1 = time.time()
        self.setup_currents_and_fields()
        # t2 = time.time()
        # print('Setup fields time = ', t2 - t1)
        
    def setup_full_grid(self):
        nn = self.num_psc
        self.grid_xyz_all = np.zeros((nn * self.symmetry, 3))
        self.alphas_total = np.zeros(nn * self.symmetry)
        self.deltas_total = np.zeros(nn * self.symmetry)
        self.coil_normals_all = np.zeros((nn * self.symmetry, 3))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new locations by flipping the y and z components, then rotating by phi0
                self.grid_xyz_all[nn * q: nn * (q + 1), 0] = self.grid_xyz[:, 0] * np.cos(phi0) - self.grid_xyz[:, 1] * np.sin(phi0) * stell
                self.grid_xyz_all[nn * q: nn * (q + 1), 1] = self.grid_xyz[:, 0] * np.sin(phi0) + self.grid_xyz[:, 1] * np.cos(phi0) * stell
                self.grid_xyz_all[nn * q: nn * (q + 1), 2] = self.grid_xyz[:, 2] * stell
                # get new normal vectors by flipping the x component, then rotating by phi0
                self.coil_normals_all[nn * q: nn * (q + 1), 0] = self.coil_normals[:, 0] * np.cos(phi0) * stell - self.coil_normals[:, 1] * np.sin(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 1] = self.coil_normals[:, 0] * np.sin(phi0) * stell + self.coil_normals[:, 1] * np.cos(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 2] = self.coil_normals[:, 2]
                normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                deltas = np.arctan2(normals[:, 0], normals[:, 2])
                alphas = -np.arcsin(normals[:, 1])
                self.alphas_total[nn * q: nn * (q + 1)] = alphas
                self.deltas_total[nn * q: nn * (q + 1)] = deltas
                q = q + 1
    
    def update_alphas_deltas(self):
        nn = self.num_psc
        self.alphas_total = np.zeros(nn * self.symmetry)
        self.deltas_total = np.zeros(nn * self.symmetry)
        self.coil_normals_all = np.zeros((nn * self.symmetry, 3))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new normal vectors by flipping the x component, then rotating by phi0
                self.coil_normals_all[nn * q: nn * (q + 1), 0] = self.coil_normals[:, 0] * np.cos(phi0) * stell - self.coil_normals[:, 1] * np.sin(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 1] = self.coil_normals[:, 0] * np.sin(phi0) * stell + self.coil_normals[:, 1] * np.cos(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 2] = self.coil_normals[:, 2]
                normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                deltas = np.arctan2(normals[:, 0], normals[:, 2])
                alphas = -np.arcsin(normals[:, 1])
                self.alphas_total[nn * q: nn * (q + 1)] = alphas
                self.deltas_total[nn * q: nn * (q + 1)] = deltas
                q = q + 1