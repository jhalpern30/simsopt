import simsoptpp as sopp
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.warning(str(e))

if MPI is not None:
    try:
        from simsopt.mhd.vmec import Vmec
        from simsopt.mhd.boozer import Boozer
    except ImportError as e:
        Vmec = None
        Boozer = None
        logger.warning(str(e))


class BoozerMagneticField(sopp.BoozerMagneticField):
    r"""
    Generic class that represents a magnetic field in Boozer coordinates
    :math:`(s,\theta,\zeta)`. Here :math:`s = \psi/\psi_0` is the normalized
    toroidal flux where :math:`2\pi\psi_0` is the toroidal flux at the boundary.
    The magnetic field in the covariant form is,

    .. math::
        \textbf B(s,\theta,\zeta) = G(s) \nabla \zeta + I(s) \nabla \theta + K(s,\theta,\zeta) \nabla \psi,

    and the contravariant form is,

    .. math::
        \textbf B(s,\theta,\zeta) = \frac{1}{\sqrt{g}} \left(\frac{\partial \mathbf r}{\partial \zeta} + \iota(s)\frac{\partial \mathbf r}{\partial \theta}\right),

    where,

    .. math::
        \sqrt{g}(s,\theta,\zeta) = \frac{G(s) + \iota(s)I(s)}{B^2}.

    Here :math:`\iota(s) = \psi_P'(\psi)` where :math:`2\pi\psi_P` is the
    poloidal flux and :math:`2\pi\psi` is the toroidal flux. Each subclass of
    :class:`BoozerMagneticField` implements functions to compute
    :math:`B`, :math:`G`, :math:`I`, :math:`\iota`, :math:`\psi_P`, and their
    derivatives. The cylindrical coordinates :math:`R(s,\theta,\zeta)` and
    :math:`Z(s,\theta,\zeta)` in addition to :math:`K(s,\theta,\zeta)` and
    :math:`\nu` where :math:`\zeta = \phi + \nu(s,\theta,\zeta)` and :math:`\phi`
    is the cylindrical azimuthal angle are also implemented by
    :class:`BoozerRadialInterpolant` and :class:`InterpolatedBoozerField`.
    The usage is similar to the :class:`MagneticField` class.

    The usage of :class:`BoozerMagneticField`` is as follows:

    .. code-block::

        booz = BoozerAnalytic(etabar,B0,N,G0,psi0,iota0) # An instance of BoozerMagneticField
        points = ... # points is a (n, 3) numpy array defining :math:`(s,\theta,\zeta)`
        booz.set_points(points)
        modB = bfield.modB() # returns the magnetic field strength at `points`

    :class:`BoozerMagneticField` has a cache to avoid repeated calculations.
    To clear this cache manually, call the :func:`clear_cached_properties()` function.
    The cache is automatically cleared when :func:`set_points` is called or one of the dependencies
    changes.
    """

    def __init__(self, psi0):
        self.psi0 = psi0
        sopp.BoozerMagneticField.__init__(self, psi0)

    def clear_cached_properties(self):
        """Clear the cache."""
        sopp.BoozerMagneticField.invalidate_cache(self)

    def recompute_bell(self, parent=None):
        if np.any(self.dofs_free_status):
            self.clear_cached_properties()

    def _modB_derivs_impl(self, modB_derivs):
        self._dmodBds_impl(np.reshape(modB_derivs[:, 0], (len(modB_derivs[:, 0]), 1)))
        self._dmodBdtheta_impl(np.reshape(modB_derivs[:, 1], (len(modB_derivs[:, 0]), 1)))
        self._dmodBdzeta_impl(np.reshape(modB_derivs[:, 2], (len(modB_derivs[:, 0]), 1)))

    def _K_derivs_impl(self, K_derivs):
        self._dKdtheta_impl(np.reshape(K_derivs[:, 0], (len(K_derivs[:, 0]), 1)))
        self._dKdzeta_impl(np.reshape(K_derivs[:, 1], (len(K_derivs[:, 0]), 1)))

    def _nu_derivs_impl(self, nu_derivs):
        self._dnuds_impl(np.reshape(nu_derivs[:, 0], (len(nu_derivs[:, 0]), 1)))
        self._dnudtheta_impl(np.reshape(nu_derivs[:, 1], (len(nu_derivs[:, 0]), 1)))
        self._dnudzeta_impl(np.reshape(nu_derivs[:, 2], (len(nu_derivs[:, 0]), 1)))

    def _R_derivs_impl(self, R_derivs):
        self._dRds_impl(np.reshape(R_derivs[:, 0], (len(R_derivs[:, 0]), 1)))
        self._dRdtheta_impl(np.reshape(R_derivs[:, 1], (len(R_derivs[:, 0]), 1)))
        self._dRdzeta_impl(np.reshape(R_derivs[:, 2], (len(R_derivs[:, 0]), 1)))

    def _Z_derivs_impl(self, Z_derivs):
        self._dZds_impl(np.reshape(Z_derivs[:, 0], (len(Z_derivs[:, 0]), 1)))
        self._dZdtheta_impl(np.reshape(Z_derivs[:, 1], (len(Z_derivs[:, 0]), 1)))
        self._dZdzeta_impl(np.reshape(Z_derivs[:, 2], (len(Z_derivs[:, 0]), 1)))


class BoozerAnalytic(BoozerMagneticField):
    r"""
    Computes a :class:`BoozerMagneticField` based on a first-order expansion in
    distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma
    Physics 2018). Here the magnetic field strength is expressed as,

    .. math::
        B(s,\theta,\zeta) = B_0 \left(1 + \overline{\eta} \sqrt{2s\psi_0/\overline{B}}\cos(\theta - N \zeta)\right),

    the covariant components are,

    .. math::
        G(s) = G_0 + \sqrt{2s\psi_0/\overline{B}} G_1

        I(s) = I_0 + \sqrt{2s\psi_0/\overline{B}} I_1

        K(s,\theta,\zeta) = \sqrt{2s\psi_0/\overline{B}} K_1 \sin(\theta - N \zeta),

    and the rotational transform is,

    .. math::
        \iota(s) = \iota_0.

    While formally :math:`I_0 = I_1 = G_1 = K_1 = 0`, these terms have been included
    in order to test the guiding center equations at finite beta.

    Args:
        etabar: magnitude of first order correction to magnetic field strength
        B0: magnetic field strength on the axis
        N: helicity of symmetry (integer)
        G0: lowest order toroidal covariant component
        psi0: (toroidal flux)/ (2*pi) on the boundary
        iota0: lowest order rotational transform
        Bbar: normalizing magnetic field strength (defaults to 1)
        I0: lowest order poloidal covariant component (defaults to 0)
        G1: first order correction to toroidal covariant component (defaults to 0)
        I1: first order correction to poloidal covariant component (defaults to 0)
        K1: first order correction to radial covariant component (defaults to 0)
    """

    def __init__(self, etabar, B0, N, G0, psi0, iota0, Bbar=1., I0=0., G1=0.,
                 I1=0., K1=0.):
        self.etabar = etabar
        self.B0 = B0
        self.Bbar = Bbar
        self.N = N
        self.G0 = G0
        self.I0 = I0
        self.I1 = I1
        self.G1 = G1
        self.K1 = K1
        self.iota0 = iota0
        self.psi0 = psi0
        BoozerMagneticField.__init__(self, psi0)

    def set_etabar(self, etabar):
        self.invalidate_cache()
        self.etabar = etabar

    def set_B0(self, B0):
        self.invalidate_cache()
        self.B0 = B0

    def set_Bbar(self, Bbar):
        self.invalidate_cache()
        self.Bbar = Bbar

    def set_N(self, N):
        self.invalidate_cache()
        self.N = N

    def set_G0(self, G0):
        self.invalidate_cache()
        self.G0 = G0

    def set_I0(self, I0):
        self.invalidate_cache()
        self.I0 = I0

    def set_G1(self, G1):
        self.invalidate_cache()
        self.G1 = G1

    def set_I1(self, I1):
        self.invalidate_cache()
        self.I1 = I1

    def set_K1(self, K1):
        self.invalidate_cache()
        self.K1 = K1

    def set_iota0(self, iota0):
        self.invalidate_cache()
        self.iota0 = iota0

    def set_psi0(self, psi0):
        self.invalidate_cache()
        self.psi0 = psi0

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:, 0] = self.psi0*s*self.iota0

    def _iota_impl(self, iota):
        iota[:, 0] = self.iota0

    def _diotads_impl(self, diotads):
        diotads[:, 0] = 0

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        G[:, 0] = self.G0 + s*self.G1

    def _dGds_impl(self, dGds):
        dGds[:, 0] = self.G1

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        I[:, 0] = self.I0 + s*self.I1

    def _dIds_impl(self, dIds):
        dIds[:, 0] = self.I1

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        modB[:, 0] = self.B0*(1 + self.etabar*r*np.cos(thetas-self.N*zetas))

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        drdpsi = 0.5*r/psi
        drds = drdpsi*self.psi0
        dmodBds[:, 0] = self.B0*self.etabar*drds*np.cos(thetas-self.N*zetas)

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdtheta[:, 0] = -self.B0*self.etabar*r*np.sin(thetas-self.N*zetas)

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdzeta[:, 0] = self.N*self.B0*self.etabar*r*np.sin(thetas-self.N*zetas)

    def _K_impl(self, K):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        K[:, 0] = self.K1*r*np.sin(thetas-self.N*zetas)

    def _dKdtheta_impl(self, dKdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dKdtheta[:, 0] = self.K1*r*np.cos(thetas-self.N*zetas)

    def _dKdzeta_impl(self, dKdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dKdzeta[:, 0] = -self.N*self.K1*r*np.cos(thetas-self.N*zetas)


class BoozerRadialInterpolant(BoozerMagneticField):
    r"""
    Given a :class:`Vmec` instance, performs a Boozer coordinate transformation using
    ``BOOZXFORM``.
    The magnetic field can be computed at any point in Boozer
    coordinates using radial spline interpolation (``scipy.interpolate.InterpolatedUnivariateSpline``)
    and an inverse Fourier transform in the two angles.
    Throughout stellarator symmetry is assumed.

    Args:
        equil: instance of :class:`simsopt.mhd.vmec.Vmec` or :class:`simsopt.mhd.boozer.Boozer`.
            If it is an instance of :class:`simsopt.mhd.boozer.Boozer`, the
            `compute_surfs` needs to include all of the grid points in the
            half-radius grid of the corresponding Vmec equilibrium.
        order: (int) order for radial interpolation. Must satisfy 1 <= order <= 5.
        mpol: (int) number of poloidal mode numbers for BOOZXFORM (defaults to 32)
        ntor: (int) number of toroidal mode numbers for BOOZXFORM (defaults to 32)
        N: Helicity of quasisymmetry to enforce. If specified, then the non-symmetric Fourier
            harmonics of :math:`B` and :math:`K` are filtered out. Otherwise, all harmonics are kept.
            (defaults to ``None``)
        enforce_vacuum: If True, a vacuum field is assumed, :math:`G` is
            set to its mean value, :math:`I = 0`, and :math:`K = 0`.
        rescale: If True, use the interpolation method in the DELTA5D code. Here, a few
            of the first radial grid points or (``bmnc``, ``rmnc``, ``zmns``, ``numns``, ``kmns``)
            are deleted (determined by ``ns_delete``). The Fourier harmonics are then rescaled as:
                bmnc(s)/s^(1/2) for m = 1

                bmnc(s)/s for m even and >= 2

                bmnc(s)/s^(3/2) for m odd and >=3

            before performing interpolation and spline differentiation to
            obtain ``dbmncds``. If ``False``, interpolation of the unscaled Fourier
            harmonics and its finite-difference derivative wrt ``s`` is performed
            instead (defaults to ``False``)
        ns_delete: (see ``rescale``) (defaults to 0)
    """

    def __init__(self, equil, order, mpol=32, ntor=32, N=None, enforce_vacuum=False, rescale=False,
                 ns_delete=0):

        if isinstance(equil, Vmec):
            equil.run()
            self.booz = Boozer(equil, mpol, ntor)
            self.booz.register(self.booz.equil.s_half_grid)
            self.booz.run()
        elif isinstance(equil, Boozer):
            self.booz = equil
            # Determine if radial grid for Boozer needs to be updated

            # Grid not initialized
            if len(self.booz.bx.s_in) == 0:
                self.booz.register(self.booz.equil.s_half_grid)
            # Grid does not have correct size
            elif (len(self.booz.bx.s_in) != len(self.booz.bx.s_b)):
                self.booz.register(self.booz.equil.s_half_grid)
            # Grid does not match Vmec half grid
            elif (np.any(self.booz.bx.s_in != self.booz.bx.s_b)):
                self.booz.register(self.booz.equil.s_half_grid)

            # Run booz_xform if needed
            if self.booz.need_to_run_code:
                self.booz.run()

        self.order = order
        self.enforce_qs = False
        self.enforce_vacuum = enforce_vacuum
        self.ns_delete = ns_delete
        self.rescale = rescale
        if (N is not None):
            self.N = N
            self.enforce_qs = True

        BoozerMagneticField.__init__(self, self.booz.equil.wout.phi[-1]/(2*np.pi))

        # self.booz.mpi.comm_world.Barrier()
        if self.booz.mpi.proc0_groups:
            self.init_splines()
        else:
            self.psip_spline = None
            self.G_spline = None
            self.I_spline = None
            self.dGds_spline = None
            self.dIds_spline = None
            self.iota_spline = None
            self.diotads_spline = None
            self.numns_splines = None
            self.rmnc_splines = None
            self.zmns_splines = None
            self.dnumnsds_splines = None
            self.drmncds_splines = None
            self.dzmnsds_splines = None
            self.bmnc_splines = None
            self.dbmncds_splines = None
            self.d_mn_factor_splines = None
            self.mn_factor_splines = None
            self.xm_b = None
            self.xn_b = None

        self.psip_spline = self.booz.mpi.comm_world.bcast(self.psip_spline, root=0)
        self.G_spline = self.booz.mpi.comm_world.bcast(self.G_spline, root=0)
        self.I_spline = self.booz.mpi.comm_world.bcast(self.I_spline, root=0)
        self.dGds_spline = self.booz.mpi.comm_world.bcast(self.dGds_spline, root=0)
        self.dIds_spline = self.booz.mpi.comm_world.bcast(self.dIds_spline, root=0)
        self.iota_spline = self.booz.mpi.comm_world.bcast(self.iota_spline, root=0)
        self.diotads_spline = self.booz.mpi.comm_world.bcast(self.diotads_spline, root=0)
        self.numns_splines = self.booz.mpi.comm_world.bcast(self.numns_splines, root=0)
        self.rmnc_splines = self.booz.mpi.comm_world.bcast(self.rmnc_splines, root=0)
        self.zmns_splines = self.booz.mpi.comm_world.bcast(self.zmns_splines, root=0)
        self.dnumnsds_splines = self.booz.mpi.comm_world.bcast(self.dnumnsds_splines, root=0)
        self.drmncds_splines = self.booz.mpi.comm_world.bcast(self.drmncds_splines, root=0)
        self.dzmnsds_splines = self.booz.mpi.comm_world.bcast(self.dzmnsds_splines, root=0)
        self.bmnc_splines = self.booz.mpi.comm_world.bcast(self.bmnc_splines, root=0)
        self.dbmncds_splines = self.booz.mpi.comm_world.bcast(self.dbmncds_splines, root=0)
        self.d_mn_factor_splines = self.booz.mpi.comm_world.bcast(self.d_mn_factor_splines, root=0)
        self.mn_factor_splines = self.booz.mpi.comm_world.bcast(self.mn_factor_splines, root=0)
        self.xm_b = self.booz.mpi.comm_world.bcast(self.xm_b, root=0)
        self.xn_b = self.booz.mpi.comm_world.bcast(self.xn_b, root=0)
        self.kmns_splines = None

    def init_splines(self):
        self.xm_b = self.booz.bx.xm_b
        self.xn_b = self.booz.bx.xn_b

        # Define quantities on extended half grid
        iota = np.zeros((self.booz.bx.ns_b+2))
        G = np.zeros((self.booz.bx.ns_b+2))
        I = np.zeros((self.booz.bx.ns_b+2))

        self.s_half_ext = np.zeros((self.booz.bx.ns_b+2))
        self.s_half_ext[1:-1] = self.booz.bx.s_in
        self.s_half_ext[-1] = 1

        ds = self.booz.bx.s_in[1]-self.booz.bx.s_in[0]

        s_full = np.linspace(0, 1, self.booz.bx.ns_in+1)

        psip = self.booz.equil.wout.chi/(2*np.pi)
        iota[1:-1] = self.booz.bx.iota
        G[1:-1] = self.booz.bx.Boozer_G
        I[1:-1] = self.booz.bx.Boozer_I
        if self.rescale:
            s_half_mn = self.booz.bx.s_in[self.ns_delete::]
            bmnc = np.zeros((len(self.xm_b), self.booz.bx.ns_in-self.ns_delete))
            rmnc = np.zeros((len(self.xm_b), self.booz.bx.ns_in-self.ns_delete))
            zmns = np.zeros((len(self.xm_b), self.booz.bx.ns_in-self.ns_delete))
            numns = np.zeros((len(self.xm_b), self.booz.bx.ns_in-self.ns_delete))

            bmnc = self.booz.bx.bmnc_b[:, self.ns_delete::]
            rmnc = self.booz.bx.rmnc_b[:, self.ns_delete::]
            zmns = self.booz.bx.zmns_b[:, self.ns_delete::]
            numns = self.booz.bx.numns_b[:, self.ns_delete::]

            mn_factor = np.ones_like(bmnc)
            d_mn_factor = np.zeros_like(bmnc)
            mn_factor[self.xm_b == 1, :] = s_half_mn[None, :]**(-0.5)
            d_mn_factor[self.xm_b == 1, :] = -0.5*s_half_mn[None, :]**(-1.5)
            mn_factor[(self.xm_b % 2 == 1)*(self.xm_b > 1), :] = s_half_mn[None, :]**(-1.5)
            d_mn_factor[(self.xm_b % 2 == 1)*(self.xm_b > 1), :] = -1.5*s_half_mn[None, :]**(-2.5)
            mn_factor[(self.xm_b % 2 == 0)*(self.xm_b > 1), :] = s_half_mn[None, :]**(-1.)
            d_mn_factor[(self.xm_b % 2 == 0)*(self.xm_b > 1), :] = -s_half_mn[None, :]**(-2.)
        else:
            s_half_mn = self.s_half_ext
            bmnc = np.zeros((len(self.xm_b), self.booz.bx.ns_in+2))
            bmnc[:, 1:-1] = self.booz.bx.bmnc_b
            bmnc[:, 0] = 1.5*bmnc[:, 1] - 0.5*bmnc[:, 2]
            bmnc[:, -1] = 1.5*bmnc[:, -2] - 0.5*bmnc[:, -3]
            dbmncds = (bmnc[:, 2:-1] - bmnc[:, 1:-2])/ds
            mn_factor = np.ones_like(bmnc)
            d_mn_factor = np.zeros_like(bmnc)

            numns = np.zeros((len(self.xm_b), self.booz.bx.ns_in+2))
            rmnc = np.zeros((len(self.xm_b), self.booz.bx.ns_in+2))
            zmns = np.zeros((len(self.xm_b), self.booz.bx.ns_in+2))
            numns[:, 1:-1] = self.booz.bx.numns_b
            numns[:, 0] = 1.5*numns[:, 1] - 0.5*numns[:, 2]
            numns[:, -1] = 1.5*numns[:, -2] - 0.5*numns[:, -3]
            rmnc[:, 1:-1] = self.booz.bx.rmnc_b
            rmnc[:, 0] = 1.5*rmnc[:, 1] - 0.5*rmnc[:, 2]
            rmnc[:, -1] = 1.5*rmnc[:, -2] - 0.5*rmnc[:, -3]
            zmns[:, 1:-1] = self.booz.bx.zmns_b
            zmns[:, 0] = 1.5*zmns[:, 1] - 0.5*zmns[:, 2]
            zmns[:, -1] = 1.5*zmns[:, -2] - 0.5*zmns[:, -3]

            drmncds = (rmnc[:, 2:-1] - rmnc[:, 1:-2])/ds
            dzmnsds = (zmns[:, 2:-1] - zmns[:, 1:-2])/ds
            dnumnsds = (numns[:, 2:-1] - numns[:, 1:-2])/ds

        # Extrapolate to get points at s = 0 and s = 1
        iota[0] = 1.5*iota[1] - 0.5*iota[2]
        G[0] = 1.5*G[1] - 0.5*G[2]
        I[0] = 1.5*I[1] - 0.5*I[2]
        iota[-1] = 1.5*iota[-2] - 0.5*iota[-3]
        G[-1] = 1.5*G[-2] - 0.5*G[-3]
        I[-1] = 1.5*I[-2] - 0.5*I[-3]
        # Compute first derivatives - on full grid points in [1,ns-1]
        dGds = (G[2:-1] - G[1:-2])/ds
        dIds = (I[2:-1] - I[1:-2])/ds
        diotads = (iota[2:-1] - iota[1:-2])/ds

        self.psip_spline = InterpolatedUnivariateSpline(s_full, psip, k=self.order)
        if not self.enforce_vacuum:
            self.G_spline = InterpolatedUnivariateSpline(self.s_half_ext, G, k=self.order)
            self.I_spline = InterpolatedUnivariateSpline(self.s_half_ext, I, k=self.order)
            self.dGds_spline = InterpolatedUnivariateSpline(s_full[1:-1], dGds, k=self.order)
            self.dIds_spline = InterpolatedUnivariateSpline(s_full[1:-1], dIds, k=self.order)
        else:
            self.G_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.mean(G)*np.ones_like(self.s_half_ext), k=self.order)
            self.I_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.zeros_like(self.s_half_ext), k=self.order)
            self.dGds_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)
            self.dIds_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)
        self.iota_spline = InterpolatedUnivariateSpline(self.s_half_ext, iota, k=self.order)
        self.diotads_spline = InterpolatedUnivariateSpline(s_full[1:-1], diotads, k=self.order)

        self.numns_splines = []
        self.rmnc_splines = []
        self.zmns_splines = []
        self.dnumnsds_splines = []
        self.drmncds_splines = []
        self.dzmnsds_splines = []
        self.bmnc_splines = []
        self.dbmncds_splines = []
        self.d_mn_factor_splines = []
        self.mn_factor_splines = []
        for im in range(len(self.xm_b)):
            self.numns_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*numns[im, :], k=self.order))
            self.rmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*rmnc[im, :], k=self.order))
            self.zmns_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*zmns[im, :], k=self.order))
            self.mn_factor_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :], k=self.order))
            self.d_mn_factor_splines.append(InterpolatedUnivariateSpline(s_half_mn, d_mn_factor[im, :], k=self.order))
            if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
                self.bmnc_splines.append(InterpolatedUnivariateSpline(s_half_bmnc, 0*bmnc[im, :], k=self.order))
                self.dbmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], 0*dbmncds[im, :], k=self.order))
            else:
                self.bmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*bmnc[im, :], k=self.order))
                if self.rescale:
                    self.dbmncds_splines.append(self.bmnc_splines[-1].derivative())
                else:
                    self.dbmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dbmncds[im, :], k=self.order))

            if self.rescale:
                self.dnumnsds_splines.append(self.numns_splines[-1].derivative())
                self.drmncds_splines.append(self.rmnc_splines[-1].derivative())
                self.dzmnsds_splines.append(self.zmns_splines[-1].derivative())
            else:
                self.dnumnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dnumnsds[im, :], k=self.order))
                self.drmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], drmncds[im, :], k=self.order))
                self.dzmnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dzmnsds[im, :], k=self.order))

    def compute_K(self):
        ntheta = 2 * (2 * self.booz.bx.mboz + 1)
        nzeta = 2 * (2 * self.booz.bx.nboz + 1)
        thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        dtheta = thetas[1]-thetas[0]
        zetas = np.linspace(0, 2*np.pi/self.booz.bx.nfp, nzeta, endpoint=False)
        dzeta = zetas[1]-zetas[0]
        thetas, zetas = np.meshgrid(thetas, zetas)
        thetas = thetas.flatten()
        zetas = zetas.flatten()

        dzmnsds_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        drmncds_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        dnumnsds_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        bmnc_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        rmnc_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        zmns_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        numns_half = np.zeros((len(self.xm_b), len(self.s_half_ext)))
        for im in range(len(self.xm_b)):
            mn_factor = self.mn_factor_splines[im](self.s_half_ext)
            d_mn_factor = self.d_mn_factor_splines[im](self.s_half_ext)
            dnumnsds_half[im, :] = ((self.dnumnsds_splines[im](self.s_half_ext) - self.numns_splines[im](self.s_half_ext)*d_mn_factor/mn_factor)/mn_factor)
            drmncds_half[im, :] = ((self.drmncds_splines[im](self.s_half_ext) - self.rmnc_splines[im](self.s_half_ext)*d_mn_factor/mn_factor)/mn_factor)
            dzmnsds_half[im, :] = ((self.dzmnsds_splines[im](self.s_half_ext) - self.zmns_splines[im](self.s_half_ext)*d_mn_factor/mn_factor)/mn_factor)
            bmnc_half[im, :] = self.bmnc_splines[im](self.s_half_ext)/self.mn_factor_splines[im](self.s_half_ext)
            rmnc_half[im, :] = self.rmnc_splines[im](self.s_half_ext)/self.mn_factor_splines[im](self.s_half_ext)
            zmns_half[im, :] = self.zmns_splines[im](self.s_half_ext)/self.mn_factor_splines[im](self.s_half_ext)
            numns_half[im, :] = self.numns_splines[im](self.s_half_ext)/self.mn_factor_splines[im](self.s_half_ext)

        G_half = self.G_spline(self.s_half_ext)
        I_half = self.I_spline(self.s_half_ext)
        iota_half = self.iota_spline(self.s_half_ext)

        kmns = sopp.compute_kmns(rmnc_half, drmncds_half, zmns_half, dzmnsds_half,
                                 numns_half, dnumnsds_half, bmnc_half, iota_half, G_half, I_half,
                                 self.xm_b, self.xn_b, thetas, zetas)
        kmns = kmns*dtheta*dzeta*self.booz.bx.nfp/self.psi0

        self.kmns_splines = []
        for im in range(len(self.xm_b)):
            if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
                self.kmns_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, 0*kmns[im, :], k=self.order))
            else:
                self.kmns_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, self.mn_factor_splines[im](self.s_half_ext)*kmns[im, :], k=self.order))

    def _K_impl(self, K):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        K[:, 0] = 0.
        if self.enforce_vacuum:
            return
        if self.kmns_splines is None:
            if self.booz.mpi.proc0_groups:
                self.compute_K()
            self.kmns_splines = self.booz.mpi.comm_world.bcast(self.kmns_splines, root=0)
        kmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                kmns[im, ip] = self.kmns_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_kmns = kmns[:, index]
            K[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_kmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dKdtheta_impl(self, dKdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        dKdtheta[:, 0] = 0.
        if self.enforce_vacuum:
            return
        if self.kmns_splines is None:
            if self.booz.mpi.proc0_groups:
                self.compute_K()
            self.kmns_splines = self.booz.mpi.comm_world.bcast(self.kmns_splines, root=0)
        kmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                kmns[im, ip] = self.kmns_splines[im](s_unique[ip]) * self.xm_b[im]/self.mn_factor_splines[im](s_unique[ip])
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_kmns = kmns[:, index]
            dKdtheta[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_kmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dKdzeta_impl(self, dKdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        dKdzeta[:, 0] = 0.
        if self.enforce_vacuum:
            return
        if self.kmns_splines is None:
            if self.booz.mpi.proc0_groups:
                self.compute_K()
            else:
                self.kmns_splines = None
            self.kmns_splines = self.booz.mpi.comm_world.bcast(self.kmns_splines, root=0)
        kmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                kmns[im, ip] = -self.kmns_splines[im](s_unique[ip]) * self.xn_b[im]/self.mn_factor_splines[im](s_unique[ip])
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_kmns = kmns[:, index]
            dKdzeta[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_kmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _nu_impl(self, nu):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        numns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                numns[im, ip] = self.numns_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        nu[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_numns = numns[:, index]
            nu[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_numns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dnudtheta_impl(self, dnudtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        numns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                numns[im, ip] = self.numns_splines[im](s_unique[ip])*self.xm_b[im]/self.mn_factor_splines[im](s_unique[ip])
        dnudtheta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_numns = numns[:, index]
            dnudtheta[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_numns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dnudzeta_impl(self, dnudzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        numns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                numns[im, ip] = -self.numns_splines[im](s_unique[ip])*self.xn_b[im]/self.mn_factor_splines[im](s_unique[ip])
        dnudzeta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_numns = numns[:, index]
            dnudzeta[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_numns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dnuds_impl(self, dnuds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        numns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                d_mn_factor = self.d_mn_factor_splines[im](s_unique[ip])
                mn_factor = self.mn_factor_splines[im](s_unique[ip])
                numns[im, ip] = ((self.dnumnsds_splines[im](s_unique[ip]) \
                                  - self.numns_splines[im](s_unique[ip])*d_mn_factor/mn_factor)/mn_factor)
        dnuds[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_numns = numns[:, index]
            dnuds[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_numns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dRdtheta_impl(self, dRdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        rmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                rmnc[im, ip] = -self.rmnc_splines[im](s_unique[ip])*self.xm_b[im]/self.mn_factor_splines[im](s_unique[ip])
        dRdtheta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_rmnc = rmnc[:, index]
            dRdtheta[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_rmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dRdzeta_impl(self, dRdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        rmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                rmnc[im, ip] = self.rmnc_splines[im](s_unique[ip])*self.xn_b[im]/self.mn_factor_splines[im](s_unique[ip])
        dRdzeta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_rmnc = rmnc[:, index]
            dRdzeta[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_rmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dRds_impl(self, dRds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        rmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                d_mn_factor = self.d_mn_factor_splines[im](s_unique[ip])
                mn_factor = self.mn_factor_splines[im](s_unique[ip])
                rmnc[im, ip] = ((self.drmncds_splines[im](s_unique[ip]) \
                                 - self.rmnc_splines[im](s_unique[ip])*d_mn_factor/mn_factor)/mn_factor)
        dRds[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_rmnc = rmnc[:, index]
            dRds[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_rmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _R_impl(self, R):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        rmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                rmnc[im, ip] = self.rmnc_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        R[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_rmnc = rmnc[:, index]
            R[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_rmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dZdtheta_impl(self, dZdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        zmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                zmns[im, ip] = self.zmns_splines[im](s_unique[ip])*self.xm_b[im]/self.mn_factor_splines[im](s_unique[ip])
        dZdtheta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_zmns = zmns[:, index]
            dZdtheta[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_zmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dZdzeta_impl(self, dZdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        zmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                zmns[im, ip] = -self.zmns_splines[im](s_unique[ip])*self.xn_b[im]/self.mn_factor_splines[im](s_unique[ip])
        dZdzeta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_zmns = zmns[:, index]
            dZdzeta[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_zmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dZds_impl(self, dZds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        zmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                d_mn_factor = self.d_mn_factor_splines[im](s_unique[ip])
                mn_factor = self.mn_factor_splines[im](s_unique[ip])
                zmns[im, ip] = ((self.dzmnsds_splines[im](s_unique[ip]) \
                                 - self.zmns_splines[im](s_unique[ip])*d_mn_factor/mn_factor)/mn_factor)
        dZds[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_zmns = zmns[:, index]
            dZds[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_zmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _Z_impl(self, Z):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        zmns = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                zmns[im, ip] = self.zmns_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        Z[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_zmns = zmns[:, index]
            Z[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_zmns, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:] = self.psip_spline(s)[:, None]

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        G[:] = self.G_spline(s)[:, None]

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        I[:] = self.I_spline(s)[:, None]

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        iota[:] = self.iota_spline(s)[:, None]

    def _dGds_impl(self, dGds):
        points = self.get_points_ref()
        s = points[:, 0]
        dGds[:] = self.dGds_spline(s)[:, None]

    def _dIds_impl(self, dIds):
        points = self.get_points_ref()
        s = points[:, 0]
        dIds[:] = self.dIds_spline(s)[:, None]

    def _diotads_impl(self, diotads):
        points = self.get_points_ref()
        s = points[:, 0]
        diotads[:] = self.diotads_spline(s)[:, None]

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        modB[:, 0] = 0.
        bmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                bmnc[im, ip] = self.bmnc_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_bmnc = bmnc[:, index]
            modB[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_bmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        bmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                bmnc[im, ip] = -self.xm_b[im]*self.bmnc_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        dmodBdtheta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_bmnc = bmnc[:, index]
            dmodBdtheta[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_bmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        bmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                bmnc[im, ip] = self.xn_b[im]*self.bmnc_splines[im](s_unique[ip])/self.mn_factor_splines[im](s_unique[ip])
        dmodBdzeta[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_bmnc = bmnc[:, index]
            dmodBdzeta[ip, 0] = sopp.inverse_fourier_transform_odd_0d(this_bmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        s_unique = np.unique(s)
        bmnc = np.zeros((len(self.xm_b), len(s_unique)))
        for ip in range(len(s_unique)):
            for im in range(len(self.xm_b)):
                mn_factor = self.mn_factor_splines[im](s_unique[ip])
                d_mn_factor = self.d_mn_factor_splines[im](s_unique[ip])
                bmnc[im, ip] = ((self.dbmncds_splines[im](s_unique[ip]) \
                                 - self.bmnc_splines[im](s_unique[ip])*d_mn_factor/mn_factor)/mn_factor)
        dmodBds[:, 0] = 0.
        for ip in range(len(s)):
            index = np.argwhere(s[ip] == s_unique)[0][0]
            this_bmnc = bmnc[:, index]
            dmodBds[ip, 0] = sopp.inverse_fourier_transform_even_0d(this_bmnc, self.xm_b, self.xn_b, thetas[ip], zetas[ip])

class InterpolatedBoozerField(sopp.InterpolatedBoozerField, BoozerMagneticField):
    r"""
    This field takes an existing :class:`BoozerMagneticField` and interpolates it on a
    regular grid in :math:`s,\theta,\zeta`. This resulting interpolant can then
    be evaluated very quickly. This is modeled after :class:`InterpolatedField`.
    """

    def __init__(self, field, degree, srange, thetarange, zetarange, extrapolate=True, nfp=1, stellsym=True):
        r"""
        Args:
            field: the underlying :class:`simsopt.field.boozermagneticfield.BoozerMagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            srange: a 3-tuple of the form ``(smin, smax, ns)``. This mean that
                the interval ``[smin, smax]`` is split into ``ns`` many subintervals.
            thetarange: a 3-tuple of the form ``(thetamin, thetamax, ntheta)``.
                thetamin must be >= 0, and thetamax must be <=2*pi.
            zetarange: a 3-tuple of the form ``(zetamin, zetamax, nzeta)``.
                zetamin must be >= 0, and thetamax must be <=2*pi.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any toroidal angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``zetamin=0`` and
                 ``zetamax=2*np.pi/nfp``.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``theta`` is always mapped to the interval :math:`[0, \pi]`,
                      hence it makes sense to use ``thetamin=0`` and ``thetamax=np.pi``.
        """
        BoozerMagneticField.__init__(self, field.psi0)
        if (np.any(np.asarray(thetarange[0:2]) < 0) or np.any(np.asarray(thetarange[0:2]) > 2*np.pi)):
            raise ValueError("thetamin and thetamax must be in [0,2*pi]")
        if (np.any(np.asarray(zetarange[0:2]) < 0) or np.any(np.asarray(zetarange[0:2]) > 2*np.pi)):
            raise ValueError("zetamin and zetamax must be in [0,2*pi]")
        if stellsym and (np.any(np.asarray(thetarange[0:2]) < 0) or np.any(np.asarray(thetarange[0:2]) > np.pi)):
            logger.warning(fr"Sure about thetarange=[{thetarange[0]},{thetarange[1]}]? When exploiting stellarator symmetry, the interpolant is only evaluated for theta in [0,pi].")
        if nfp > 1 and (np.any(np.asarray(zetarange[0:2]) < 0) or np.any(np.asarray(zetarange[0:2]) > 2*np.pi/nfp)):
            logger.warning(fr"Sure about zetarange=[{zetarange[0]},{zetarange[1]}]? When exploiting rotational symmetry, the interpolant is only evaluated for zeta in [0,2\pi/nfp].")

        sopp.InterpolatedBoozerField.__init__(self, field, degree, srange, thetarange, zetarange, extrapolate, nfp, stellsym)
