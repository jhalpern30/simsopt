from deprecated import deprecated

from .._core.util import ObjectiveFailure
import numpy as np
from jax import grad, vjp, lax
import jax.numpy as jnp
from jax.lax import dynamic_slice
from .hull import hull2D

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp
from simsopt.geo.framedcurve import FramedCurveCentroid 

__all__ = ['CurveLength', 'LpCurveCurvature', 'LpCurveTorsion',
           'CurveCurveDistance', 'CurveSurfaceDistance', 'ArclengthVariation',
           'MeanSquaredCurvature', 'LinkingNumber', 'CurveCylinderDistance', 'FramedCurveTwist', 'MinCurveCurveDistance']


@jit
def curve_length_pure(l):
    """
    This function is used in a Python+Jax implementation of the curve length formula.
    """
    return jnp.mean(l)


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        self.thisgrad = jit(lambda l: grad(curve_length_pure)(l))
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(self.curve.incremental_arclength())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """

        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature, given by the argument ``threshold``.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_torsion_pure(torsion, gammadash, p, threshold):
    """
    This function is used in a Python+Jax implementation of the formula for the torsion penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(jnp.abs(torsion)-threshold, 0)**p * arc_length)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(|\tau|-\tau_0, 0)^p ~dl.

    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.thisgrad1 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula.
    """
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1)[:, None] * jnp.linalg.norm(l2, axis=1)[None, :]
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])

class CurveCurveDistance(Optimizable):
    r"""
    CurveCurveDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    If num_basecurves is passed, then the code only computes the distance to
    the first `num_basecurves` many curves, which is useful when the coils
    satisfy symmetries that can be exploited.

    """

    def __init__(self, curves, minimum_distance, num_basecurves=None):
        self.curves = curves
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gamma1, l1, gamma2, l2: cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance))
        self.thisgrad0 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2))
        self.thisgrad1 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2))
        self.thisgrad2 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2))
        self.thisgrad3 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2))
        self.candidates = None
        self.num_basecurves = num_basecurves or len(curves)
        super().__init__(depends_on=curves)

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
                [c.gamma() for c in self.curves], self.minimum_distance, self.num_basecurves)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma())) for i, j in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        return min([np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma())) for i in range(len(self.curves)) for j in range(i)])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            res += self.J_jax(gamma1, l1, gamma2, l2)

        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gamma1, l1, gamma2, l2)
            dgamma_by_dcoeff_vjp_vecs[j] += self.thisgrad2(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[j] += self.thisgrad3(gamma1, l1, gamma2, l2)

        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}




def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    
    return jnp.mean(integralweight * jnp.maximum( minimum_distance-dists, 0)**2)


class CurveSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.
    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas, ns: cs_distance_pure(gammac, lc, gammas, ns, minimum_distance))
        self.thisgrad0 = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=0)(gammac, lc, gammas, ns))
        self.thisgrad1 = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=1)(gammac, lc, gammas, ns))
        self.candidates = None
        super().__init__(depends_on=curves)  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.surface.gamma().reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas, ns)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        gammas = self.surface.gamma().reshape((-1, 3))

        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc, gammas, ns)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc, gammas, ns)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    return jnp.var(mat @ l)


class ArclengthVariation(Optimizable):

    def __init__(self, curve, nintervals="full"):
        r"""
        This class penalizes variation of the arclength along a curve.
        The idea of this class is to avoid ill-posedness of curve objectives due to
        non-uniqueness of the underlying parametrization. Essentially we want to
        achieve constant arclength along the curve. Since we can not expect
        perfectly constant arclength along the entire curve, this class has
        some support to relax this notion. Consider a partition of the :math:`[0, 1]`
        interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
        on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance

        .. math::
            J = \mathrm{Var}(\ell_i)

        it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
        If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
        points of the curve. If ``nintervals="partial"``, then the argument is as follows:

        A curve in 3d space is defined uniquely by an initial point, an initial
        direction, and the arclength, curvature, and torsion along the curve. For a
        :mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
        follows: assuming that the curve has order :math:`p`, that means we have
        :math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
        required for both the initial position and direction, :math:`6p-3` are left
        over for curvature, torsion, and arclength. We want to fix the arclength,
        so we can afford :math:`2p-1` constraints, which corresponds to
        :math:`L=2p`.

        Finally, the user can also provide an integer value for `nintervals`
        and thus specify the number of intervals directly.
        """
        super().__init__(depends_on=[curve])

        assert nintervals in ["full", "partial"] \
            or (isinstance(nintervals, int) and 0 < nintervals <= curve.gamma().shape[0])
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        if nintervals == "full":
            nintervals = curve.gamma().shape[0]
        elif nintervals == "partial":
            from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
            if isinstance(curve, CurveXYZFourier) or isinstance(curve, JaxCurveXYZFourier):
                nintervals = 2*curve.order
            else:
                raise RuntimeError("Please provide a value other than `partial` for `nintervals`. We only have a default for `CurveXYZFourier` and `JaxCurveXYZFourier`.")

        self.nintervals = nintervals
        indices = np.floor(np.linspace(0, nquadpoints, nintervals+1, endpoint=True)).astype(int)
        mat = np.zeros((nintervals, nquadpoints))
        for i in range(nintervals):
            mat[i, indices[i]:indices[i+1]] = 1/(indices[i+1]-indices[i])
        self.mat = mat
        self.thisgrad = jit(lambda l: grad(lambda x: curve_arclengthvariation_pure(x, mat))(l))

    def J(self):
        return float(curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat))

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_msc_pure(kappa, gammadash):
    """
    This function is used in a Python+Jax implementation of the mean squared curvature objective.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)


class MeanSquaredCurvature(Optimizable):

    def __init__(self, curve):
        r"""
        Compute the mean of the squared curvature of a curve.

        .. math::
            J = (1/L) \int_{\text{curve}} \kappa^2 ~dl

        where :math:`L` is the curve length, :math:`\ell` is the incremental
        arclength, and :math:`\kappa` is the curvature.

        Args:
            curve: the curve of which the curvature should be computed.
        """
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))

    def J(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)


@deprecated("`MinimumDistance` has been deprecated and will be removed. Please use `CurveCurveDistance` instead.")
class MinimumDistance(CurveCurveDistance):
    pass


class LinkingNumber(Optimizable):

    def __init__(self, curves, downsample=1):
        Optimizable.__init__(self, depends_on=curves)
        self.curves = curves
        for curve in curves:
            assert np.mod(len(curve.quadpoints), downsample) == 0, f"Downsample {downsample} does not divide the number of quadpoints {len(curve.quadpoints)}."

        self.downsample = downsample
        self.dphis = np.array([(c.quadpoints[1] - c.quadpoints[0]) * downsample for c in self.curves])

        r"""
        Compute the Gauss linking number of a set of curves, i.e. whether the curves
        are interlocked or not.

        The value is an integer, >= 1 if the curves are interlocked, 0 if not. For each pair
        of curves, the contribution to the linking number is
        
        .. math::
            Link(c_1, c_2) = \frac{1}{4\pi} \left| \oint_{c_1}\oint_{c_2}\frac{\textbf{r}_1 - \textbf{r}_2}{|\textbf{r}_1 - \textbf{r}_2|^3} (d\textbf{r}_1 \times d\textbf{r}_2) \right|
            
        where :math:`c_1` is the first curve, :math:`c_2` is the second curve,
        :math:`\textbf{r}_1` is the position vector along the first curve, and
        :math:`\textbf{r}_2` is the position vector along the second curve.

        Args:
            curves: the set of curves for which the linking number should be computed.
            downsample: integer factor by which to downsample the quadrature
                points when computing the linking number. Setting this parameter to
                a value larger than 1 will speed up the calculation, which may
                be useful if the set of coils is large, though it may introduce
                inaccuracy if ``downsample`` is set too large.
        """

    def J(self):
        return sopp.compute_linking_number(
            [c.gamma() for c in self.curves],
            [c.gammadash() for c in self.curves],
            self.dphis,
            self.downsample,
        )

    @derivative_dec
    def dJ(self):
        return Derivative({})
    

@jit
def curve_cylinder_distance_pure(g, R0):
    R = jnp.sqrt(g[:,0]**2 + g[:,1]**2)
    npts = g.shape[0]
    return jnp.sqrt(jnp.sum((R-R0)**2)) / npts

@jit
def curve_centroid_cylinder_distance_pure(g, R0):
    c = jnp.mean(g,axis=0)
    Rc = jnp.sqrt(c[0]**2+c[1]**2)
    return Rc-R0

class CurveCylinderDistance(Optimizable):
    def __init__(self, curve, R0, mth='pts'):
        self.curve = curve
        self.R0 = R0
        self.mth = mth

        if mth=='pts':
            self._Jlocal = curve_cylinder_distance_pure
        elif mth=='ctr':
            self._Jlocal = curve_centroid_cylinder_distance_pure
        else:
            raise ValueError('Unknown mth')
        
        self.thisgrad = jit(lambda g, R: grad(self._Jlocal, argnums=0)(g, R))
        super().__init__(depends_on=[curve])

    def J(self):
        return self._Jlocal(self.curve.gamma(), self.R0)
    
    @derivative_dec
    def dJ(self):
        grad0 = self.thisgrad(self.curve.gamma(), self.R0)
        return self.curve.dgamma_by_dcoeff_vjp(grad0)

@jit
def frametwist_pure(n1,n2,b1,b2,b1dash,n2dash):
    dot1 = b1dash[:,0]*n2[:,0] + b1dash[:,1]*n2[:,1] + b1dash[:,2]*n2[:,2]
    dot2 = n2dash[:,0]*b1[:,0] + n2dash[:,1]*b1[:,1] + n2dash[:,2]*b1[:,2]
    dot3 = n1[:,0]*n2[:,0] + n1[:,1]*n2[:,1] + n1[:,2]*n2[:,2]
    dot4 = n1[:,0]*b2[:,0] + n1[:,1]*b2[:,1] + n1[:,2]*b2[:,2]
    size = jnp.size(n1[:,0])
    dphi = 1/(size)
    data0 = jnp.arctan2(-dot4[0],dot3[0])
    data = jnp.zeros((size,))
    integrand = dphi*0.5*(dot1 + dot2)/dot3
    data = data.at[0].set(data0)
    def body_fun(i, val):
        val = val.at[i].set(val[i-1] + (integrand[i-1] + integrand[i]))
        return val
    data = lax.fori_loop(1, len(n1[:,0]), body_fun, data)
    data = data.at[-1].set(data[-2] + (integrand[-1] + integrand[1]))
    return data

@jit
def frametwist_net_pure(frametwist):
    return frametwist[-1] - frametwist[0]

@jit 
def frametwist_range_pure(frametwist):
    return jnp.max(frametwist) - jnp.min(frametwist)

@jit 
def frametwist_max_pure(frametwist):
    return jnp.max(jnp.abs(frametwist))

@jit
def frametwist_lp_pure(frametwist,gammadash,p):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (jnp.mean(frametwist**p * arc_length)/jnp.mean(arc_length))**(1/p)

class FramedCurveTwist(Optimizable):

    def __init__(self, framedcurve, f="lp", p=2):
        r"""
        Computes the maximum relative twist angle between the framedcurve
        and the centroid frame. If the frame is evaluated with respect to 
        the centroid frame, the frame twist is equivalent to the rotation,
        alpha. If not, then the net rotation is evaluated by integrating
        along the curve. The frame rotation can be used within the context
        of HTS strain optimization to avoid 180-degree or 360-degree turns, 
        which can be challenging to wind. 

        Args:
            framedcurve: A FramedCurve from which the twist angle is evaluated
        
        """
        Optimizable.__init__(self, depends_on=[framedcurve])
        assert f in ["net","range","lp","max"]
        self.f = f
        self.p = p 
        self.framedcurve = framedcurve 
        self.framedcurve_centroid = FramedCurveCentroid(framedcurve.curve)
        self.framedcurve_centroid.rotation.fix_all()
        self.frametwist_vjp0 = jit(lambda n1,n2,b1,b2,b1dash,n2dash,v: vjp(
            lambda g: frametwist_pure(g,n2,b1,b2,b1dash,n2dash), n1)[1](v)[0])
        self.frametwist_vjp1 = jit(lambda n1,n2,b1,b2,b1dash,n2dash,v: vjp(
            lambda g: frametwist_pure(n1,g,b1,b2,b1dash,n2dash), n2)[1](v)[0])
        self.frametwist_vjp2 = jit(lambda n1,n2,b1,b2,b1dash,n2dash,v: vjp(
            lambda g: frametwist_pure(n1,n2,g,b2,b1dash,n2dash), b1)[1](v)[0])
        self.frametwist_vjp3 = jit(lambda n1,n2,b1,b2,b1dash,n2dash,v: vjp(
            lambda g: frametwist_pure(n1,n2,b1,g,b1dash,n2dash), b2)[1](v)[0])
        self.frametwist_vjp4 = jit(lambda n1,n2,b1,b2,b1dash,n2dash,v: vjp(
            lambda g: frametwist_pure(n1,n2,b1,b2,g,n2dash), b1dash)[1](v)[0])
        self.frametwist_vjp5 = jit(lambda n1,n2,b1,b2,b1dash,n2dash,v: vjp(
            lambda g: frametwist_pure(n1,n2,b1,b2,b1dash,g), n2dash)[1](v)[0])
        self.range_grad = jit(lambda twist: grad(frametwist_range_pure, argnums=0)(twist))
        self.net_grad = jit(lambda twist: grad(frametwist_net_pure, argnums=0)(twist))
        self.lp_grad0 = jit(lambda twist, gammadash, p: grad(frametwist_lp_pure, argnums=0)(twist,gammadash,p))
        self.lp_grad1 = jit(lambda twist, gammadash, p: grad(frametwist_lp_pure, argnums=1)(twist,gammadash,p))

    def angle_profile(self,endpoint=False):
        """
        Returns the value of alpha
        """
        _, n1, b1 = self.framedcurve.rotated_frame()
        _, n2, b2 = self.framedcurve_centroid.rotated_frame()
        _, n1dash, b1dash = self.framedcurve.rotated_frame_dash()
        _, n2dash, b2dash = self.framedcurve_centroid.rotated_frame_dash()
        if endpoint:
            n1 = np.concatenate((n1,n1[0:1,:]))
            n2 = np.concatenate((n2,n2[0:1,:]))
            b1 = np.concatenate((b1,b1[0:1,:]))
            b2 = np.concatenate((b2,b2[0:1,:]))
            b1dash = np.concatenate((b1dash,b1dash[0:1,:]),axis=0)
            n2dash = np.concatenate((n2dash,n2dash[0:1,:]),axis=0)        
        return frametwist_pure(n1,n2,b1,b2,b1dash,n2dash)

    def J(self,f=None,p=None):
        if f is None:
            f = self.f
        else:
            assert f in ["net","range","lp","max"]
        data = self.angle_profile()
        if (f == "net"):
            return frametwist_net_pure(data)
        elif (f == "range"):
            return frametwist_range_pure(data)
        elif (f == "max"):
            return frametwist_max_pure(data)            
        elif (f == "lp"):
            if p is None:
                p = self.p 
            data = self.angle_profile()
            gammadash = self.framedcurve.curve.gammadash()
            return frametwist_lp_pure(data,gammadash,p)
        else:
            raise Exception('incorrect wrapping function f provided')

    @derivative_dec
    def dJ(self):
        # if (self.f == "net"):
        #     return Derivative({})
        #     # endpoint = True
        #     # data = self.angle_profile(endpoint=endpoint)
        #     # grad0 = self.net_grad(data)
        # elif (self.f == "range"):
        #     return Derivative({})

            # endpoint = False
            # data = self.angle_profile(endpoint=endpoint)
            # grad0 = self.range_grad(data)
        if (self.f == "lp"):
            endpoint = False
            data = self.angle_profile(endpoint=endpoint)
            gammadash = self.framedcurve.curve.gammadash()
            grad0 = self.lp_grad0(data,gammadash,self.p)
            grad1 = self.lp_grad1(data,gammadash,self.p)
        else:
            return Derivative({})
            # raise Exception('incorrect wrapping function f provided')
        _, n1, b1 = self.framedcurve.rotated_frame()
        _, n2, b2 = self.framedcurve_centroid.rotated_frame()
        _, _, b1dash = self.framedcurve.rotated_frame_dash()
        _, n2dash, _ = self.framedcurve_centroid.rotated_frame_dash()

        vjp0 = self.frametwist_vjp0(n1,n2,b1,b2,b1dash,n2dash,grad0)
        vjp1 = self.frametwist_vjp1(n1,n2,b1,b2,b1dash,n2dash,grad0)
        vjp2 = self.frametwist_vjp2(n1,n2,b1,b2,b1dash,n2dash,grad0)
        vjp3 = self.frametwist_vjp3(n1,n2,b1,b2,b1dash,n2dash,grad0)
        vjp4 = self.frametwist_vjp4(n1,n2,b1,b2,b1dash,n2dash,grad0)
        vjp5 = self.frametwist_vjp5(n1,n2,b1,b2,b1dash,n2dash,grad0)
        zero = np.zeros_like(vjp0)

        grad = self.framedcurve.rotated_frame_dcoeff_vjp(zero,vjp0,vjp2) \
            +  self.framedcurve.rotated_frame_dash_dcoeff_vjp(zero,zero,vjp4) \
            +  self.framedcurve_centroid.rotated_frame_dcoeff_vjp(zero,vjp1,vjp3) \
            +  self.framedcurve_centroid.rotated_frame_dash_dcoeff_vjp(zero,vjp5,zero) 
        if (self.f == "lp"):
            grad += self.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1)

        return grad 



def max_distance_pure(g1, g2, dmax, p):
    """
    This returns 0 if all points of g1 have at least one point of g2 at a distance smaller or equal to dmax
    Otherwise, returns the sum of |g2-g1_i|-dmax where only points further than dmax are considered.
    The minimum distance between a point g1_i and g2 is obtained using the p-norm, with p < -1.
    """
    dists = jnp.sqrt(jnp.sum( (g1[:, None, :] - g2[None, :, :])**2, axis=2))

    # Estimate min of dists using p-norm. The minimum is taken along the axis=1. mindists is then an array of length g1.size, where mindists[i]=min_j(|g1[i]-g2[j]|)
    mindists = jnp.sum(dists**p, axis=1)**(1./p)


    # We now evaluate if any of mindists is larger than dmax. If yes, we add the value of (mindists[i]-dmax)**2 to the output. 
    # We normalize by the number of quadrature points along the first curve g1.
    return jnp.sum(jnp.maximum(mindists-dmax, 0)**2) / g1.shape[0]


class MinCurveCurveDistance(Optimizable):
    """
    This class can be used to constrain a curve to remain close
    to another curve.
    """
    def __init__(self, curve1, curve2, maximum_distance, p=-10):
        self.curve1 = curve1
        self.curve2 = curve2
        self.maximum_distance = maximum_distance

        self.J_jax = lambda g1, g2: max_distance_pure(g1, g2, self.maximum_distance, p)
        self.this_grad_0 = jit(lambda g1, g2: grad(self.J_jax, argnums=0)(g1, g2))
        self.this_grad_1 = jit(lambda g1, g2: grad(self.J_jax, argnums=1)(g1, g2))

        Optimizable.__init__(self, depends_on=[curve1, curve2])

    def max_distance(self):
        """
        returns the distance the most isolated point of curve1 to curve2
        """
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()

        # Evaluate all distances
        dists = np.sqrt(np.sum( (g1[:, None, :] - g2[None, :, :])**2, axis=2))

        # Find all min distances
        mindists = np.min(dists, axis=1)

        return np.max(mindists)

    def J(self):
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()

        return self.J_jax( g1, g2 )
    
    @derivative_dec
    def dJ(self):
        g1 = self.curve1.gamma()
        g2 = self.curve2.gamma()

        grad0 = self.this_grad_0(g1, g2)
        grad1 = self.this_grad_1(g1, g2)

        return self.curve1.dgamma_by_dcoeff_vjp( grad0 ) + self.curve2.dgamma_by_dcoeff_vjp( grad1 )
