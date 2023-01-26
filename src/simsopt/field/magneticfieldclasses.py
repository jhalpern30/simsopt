import logging

import numpy as np
from scipy.special import ellipk, ellipe
try:
    from sympy.parsing.sympy_parser import parse_expr
    import sympy as sp
    sympy_found = True
except ImportError:
    sympy_found = False
from pyevtk.hl import pointsToVTK

import simsoptpp as sopp
from .magneticfield import MagneticField
from .._core.json import GSONable, GSONDecoder

logger = logging.getLogger(__name__)

__all__ = ['ToroidalField', 'PoloidalField', 'ScalarPotentialRZMagneticField',
           'CircularCoil', 'Dommaschk', 'Reiman', 'InterpolatedField', 
           'DipoleField', 'WindingVolumeField']


class ToroidalField(MagneticField):
    """
    Magnetic field purely in the toroidal direction, that is, in the phi
    direction with (R,phi,Z) the standard cylindrical coordinates.
    Its modulus is given by B = B0*R0/R where R0 is the first input and B0 the second input to the function.

    Args:
        B0:  modulus of the magnetic field at R0
        R0:  radius of normalization
    """

    def __init__(self, R0, B0):
        MagneticField.__init__(self)
        self.R0 = R0
        self.B0 = B0

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi), R), np.divide(np.cos(phi), R), np.zeros(len(phi)))).T
        B[:] = np.multiply(self.B0*self.R0, phiUnitVectorOverR)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        phi = np.arctan2(points[:, 1], points[:, 0])
        R = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))

        x = points[:, 0]
        y = points[:, 1]

        dB_by_dX1 = np.vstack((
            np.multiply(np.divide(self.B0*self.R0, R**4), 2*np.multiply(x, y)),
            np.multiply(np.divide(self.B0*self.R0, R**4), y**2-x**2),
            0*R))
        dB_by_dX2 = np.vstack((
            np.multiply(np.divide(self.B0*self.R0, R**4), y**2-x**2),
            np.multiply(np.divide(self.B0*self.R0, R**4), -2*np.multiply(x, y)),
            0*R))
        dB_by_dX3 = np.vstack((0*R, 0*R, 0*R))

        dB[:] = np.array([dB_by_dX1, dB_by_dX2, dB_by_dX3]).T

    def _d2B_by_dXdX_impl(self, ddB):
        points = self.get_points_cart_ref()
        x = points[:, 0]
        y = points[:, 1]
        ddB[:] = 2*self.B0*self.R0*np.multiply(
            1/(points[:, 0]**2+points[:, 1]**2)**3, np.array([
                [[3*points[:, 0]**2+points[:, 1]**3, points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, np.zeros((len(points)))], [
                    points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                    np.zeros((len(points)))],
                 np.zeros((3, len(points)))],
                [[points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                  np.zeros((len(points)))],
                 [3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2,
                  np.zeros((len(points)))], np.zeros((3, len(points)))],
                np.zeros((3, 3, len(points)))])).T

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        A[:] = self.B0*self.R0*np.array([
            points[:, 2]*points[:, 0]/(points[:, 0]**2+points[:, 1]**2),
            points[:, 2]*points[:, 1]/(points[:, 0]**2+points[:, 1]**2),
            0*points[:, 2]]).T

    def _dA_by_dX_impl(self, dA):
        points = self.get_points_cart_ref()
        dA[:] = self.B0*self.R0*np.array((points[:, 2]/(points[:, 0]**2+points[:, 1]**2)**2)*np.array(
            [[-points[:, 0]**2+points[:, 1]**2, -2*points[:, 0]*points[:, 1], np.zeros((len(points)))],
             [-2*points[:, 0]*points[:, 1], points[:, 0]**2-points[:, 1]**2, np.zeros((len(points)))],
             [points[:, 0]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
              points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2], np.zeros((len(points)))]])).T

    def _d2A_by_dXdX_impl(self, ddA):
        points = self.get_points_cart_ref()
        ddA[:] = 2*self.B0*self.R0*np.array(
            (points[:, 2]/(points[:, 0]**2+points[:, 1]**2)**3)*np.array([
                [[points[:, 0]**3-3*points[:, 0]*points[:, 1]**2, 3*points[:, 0]**2*points[:, 1]-points[:, 1]**3,
                  (-points[:, 0]**4+points[:, 1]**4)/(2*points[:, 2])],
                 [3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2, -points[:, 0]*points[:, 1]*(
                     points[:, 0]**2+points[:, 1]**2)/points[:, 2]],
                 [(-points[:, 0]**4+points[:, 1]**4)/(2*points[:, 2]),
                  -points[:, 0]*points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
                  np.zeros((len(points)))]],
                [[3*points[:, 0]**2*points[:, 1]-points[:, 1]**3, -points[:, 0]**3+3*points[:, 0]*points[:, 1]**2,
                  -points[:, 0]*points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2]],
                 [-points[:, 0]**3+3*points[:, 0]*points[:, 1]**2, -3*points[:, 0]**2*points[:, 1]+points[:, 1]**3, (
                     points[:, 0]**4-points[:, 1]**4)/(2*points[:, 2])],
                 [-points[:, 0]*points[:, 1]*(points[:, 0]**2+points[:, 1]**2)/points[:, 2],
                  (points[:, 0]**4-points[:, 1]**4)/(2*points[:, 2]), np.zeros((len(points)))]],
                np.zeros((3, 3, len(points)))])).transpose((3, 0, 1, 2))

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["R0"], d["B0"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class PoloidalField(MagneticField):
    '''
    Magnetic field purely in the poloidal direction, that is, in the
    theta direction of a poloidal-toroidal coordinate system.  Its
    modulus is given by B = B0 * r / (R0 * q) so that, together with
    the toroidal field, it creates a safety factor equals to q

    Args:
        B0: modulus of the magnetic field at R0
        R0: major radius of the magnetic axis
        q: safety factor/pitch angle of the magnetic field lines
    '''

    def __init__(self, R0, B0, q):
        MagneticField.__init__(self)
        self.R0 = R0
        self.B0 = B0
        self.q = q

    def _B_impl(self, B):
        points = self.get_points_cart_ref()

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        phi = np.arctan2(y, x)
        theta = np.arctan2(z, np.sqrt(x**2+y**2)-self.R0)
        r = np.sqrt((np.sqrt(x**2+y**2)-self.R0)**2+z**2)
        thetaUnitVectorOver_times_r = np.vstack((-np.multiply(np.sin(theta), r)*np.cos(phi), -np.multiply(np.sin(theta), r)*np.sin(phi), np.multiply(np.cos(theta), r))).T
        B[:] = self.B0/self.R0/self.q*thetaUnitVectorOver_times_r

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        phi = np.arctan2(y, x)
        theta = np.arctan2(z, np.sqrt(x**2+y**2)-self.R0)
        r = np.sqrt((np.sqrt(x**2+y**2)-self.R0)**2+z**2)

        dtheta_by_dX1 = -((x*z)/(np.sqrt(x**2+y**2)*(x**2+y**2+z**2-2*np.sqrt(x**2+y**2)*self.R0+(self.R0)**2)))
        dtheta_by_dX2 = -((y*z)/(np.sqrt(x**2+y**2)*(x**2+y**2+z**2-2*np.sqrt(x**2+y**2)*self.R0+(self.R0)**2)))
        dtheta_by_dX3 = 1/((-self.R0+np.sqrt(x**2+y**2))*(1+z**2/(self.R0-np.sqrt(x**2+y**2))**2))

        dphi_by_dX1 = -(y/(x**2 + y**2))
        dphi_by_dX2 = x/(x**2 + y**2)
        dphi_by_dX3 = 0.*z

        dthetaunitvector_by_dX1 = np.vstack((
            -np.cos(theta)*np.cos(phi)*dtheta_by_dX1+np.sin(theta)*np.sin(phi)*dphi_by_dX1,
            -np.cos(theta)*np.sin(phi)*dtheta_by_dX1-np.sin(theta)*np.cos(phi)*dphi_by_dX1,
            -np.sin(theta)*dtheta_by_dX1
        )).T
        dthetaunitvector_by_dX2 = np.vstack((
            -np.cos(theta)*np.cos(phi)*dtheta_by_dX2+np.sin(theta)*np.sin(phi)*dphi_by_dX2,
            -np.cos(theta)*np.sin(phi)*dtheta_by_dX2-np.sin(theta)*np.cos(phi)*dphi_by_dX2,
            -np.sin(theta)*dtheta_by_dX2
        )).T
        dthetaunitvector_by_dX3 = np.vstack((
            -np.cos(theta)*np.cos(phi)*dtheta_by_dX3+np.sin(theta)*np.sin(phi)*dphi_by_dX3,
            -np.cos(theta)*np.sin(phi)*dtheta_by_dX3-np.sin(theta)*np.cos(phi)*dphi_by_dX3,
            -np.sin(theta)*dtheta_by_dX3
        )).T

        dB_by_dX1_term1 = np.multiply(dthetaunitvector_by_dX1.T, r)
        dB_by_dX2_term1 = np.multiply(dthetaunitvector_by_dX2.T, r)
        dB_by_dX3_term1 = np.multiply(dthetaunitvector_by_dX3.T, r)

        thetaUnitVector_1 = -np.sin(theta)*np.cos(phi)
        thetaUnitVector_2 = -np.sin(theta)*np.sin(phi)
        thetaUnitVector_3 = np.cos(theta)

        dr_by_dX1 = (x*(-self.R0+np.sqrt(x**2+y**2)))/(np.sqrt(x**2+y**2)*np.sqrt((self.R0-np.sqrt(x**2+y**2))**2+z**2))
        dr_by_dX2 = (y*(-self.R0+np.sqrt(x**2+y**2)))/(np.sqrt(x**2+y**2)*np.sqrt((self.R0-np.sqrt(x**2+y**2))**2+z**2))
        dr_by_dX3 = z/np.sqrt((self.R0-np.sqrt(x**2+y**2))**2+z**2)

        dB_by_dX1_term2 = np.vstack((
            thetaUnitVector_1*dr_by_dX1,
            thetaUnitVector_2*dr_by_dX1,
            thetaUnitVector_3*dr_by_dX1))
        dB_by_dX2_term2 = np.vstack((
            thetaUnitVector_1*dr_by_dX2,
            thetaUnitVector_2*dr_by_dX2,
            thetaUnitVector_3*dr_by_dX2))
        dB_by_dX3_term2 = np.vstack((
            thetaUnitVector_1*dr_by_dX3,
            thetaUnitVector_2*dr_by_dX3,
            thetaUnitVector_3*dr_by_dX3))

        dB[:] = self.B0/self.R0/self.q*np.array([dB_by_dX1_term1+dB_by_dX1_term2, dB_by_dX2_term1+dB_by_dX2_term2, dB_by_dX3_term1+dB_by_dX3_term2]).T

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d):
        field = cls(d["R0"], d["B0"], d["q"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class ScalarPotentialRZMagneticField(MagneticField):
    """
    Vacuum magnetic field as a solution of B = grad(Phi) where Phi is the
    magnetic field scalar potential.  It takes Phi as an input string, which
    should contain an expression involving the standard cylindrical coordinates
    (R, phi, Z) Example: ScalarPotentialRZMagneticField("2*phi") yields a
    magnetic field B = grad(2*phi) = (0,2/R,0). In order for the analytical
    derivatives to be performed by sympy, a term 1e-30*Phi*R*Z is added
    to every entry. Note: this function needs sympy.

    Args:
        phi_str:  string containing vacuum scalar potential expression as a function of R, Z and phi
    """

    ## TRY to add C*R*phi*Z in all entries and then put C=0

    def __init__(self, phi_str):
        MagneticField.__init__(self)
        if not sympy_found:
            raise RuntimeError("Sympy is required for the ScalarPotentialRZMagneticField class")
        self.phi_str = phi_str
        self.phi_parsed = parse_expr(phi_str)
        R, Z, Phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.phi_parsed.diff(R)+1e-30*Phi*R*Z,\
                                                   self.phi_parsed.diff(Phi)/R+1e-30*Phi*R*Z,\
                                                   self.phi_parsed.diff(Z)+1e-30*Phi*R*Z])
        self.dBlambdify_by_dX = sp.lambdify(
            (R, Z, Phi),
            [[1e-30*Phi*R*Z+sp.cos(Phi)*self.phi_parsed.diff(R).diff(R)-(sp.sin(Phi)/R)*self.phi_parsed.diff(R).diff(Phi),
              1e-30*Phi*R*Z+sp.cos(Phi)*(self.phi_parsed.diff(Phi)/R).diff(R)-(sp.sin(Phi)/R)*(self.phi_parsed.diff(Phi)/R).diff(Phi),
              1e-30*Phi*R*Z+sp.cos(Phi)*self.phi_parsed.diff(Z).diff(R)-(sp.sin(Phi)/R)*self.phi_parsed.diff(Z).diff(Phi)],
             [1e-30*Phi*R*Z+sp.sin(Phi)*self.phi_parsed.diff(R).diff(R)+(sp.cos(Phi)/R)*self.phi_parsed.diff(R).diff(Phi),
              1e-30*Phi*R*Z+sp.sin(Phi)*(self.phi_parsed.diff(Phi)/R).diff(R)+(sp.cos(Phi)/R)*(self.phi_parsed.diff(Phi)/R).diff(Phi),
              1e-30*Phi*R*Z+sp.sin(Phi)*self.phi_parsed.diff(Z).diff(R)+(sp.cos(Phi)/R)*self.phi_parsed.diff(Z).diff(Phi)],
             [1e-30*Phi*R*Z+self.phi_parsed.diff(R).diff(Z),
              1e-30*Phi*R*Z+(self.phi_parsed.diff(Phi)/R).diff(Z),
              1e-30*Phi*R*Z+self.phi_parsed.diff(Z).diff(Z)]])

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        B_cyl = np.array(self.Blambdify(r, z, phi)).T
        # Bx = Br cos(phi) - Bphi sin(phi)
        B[:, 0] = B_cyl[:, 0] * np.cos(phi) - B_cyl[:, 1] * np.sin(phi)
        # By = Br sin(phi) + Bphi cos(phi)
        B[:, 1] = B_cyl[:, 0] * np.sin(phi) + B_cyl[:, 1] * np.cos(phi)
        B[:, 2] = B_cyl[:, 2]

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        z = points[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        dB_cyl = np.array(self.dBlambdify_by_dX(r, z, phi)).transpose((2, 0, 1))
        dBrdx = dB_cyl[:, 0, 0]
        dBrdy = dB_cyl[:, 1, 0]
        dBrdz = dB_cyl[:, 2, 0]
        dBphidx = dB_cyl[:, 0, 1]
        dBphidy = dB_cyl[:, 1, 1]
        dBphidz = dB_cyl[:, 2, 1]
        dB[:, 0, 2] = dB_cyl[:, 0, 2]
        dB[:, 1, 2] = dB_cyl[:, 1, 2]
        dB[:, 2, 2] = dB_cyl[:, 2, 2]
        dcosphidx = -points[:, 0]**2/r**3 + 1/r
        dsinphidx = -points[:, 0]*points[:, 1]/r**3
        dcosphidy = -points[:, 0]*points[:, 1]/r**3
        dsinphidy = -points[:, 1]**2/r**3 + 1/r
        B_cyl = np.array(self.Blambdify(r, z, phi)).T
        Br = B_cyl[:, 0]
        Bphi = B_cyl[:, 1]
        # Bx = Br cos(phi) - Bphi sin(phi)
        dB[:, 0, 0] = dBrdx * np.cos(phi) + Br * dcosphidx - dBphidx * np.sin(phi) \
            - Bphi * dsinphidx
        dB[:, 1, 0] = dBrdy * np.cos(phi) + Br * dcosphidy - dBphidy * np.sin(phi) \
            - Bphi * dsinphidy
        dB[:, 2, 0] = dBrdz * np.cos(phi) - dBphidz * np.sin(phi)
        # By = Br sin(phi) + Bphi cos(phi)
        dB[:, 0, 1] = dBrdx * np.sin(phi) + Br * dsinphidx + dBphidx * np.cos(phi) \
            + Bphi * dcosphidx
        dB[:, 1, 1] = dBrdy * np.sin(phi) + Br * dsinphidy + dBphidy * np.cos(phi) \
            + Bphi * dcosphidy
        dB[:, 2, 1] = dBrdz * np.sin(phi) + dBphidz * np.cos(phi)

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["phi_str"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class CircularCoil(MagneticField):
    '''
    Magnetic field created by a single circular coil evaluated using analytical
    functions, including complete elliptic integrals of the first and second
    kind.  As inputs, it takes the radius of the coil (r0), its center, current
    (I) and its normal vector [either spherical angle components
    (normal=[theta,phi]) or (x,y,z) components of a vector (normal=[x,y,z])]).
    The (theta,phi) angles are related to the (x,y,z) components of the normal vector via
    theta = np.arctan2(normal[1], normal[0]) and phi = np.arctan2(np.sqrt(normal[0]**2+normal[1]**2), normal[2]).
    Sign convention: CircularCoil with a positive current produces a magnetic field
    vector in the same direction as the normal when evaluated at the center of the coil.a

    Args:
        r0: radius of the coil
        center: point at the coil center
        I: current of the coil in Ampere's
        normal: if list with two values treats it as spherical angles theta and
                phi of the normal vector to the plane of the coil centered at the coil
                center, if list with three values treats it a vector
    '''

    def __init__(self, r0=0.1, center=[0, 0, 0], I=5e5/np.pi, normal=[0, 0]):
        MagneticField.__init__(self)
        self.r0 = r0
        self.Inorm = I*4e-7
        self.center = center
        self.normal = normal
        if len(normal) == 2:
            theta = normal[0]
            phi = normal[1]
        else:
            theta = np.arctan2(normal[1], normal[0])
            phi = np.arctan2(np.sqrt(normal[0]**2+normal[1]**2), normal[2])

        self.rotMatrix = np.array([
            [np.cos(phi) * np.cos(theta)**2 + np.sin(theta)**2,
             -np.sin(phi / 2)**2 * np.sin(2 * theta),
             np.cos(theta) * np.sin(phi)],
            [-np.sin(phi / 2)**2 * np.sin(2 * theta),
             np.cos(theta)**2 + np.cos(phi) * np.sin(theta)**2,
             np.sin(phi) * np.sin(theta)],
            [-np.cos(theta) * np.sin(phi),
             -np.sin(phi) * np.sin(theta),
             np.cos(phi)]
        ])

        self.rotMatrixInv = np.array(self.rotMatrix.T)

    @property
    def I(self):
        return self.Inorm * 25e5

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self.rotMatrixInv, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        gamma = np.square(points[:, 0]) - np.square(points[:, 1])
        B[:] = np.dot(self.rotMatrix, np.array(
            [self.Inorm*points[:, 0]*points[:, 2]/(2*alpha**2*beta*rho**2+1e-31)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm*points[:, 1]*points[:, 2]/(2*alpha**2*beta*rho**2+1e-31)*((self.r0**2+r**2)*ellipek2-alpha**2*ellipkk2),
             self.Inorm/(2*alpha**2*beta+1e-31)*((self.r0**2-r**2)*ellipek2+alpha**2*ellipkk2)])).T

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self.rotMatrixInv, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)
        gamma = np.square(points[:, 0]) - np.square(points[:, 1])
        dBxdx = (self.Inorm*points[:, 2]*(
            ellipkk2*alpha**2*((2*points[:, 0]**4 + gamma*(
                points[:, 1]**2 + points[:, 2]**2))*r**2 + self.r0**2*(
                    gamma*(self.r0**2 + 2*points[:, 2]**2) - (3*points[:, 0]**2 - 2*points[:, 1]**2)*rho**2))
            + ellipek2*(-((2*points[:, 0]**4 + gamma*(points[:, 1]**2 + points[:, 2]**2))*r**4)
                        + self.r0**4*(-(gamma*(self.r0**2 + 3*points[:, 2]**2)) + (8*points[:, 0]**2 - points[:, 1]**2)*rho**2)
                        - self.r0**2*(
                            3*gamma*points[:, 2]**4 - 2*(2*points[:, 0]**2 + points[:, 1]**2)*points[:, 2]**2 * rho**2
                            + (5*points[:, 0]**2 + points[:, 1]**2)*rho**4
            ))
        ))/(2*alpha**4*beta**3*rho**4+1e-31)

        dBydx = (self.Inorm*points[:, 0]*points[:, 1]*points[:, 2]*(
            ellipkk2*alpha**2*(
                2*self.r0**4 + r**2*(2*r**2 + rho**2) - self.r0**2*(-4*points[:, 2]**2 + 5*rho**2))
            + ellipek2*(-2*self.r0**6 - r**4*(2*r**2 + rho**2) + 3*self.r0**4*(-2*points[:, 2]**2 + 3*rho**2) - 2*self.r0**2*(3*points[:, 2]**4 - points[:, 2]**2*rho**2 + 2*rho**4))
        ))/(2*alpha**4*beta**3*rho**4+1e-31)

        dBzdx = (self.Inorm*points[:, 0]*(
            - (ellipkk2*alpha**2*((-self.r0**2 + rho**2)**2 + points[:, 2]**2*(self.r0**2 + rho**2)))
            + ellipek2*(
                points[:, 2]**4*(self.r0**2 + rho**2) + (-self.r0**2 + rho**2)**2*(self.r0**2 + rho**2)
                + 2*points[:, 2]**2*(self.r0**4 - 6*self.r0**2*rho**2 + rho**4))
        ))/(2*alpha**4*beta**3*rho**2+1e-31)
        dBxdy = dBydx

        dBydy = (self.Inorm*points[:, 2]*(
            ellipkk2*alpha**2*((2*points[:, 1]**4 - gamma*(points[:, 0]**2 + points[:, 2]**2))*r**2 +
                               self.r0**2*(-(gamma*(self.r0**2 + 2*points[:, 2]**2)) - (-2*points[:, 0]**2 + 3*points[:, 1]**2)*rho**2)) +
            ellipek2*(-((2*points[:, 1]**4 - gamma*(points[:, 0]**2 + points[:, 2]**2))*r**4) +
                      self.r0**4*(gamma*(self.r0**2 + 3*points[:, 2]**2) + (-points[:, 0]**2 + 8*points[:, 1]**2)*rho**2) -
                      self.r0**2*(-3*gamma*points[:, 2]**4 - 2*(points[:, 0]**2 + 2*points[:, 1]**2)*points[:, 2]**2*rho**2 +
                                  (points[:, 0]**2 + 5*points[:, 1]**2)*rho**4))))/(2*alpha**4*beta**3*rho**4+1e-31)

        dBzdy = dBzdx*points[:, 1]/(points[:, 0]+1e-31)

        dBxdz = dBzdx

        dBydz = dBzdy

        dBzdz = (self.Inorm*points[:, 2]*(ellipkk2*alpha**2*(self.r0**2 - r**2) +
                                          ellipek2*(-7*self.r0**4 + r**4 + 6*self.r0**2*(-points[:, 2]**2 + rho**2))))/(2*alpha**4*beta**3+1e-31)

        dB_by_dXm = np.array([
            [dBxdx, dBydx, dBzdx],
            [dBxdy, dBydy, dBzdy],
            [dBxdz, dBydz, dBzdz]])

        dB[:] = np.array([
            [np.dot(self.rotMatrixInv[:, 0], np.dot(self.rotMatrix[0, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 1], np.dot(self.rotMatrix[0, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 2], np.dot(self.rotMatrix[0, :], dB_by_dXm))],
            [np.dot(self.rotMatrixInv[:, 0], np.dot(self.rotMatrix[1, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 1], np.dot(self.rotMatrix[1, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 2], np.dot(self.rotMatrix[1, :], dB_by_dXm))],
            [np.dot(self.rotMatrixInv[:, 0], np.dot(self.rotMatrix[2, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 1], np.dot(self.rotMatrix[2, :], dB_by_dXm)),
             np.dot(self.rotMatrixInv[:, 2], np.dot(self.rotMatrix[2, :], dB_by_dXm))]]).T

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        points = np.array(np.dot(self.rotMatrixInv, np.array(np.subtract(points, self.center)).T).T)
        rho = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        r = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
        alpha = np.sqrt(self.r0**2 + np.square(r) - 2*self.r0*rho)
        beta = np.sqrt(self.r0**2 + np.square(r) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.square(alpha), np.square(beta)))
        ellipek2 = ellipe(k**2)
        ellipkk2 = ellipk(k**2)

        num = (2*self.r0+np.sqrt(points[:, 0]**2+points[:, 1]**2)*ellipek2+(self.r0**2+points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)*(ellipe(k**2)-ellipkk2))
        denom = ((points[:, 0]**2+points[:, 1]**2+1e-31)*np.sqrt(self.r0**2+points[:, 0]**2+points[:, 1]**2+2*self.r0*np.sqrt(points[:, 0]**2+points[:, 1]**2)+points[:, 2]**2+1e-31))
        fak = num/denom
        pts = fak[:, None]*np.concatenate((-points[:, 1][:, None], points[:, 0][:, None], np.zeros((points.shape[0], 1))), axis=-1)
        A[:] = -self.Inorm/2*np.dot(self.rotMatrix, pts.T).T

    def as_dict(self, serial_objs_dict):
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["r0"], d["center"], d["I"], d["normal"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class WindingVolumeField(MagneticField):
    r"""
    Computes the MagneticField induced by N grid cells, each with spatially varying and
    locally divergence-free current, for the Winding Volume method. This is done by
    calling the normal BiotSavart field for each of the cells, and summing the result.

    Args:
        winding_volume: WindingVolume grid class, containing the grid cell locations,
                        integration points in each cell, etc.
    """

    def __init__(self, winding_volume):
        MagneticField.__init__(self)
        self.winding_volume = winding_volume
        self.s = winding_volume.plasma_boundary
        self.integration_points = winding_volume.XYZ_integration
        self.N_grid = winding_volume.N_grid
        self.Phi = winding_volume.Phi
        self.grid_scaling = winding_volume.dx * winding_volume.dy * winding_volume.dz / (winding_volume.nx * winding_volume.ny * winding_volume.nz)
        # self.Phi_full = winding_volume.Phi_full
        # self.integration_points_full = winding_volume.integration_points_full
        self.nsym = winding_volume.Phi_full.shape[1] // self.N_grid
        alphas_full = np.zeros((self.N_grid * self.nsym, self.Phi.shape[0]))
        for i in range(self.nsym):
            alphas_full[i * self.N_grid:(i + 1) * self.N_grid, :] = winding_volume.alphas.reshape(self.N_grid, self.Phi.shape[0])
        self.alphas_full = alphas_full

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        Jx = self.winding_volume.J[:, :, 0]
        Jy = self.winding_volume.J[:, :, 1]
        Jz = self.winding_volume.J[:, :, 2]
        J_temp = np.zeros(self.winding_volume.J.shape)
        contig = np.ascontiguousarray
        if self.winding_volume.range == 'full torus':
            stell_list = [1]
            nfp = 1
        else:
            stell_list = [1, -1]
            nfp = 2
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                ox_temp = self.integration_points[:, :, 0] * np.cos(phi0) - self.integration_points[:, :, 1] * np.sin(phi0) * stell
                oy_temp = self.integration_points[:, :, 0] * np.sin(phi0) + self.integration_points[:, :, 1] * np.cos(phi0) * stell
                oz_temp = self.integration_points[:, :, 2] * stell

                # get new dipole vectors by flipping the x component, then rotating by phi0
                J_temp[:, :, 0] = Jx * np.cos(phi0) * stell - Jy * np.sin(phi0)
                J_temp[:, :, 1] = Jx * np.sin(phi0) * stell + Jy * np.cos(phi0)
                J_temp[:, :, 2] = Jz

                int_points = np.transpose(np.array([ox_temp, oy_temp, oz_temp]), [1, 2, 0])
                B[:] += sopp.winding_volume_field_B(
                    contig(points), 
                    contig(int_points), 
                    contig(J_temp)
                ) * self.grid_scaling

        # function returns factor of shape (num_points, 3, num_cells, n_functions)
        # factor = sopp.winding_volume_field_Bext(points, self.winding_volume.integration_points_full, self.winding_volume.Phi_full) * self.grid_scaling
        # B[:] = np.tensordot(factor, self.alphas_full, ((2, 3), (0, 1)))

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = sopp.winding_volume_field_B(points, self.integration_points, self.J)

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        A[:] = sopp.winding_volume_field_B(points, self.integration_points, self.J)

    def _dA_by_dX_impl(self, dA):
        points = self.get_points_cart_ref()
        dA[:] = sopp.winding_volume_field_B(points, self.integration_points, self.J)

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["winding_volume"] = self.winding_volume
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["winding_volume"])


class DipoleField(MagneticField):
    r"""
    Computes the MagneticField induced by N dipoles. This is very simple but needs to be
    a type MagneticField class for using the other simsopt functionality.
    The field is given by

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{i=1}^{N} (\frac{3\mathbf{r}_i\cdot \mathbf{m}_i}{|\mathbf{r}_i|^5}\mathbf{r}_i - \frac{\mathbf{m}_i}{|\mathbf{r}_i|^3})

    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant and :math:\mathbf{r_i} = \mathbf{x} - \mathbf{x}^{dipole}_i is the vector between the field evaluation point and the dipole i position.

    Args:
        pm_opt: A PermanentMagnetOptimizer object that has already been optimized.
        m: Solution for the dipoles using the pm_opt object.
    """

    def __init__(self, pm_opt=None):
        MagneticField.__init__(self)
        self._dipole_fields_from_symmetries(pm_opt)
        # reformat memory for c++ routines
        self.dipole_grid = np.ascontiguousarray(self.dipole_grid)
        self.m_vec = np.ascontiguousarray(self.m_vec)
        self.pm_opt = pm_opt

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = sopp.dipole_field_B(points, self.dipole_grid, self.m_vec)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = sopp.dipole_field_dB(points, self.dipole_grid, self.m_vec)

    def _A_impl(self, A):
        points = self.get_points_cart_ref()
        A[:] = sopp.dipole_field_A(points, self.dipole_grid, self.m_vec)

    def _dA_by_dX_impl(self, dA):
        points = self.get_points_cart_ref()
        dA[:] = sopp.dipole_field_dA(points, self.dipole_grid, self.m_vec)

    def _dipole_fields_from_symmetries(self, pm_opt):
        """
        Takes the dipoles and grid initialized in a PermanentMagnetOptimizer (for a half-period surface)
        and generates the full dipole manifold so that the call to B() (the magnetic field from
        the dipoles) correctly returns contributions from all the dipoles from symmetries.

        Args:
            pm_opt (PermanentMagnetOptimizer): Permanent magnet optimizer class object with
                                               a set of initialized dipoles + dipole grid.
        """
        # Read in the required fields from pm_opt object
        nfp = pm_opt.plasma_boundary.nfp
        stellsym = pm_opt.plasma_boundary.stellsym
        dipole_grid = pm_opt.dipole_grid_xyz
        ndipoles = pm_opt.ndipoles
        if stellsym:
            stell_list = [1, -1]
            nsym = nfp * 2
        else:
            stell_list = [1]
            nsym = nfp
        m = pm_opt.m.reshape(ndipoles, 3)

        # Initialize new grid and dipole vectors for all the dipoles
        # after we account for the symmetries below.
        dipole_grid_x = np.zeros(ndipoles * nsym)
        dipole_grid_y = np.zeros(ndipoles * nsym)
        dipole_grid_z = np.zeros(ndipoles * nsym)
        m_vec = np.zeros((ndipoles * nsym, 3))
        m_maxima = np.zeros(ndipoles * nsym)

        # Load in the dipole locations for a half-period surface
        ox = pm_opt.dipole_grid_xyz[:, 0]
        oy = pm_opt.dipole_grid_xyz[:, 1]
        oz = pm_opt.dipole_grid_xyz[:, 2]

        # loop through the dipoles and repeat for fp and stellarator symmetries
        index = 0
        n = ndipoles

        # get the components in Cartesian, converting if needed
        mmx = m[:, 0]
        mmy = m[:, 1]
        mmz = m[:, 2]
        if pm_opt.coordinate_flag == 'cylindrical':
            phi_dipole = np.arctan2(oy, ox)
            mmx_temp = mmx * np.cos(phi_dipole) - mmy * np.sin(phi_dipole)
            mmy_temp = mmx * np.sin(phi_dipole) + mmy * np.cos(phi_dipole)
            mmx = mmx_temp
            mmy = mmy_temp
        if pm_opt.coordinate_flag == 'toroidal':
            phi_dipole = np.arctan2(oy, ox)
            theta_dipole = np.arctan2(oz, np.sqrt(ox ** 2 + oy ** 2) - pm_opt.R0)
            mmx_temp = mmx * np.cos(phi_dipole) * np.cos(theta_dipole) - mmy * np.sin(phi_dipole) - mmz * np.cos(phi_dipole) * np.sin(theta_dipole)
            mmy_temp = mmx * np.sin(phi_dipole) * np.cos(theta_dipole) + mmy * np.cos(phi_dipole) - mmz * np.sin(phi_dipole) * np.sin(theta_dipole)
            mmz_temp = mmx * np.sin(theta_dipole) + mmz * np.cos(theta_dipole)
            mmx = mmx_temp
            mmy = mmy_temp
            mmz = mmz_temp

        # Loop over stellarator and field-period symmetry contributions
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                dipole_grid_x[index:index + n] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                dipole_grid_y[index:index + n] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                dipole_grid_z[index:index + n] = oz * stell

                # get new dipole vectors by flipping the x component, then rotating by phi0
                m_vec[index:index + n, 0] = mmx * np.cos(phi0) * stell - mmy * np.sin(phi0)
                m_vec[index:index + n, 1] = mmx * np.sin(phi0) * stell + mmy * np.cos(phi0)
                m_vec[index:index + n, 2] = mmz

                m_maxima[index:index + n] = pm_opt.m_maxima
                index += n

        self.dipole_grid = np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).T
        self.m_vec = m_vec
        self.m_maxima = m_maxima

    def _toVTK(self, vtkname):
        """
            Write dipole data into a VTK file (stolen from Caoxiang's CoilPy code).

        Args:
            vtkname (str): VTK filename, will be appended with .vts or .vtu.
            dim (tuple, optional): Dimension information if saved as structured grids. Defaults to (1).
        """

        # get the coordinates
        ox = np.ascontiguousarray(self.dipole_grid[:, 0])
        oy = np.ascontiguousarray(self.dipole_grid[:, 1])
        oz = np.ascontiguousarray(self.dipole_grid[:, 2])
        ophi = np.arctan2(oy, ox)
        otheta = np.arctan2(oz, np.sqrt(ox ** 2 + oy ** 2) - self.pm_opt.R0)

        # define the m vectors and the normalized m vectors
        # in Cartesian, cylindrical, and simple toroidal coordinates.
        mx = np.ascontiguousarray(self.m_vec[:, 0])
        my = np.ascontiguousarray(self.m_vec[:, 1])
        mz = np.ascontiguousarray(self.m_vec[:, 2])
        mmag = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        mx_normalized = np.ascontiguousarray(mx / self.m_maxima)
        my_normalized = np.ascontiguousarray(my / self.m_maxima)
        mz_normalized = np.ascontiguousarray(mz / self.m_maxima)
        mr = np.ascontiguousarray(mx * np.cos(ophi) + my * np.sin(ophi))
        mrminor = np.ascontiguousarray(mx * np.cos(ophi) * np.cos(otheta) + my * np.sin(ophi) * np.cos(otheta) + np.sin(otheta) * mz)
        mphi = np.ascontiguousarray(-mx * np.sin(ophi) + my * np.cos(ophi))
        mtheta = np.ascontiguousarray(-mx * np.cos(ophi) * np.sin(otheta) - my * np.sin(ophi) * np.sin(otheta) + np.cos(otheta) * mz)
        mr_normalized = np.ascontiguousarray(mr / self.m_maxima)
        mrminor_normalized = np.ascontiguousarray(mrminor / self.m_maxima)
        mphi_normalized = np.ascontiguousarray(mphi / self.m_maxima)
        mtheta_normalized = np.ascontiguousarray(mtheta / self.m_maxima)

        # Save all the data to a vtk file which can be visualized nicely with ParaView
        data = {"m": (mx, my, mz), "m_normalized": (mx_normalized, my_normalized, mz_normalized), "m_rphiz": (mr, mphi, mz), "m_rphiz_normalized": (mr_normalized, mphi_normalized, mz_normalized), "m_rphitheta": (mrminor, mphi, mtheta), "m_rphitheta_normalized": (mrminor_normalized, mphi_normalized, mtheta_normalized)}
        pointsToVTK(
            vtkname, ox, oy, oz, data=data
        )

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["pm_opt"] = self.pm_opt
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["pm_opt"])


class Dommaschk(MagneticField):
    """
    Vacuum magnetic field created by an explicit representation of the magnetic
    field scalar potential as proposed by W. Dommaschk (1986), Computer Physics
    Communications 40, 203-218. As inputs, it takes the arrays for the harmonics
    m, n and its corresponding coefficients.

    Args:
        m: first harmonic array
        n: second harmonic array
        coeffs: coefficient for Vml for each of the ith index of the harmonics m and n
    """

    def __init__(self, mn=[[0, 0]], coeffs=[[0, 0]]):
        MagneticField.__init__(self)
        self.m = np.array(mn, dtype=np.int16)[:, 0]
        self.n = np.array(mn, dtype=np.int16)[:, 1]
        self.coeffs = coeffs
        self.Btor = ToroidalField(1, 1)

    def _set_points_cb(self):
        self.Btor.set_points_cart(self.get_points_cart_ref())

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = np.add.reduce(sopp.DommaschkB(self.m, self.n, self.coeffs, points))+self.Btor.B()

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = np.add.reduce(sopp.DommaschkdB(self.m, self.n, self.coeffs, points))+self.Btor.dB_by_dX()

    @property
    def mn(self):
        return np.column_stack((self.m, self.n))

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        mn = decoder.process_decoded(d["mn"], serial_objs_dict, recon_objs)
        field = cls(mn, d["coeffs"])
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class Reiman(MagneticField):
    '''
    Magnetic field model in section 5 of Reiman and Greenside, Computer Physics Communications 43 (1986) 157—167.
    This field allows for an analytical expression of the magnetic island width
    that can be used for island optimization.  However, the field is not
    completely physical as it does not have nested flux surfaces.

    Args:
        iota0: unperturbed rotational transform
        iota1: unperturbed global magnetic shear
        k: integer array specifying the Fourier modes used
        epsilonk: coefficient of the Fourier modes
        m0: toroidal symmetry parameter (normally m0=1)
    '''

    def __init__(self, iota0=0.15, iota1=0.38, k=[6], epsilonk=[0.01], m0=1):
        MagneticField.__init__(self)
        self.iota0 = iota0
        self.iota1 = iota1
        self.k = k
        self.epsilonk = epsilonk
        self.m0 = m0

    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = sopp.ReimanB(self.iota0, self.iota1, self.k, self.epsilonk, self.m0, points)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = sopp.ReimandB(self.iota0, self.iota1, self.k, self.epsilonk, self.m0, points)

    def as_dict(self, serial_objs_dict):
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        field = cls(d["iota0"], d["iota1"], d["k"], d["epsilonk"], d["m0"])
        decoder = GSONDecoder()
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class UniformInterpolationRule(sopp.UniformInterpolationRule):
    pass


class ChebyshevInterpolationRule(sopp.ChebyshevInterpolationRule):
    pass


class InterpolatedField(sopp.InterpolatedField, MagneticField):
    r"""
    This field takes an existing field and interpolates it on a regular grid in :math:`r,\phi,z`.
    This resulting interpolant can then be evaluated very quickly.
    """

    def __init__(self, field, degree, rrange, phirange, zrange, extrapolate=True, nfp=1, stellsym=False, skip=None):
        r"""
        Args:
            field: the underlying :mod:`simsopt.field.magneticfield.MagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            rrange: a 3-tuple of the form ``(rmin, rmax, nr)``. This mean that the interval :math:`[rmin, rmax]` is
                    split into ``nr`` many subintervals.
            phirange: a 3-tuple of the form ``(phimin, phimax, nphi)``.
            zrange: a 3-tuple of the form ``(zmin, zmax, nz)``.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``phimin=0`` and
                 ``phimax=2*np.pi/nfp``.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``z`` is always mapped to be positive, hence it makes sense to use
                      ``zmin=0``.
            skip: a function that takes in a point (in cylindrical (r,phi,z)
                  coordinates) and returns whether to skip that location when
                  building the interpolant or not. The signature should be

                  .. code-block:: Python

                      def skip(r: double, phi: double, z: double) -> bool:
                          ...

                  See also here
                  https://github.com/hiddenSymmetries/simsopt/pull/227 for a
                  graphical illustration.

        """
        MagneticField.__init__(self)
        if stellsym and zrange[0] != 0:
            logger.warning(fr"Sure about zrange[0]={zrange[0]}? When exploiting stellarator symmetry, the interpolant is never evaluated for z<0.")
        if nfp > 1 and abs(phirange[1] - 2*np.pi/nfp) > 1e-14:
            logger.warning(fr"Sure about phirange[1]={phirange[1]}? When exploiting rotational symmetry, the interpolant is never evaluated for phi>2\pi/nfp.")

        if skip is None:
            def skip(xs, ys, zs):
                return [False for _ in xs]

        sopp.InterpolatedField.__init__(self, field, degree, rrange, phirange, zrange, extrapolate, nfp, stellsym, skip)
        self.__field = field

    def to_vtk(self, filename):
        """Export the field evaluated on a regular grid for visualisation with e.g. Paraview."""
        degree = self.rule.degree
        MagneticField.to_vtk(
            self, filename,
            nr=self.r_range[2]*degree+1,
            nphi=self.phi_range[2]*degree+1,
            nz=self.z_range[2]*degree+1,
            rmin=self.r_range[0], rmax=self.r_range[1],
            zmin=self.z_range[0], zmax=self.z_range[1]
        )
