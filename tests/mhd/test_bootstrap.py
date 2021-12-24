# Test ideas:
# * Reproduce fig 2, 3, 4, 5, 6 from Redl paper
# * Evaluate Redl's <j dot B> for Wistell-A and LDRD QA configs for which I already have self-consistent SFINCS calculations

import unittest
import logging
import os
import numpy as np
from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl, \
    vmec_j_dot_B_Redl
from simsopt.mhd.profiles import ProfilePolynomial
from simsopt.mhd.vmec import Vmec
from simsopt.util.constants import ELEMENTARY_CHARGE
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)


class BootstrapTests(unittest.TestCase):
    def test_compute_trapped_fraction(self):
        ns = 2
        ntheta = 15
        nphi = 7
        nfp = 3
        modB = np.zeros((ntheta, nphi, ns))
        sqrtg = np.zeros((ntheta, nphi, ns))

        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi, theta = np.meshgrid(phi1d, theta1d)

        sqrtg[:, :, 0] = 10.0
        sqrtg[:, :, 1] = -25.0

        modB[:, :, 0] = 13.0 + 2.6 * np.cos(theta)
        modB[:, :, 1] = 9.0 + 3.7 * np.sin(theta - nfp * phi)

        Bmin, Bmax, epsilon, fsa_b2, f_t = compute_trapped_fraction(modB, sqrtg)
        # The average of (b0 + b1 cos(theta))^2 is b0^2 + (1/2) * b1^2
        np.testing.assert_allclose(fsa_b2, [13.0 ** 2 + 0.5 * 2.6 ** 2, 9.0 ** 2 + 0.5 * 3.7 ** 2])
        np.testing.assert_allclose(Bmin, [13.0 - 2.6, 9.0 - 3.7], rtol=1e-4)
        np.testing.assert_allclose(Bmax, [13.0 + 2.6, 9.0 + 3.7], rtol=1e-4)
        np.testing.assert_allclose(epsilon, [2.6 / 13.0, 3.7 / 9.0], rtol=1e-3)

    def test_trapped_fraction_Kim(self):
        """
        Compare the trapped fraction to eq (C18) in Kim, Diamond, &
        Groebner, Physics of Fluids B 3, 2050 (1991)
        """
        ns = 50
        ntheta = 100
        nphi = 3
        B0 = 7.5
        epsilon_in = np.linspace(0, 1, ns, endpoint=False)  # Avoid divide-by-0 when epsilon=1
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        nfp = 3
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        #phi1d = np.array([0])
        phi, theta = np.meshgrid(phi1d, theta1d)
        modB = np.zeros((ntheta, nphi, ns))
        sqrtg = np.zeros((ntheta, nphi, ns))
        for js in range(ns):
            # Eq (A6)
            modB[:, :, js] = B0 / (1 + epsilon_in[js] * np.cos(theta))
            # For Jacobian, use eq (A7) for the theta dependence,
            # times an arbitrary overall scale factor
            sqrtg[:, :, js] = 6.7 * (1 + epsilon_in[js] * np.cos(theta))
        Bmin, Bmax, epsilon_out, fsa_B2, f_t = compute_trapped_fraction(modB, sqrtg)

        f_t_Kim = 1.46 * np.sqrt(epsilon_in) - 0.46 * epsilon_in  # Eq (C18) in Kim et al

        np.testing.assert_allclose(Bmin, B0 / (1 + epsilon_in))
        np.testing.assert_allclose(Bmax, B0 / (1 - epsilon_in))
        np.testing.assert_allclose(epsilon_in, epsilon_out)
        # Eq (A8):
        np.testing.assert_allclose(fsa_B2, B0 * B0 / np.sqrt(1 - epsilon_in ** 2), rtol=1e-6)
        np.testing.assert_allclose(f_t, f_t_Kim, rtol=0.1, atol=0.07)  # We do not expect precise agreement. 

        if False:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            #plt.plot(epsilon_in, f_t, label='simsopt')
            #plt.plot(epsilon_in, f_t_Kim, label='Kim')
            plt.plot(np.sqrt(epsilon_in), f_t, label='simsopt')
            plt.plot(np.sqrt(epsilon_in), f_t_Kim, label='Kim')
            plt.xlabel('sqrt(epsilon)')
            plt.title('Trapped fraction $f_t$')
            plt.legend(loc=0)

            plt.subplot(1, 2, 2)
            #plt.plot(epsilon_in, (f_t_Kim - f_t) / (1 - f_t))
            plt.plot(epsilon_in, f_t_Kim - f_t)
            plt.title('Relative difference in $f_c$')
            plt.xlabel('epsilon')
            #plt.plot(epsilon_in, epsilon_out, 'r')
            #plt.plot(epsilon_in, epsilon_in, ':k')
            plt.show()

    def test_Redl_second_pass(self):
        """
        A second pass through coding up the equations from Redl et al,
        Phys Plasmas (2021) to make sure I didn't make transcription
        errors.
        """
        # Make up some arbitrary functions to use for input:
        ne = ProfilePolynomial(1.0e20 * np.array([1, 0, -0.8]))
        Te = ProfilePolynomial(25e3 * np.array([1, -0.9]))
        Ti = ProfilePolynomial(20e3 * np.array([1, -0.9]))
        Zeff = ProfilePolynomial(np.array([1.5, 0.5]))
        s = np.linspace(0, 1, 20)
        s = s[1:]  # Avoid divide-by-0 on axis
        helicity_N = 4
        G = 32.0 - s
        epsilon = 0.3 * np.sqrt(s)
        f_t = 1.46 * np.sqrt(epsilon)
        R = 6.0 + 0.3 * s
        psi_edge = 68 / (2 * np.pi)
        iota = 0.95 - 0.7 * s

        # Evaluate profiles on the s grid:
        ne_s = ne(s)
        Te_s = Te(s)
        Ti_s = Ti(s)
        Zeff_s = Zeff(s)
        ni_s = ne_s / Zeff_s
        d_ne_d_s = ne.dfds(s)
        d_Te_d_s = Te.dfds(s)
        d_Ti_d_s = Ti.dfds(s)

        # Sauter eq (18d)-(18e):
        ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_s) / Te_s)
        ln_Lambda_ii = 30.0 - np.log((Zeff_s ** 3) * np.sqrt(ni_s) / (Ti_s ** 1.5))

        # Sauter eq (18b)-(18c):
        nu_e = (6.921e-18) * R * ne_s * Zeff_s * ln_Lambda_e \
            / (iota * (Te_s ** 2) * (epsilon ** 1.5))
        nu_i = (4.90e-18) * R * ni_s * (Zeff_s ** 4) * ln_Lambda_ii \
            / (iota * (Ti_s ** 2) * (epsilon ** 1.5))

        # Redl eq (11):
        X31 = f_t / (1 + 0.67 * (1 - 0.7 * f_t) * np.sqrt(nu_e) / (0.56 + 0.44 * Zeff_s) \
                     + (0.52 + 0.086 * np.sqrt(nu_e)) * (1 + 0.87 * f_t) * nu_e / (1 + 1.13 * np.sqrt(Zeff_s - 1)))

        # Redl eq (10):
        Zfac = Zeff_s ** 1.2 - 0.71
        L31 = (1 + 0.15 / Zfac) * X31 - 0.22 / Zfac * (X31 ** 2) \
            + 0.01 / Zfac * (X31 ** 3) + 0.06 / Zfac * (X31 ** 4)

        # Redl eq (14):
        X32e = f_t / (1 + 0.23 * (1 - 0.96 * f_t) * np.sqrt(nu_e / Zeff_s) \
                      + 0.13 * (1 - 0.38 * f_t) * nu_e / (Zeff_s ** 2) \
                      * (np.sqrt(1 + 2 * np.sqrt(Zeff_s - 1))
                         + f_t * f_t * np.sqrt((0.075 + 0.25 * ((Zeff_s - 1) ** 2)) * nu_e)))

        # Redl eq (13):
        F32ee = (0.1 + 0.6 * Zeff_s) / (Zeff_s * (0.77 + 0.63 * (1 + (Zeff_s - 1) ** 1.1))) * (X32e - X32e ** 4) \
            + 0.7 / (1 + 0.2 * Zeff_s) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4)) \
            + 1.3 / (1 + 0.5 * Zeff_s) * (X32e ** 4)

        # Redl eq (16)
        X32ei = f_t / (1 + 0.87 * (1 + 0.39 * f_t) * np.sqrt(nu_e) / (1 + 2.95 * ((Zeff_s - 1) ** 2)) \
                       + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_s - 1)))

        # Redl eq (15)
        F32ei = -(0.4 + 1.93 * Zeff_s) / (Zeff_s * (0.8 + 0.6 * Zeff_s)) * (X32ei - X32ei ** 4) \
            + 5.5 / (1.5 + 2 * Zeff_s) * (X32ei ** 2 - X32ei ** 4 - 0.8 * (X32ei ** 3 - X32ei ** 4)) \
            - 1.3 / (1 + 0.5 * Zeff_s) * (X32ei ** 4)

        # Redl eq (12)
        L32 = F32ei + F32ee

        # Redl eq (20):
        alpha0 = -(0.62 + 0.055 * (Zeff_s - 1)) * (1 - f_t) \
            / ((0.53 + 0.17 * (Zeff_s - 1)) * (1 - (0.31 - 0.065 * (Zeff_s - 1)) * f_t - 0.25 * (f_t ** 2)))

        # Redl eq (21):
        alpha = ((alpha0 + 0.7 * Zeff_s * np.sqrt(f_t * nu_i)) / (1 + 0.18 * np.sqrt(nu_i)) - 0.002 * (nu_i ** 2) * (f_t ** 6)) \
            / (1 + 0.004 * (nu_i ** 2) * (f_t ** 6))

        # Redl eq (19):
        L34 = L31

        Te_J = Te_s * ELEMENTARY_CHARGE
        Ti_J = Ti_s * ELEMENTARY_CHARGE
        d_Te_d_s_J = d_Te_d_s * ELEMENTARY_CHARGE
        d_Ti_d_s_J = d_Ti_d_s * ELEMENTARY_CHARGE
        pe = ne_s * Te_J
        pi = ni_s * Ti_J
        p = pe + pi
        Rpe = pe / p
        d_ni_d_s = d_ne_d_s / Zeff_s
        d_p_d_s = ne_s * d_Te_d_s_J + Te_J * d_ne_d_s + ni_s * d_Ti_d_s_J + Ti_J * d_ni_d_s
        # Equation from the bottom of the right column of Sauter errata:
        factors = -G * pe / (psi_edge * (iota - helicity_N))
        dnds_term = factors * L31 * p / pe * ((Te_J * d_ne_d_s + Ti_J * d_ni_d_s) / p)
        dTeds_term = factors * (L31 * p / pe * (ne_s * d_Te_d_s_J) / p + L32 * (d_Te_d_s_J / Te_J))
        dTids_term = factors * (L31 * p / pe * (ni_s * d_Ti_d_s_J) / p + L34 * alpha * (1 - Rpe) / Rpe * (d_Ti_d_s_J / Ti_J))
        jdotB_pass2 = -G * pe * (L31 * p / pe * (d_p_d_s / p) + L32 * (d_Te_d_s_J / Te_J) + L34 * alpha * (1 - Rpe) / Rpe * (d_Ti_d_s_J / Ti_J)) / (psi_edge * (iota - helicity_N))

        jdotB, details = j_dot_B_Redl(s, ne, Te, Ti, Zeff, R, iota, G, epsilon, f_t, psi_edge, helicity_N)

        atol = 1e-13
        rtol = 1e-13
        np.testing.assert_allclose(details.nu_e_star, nu_e, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.nu_i_star, nu_i, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.X31, X31, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.L31, L31, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.X32e, X32e, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.F32ee, F32ee, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.X32ei, X32ei, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.F32ei, F32ei, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.L32, L32, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.alpha, alpha, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.dTeds_term, dTeds_term, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.dTids_term, dTids_term, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.dnds_term, dnds_term, atol=atol, rtol=rtol)
        np.testing.assert_allclose(jdotB, jdotB_pass2, atol=atol, rtol=rtol)

    def test_vmec_j_dot_B_Redl(self):
        ne = ProfilePolynomial(1.0e20 * np.array([1, 0, -0.8]))
        Te = ProfilePolynomial(25e3 * np.array([1, -0.9]))
        Ti = Te
        Zeff = 1
        surfaces = np.linspace(0, 1, 20)
        helicity_N = 0
        filename = os.path.join(TEST_DIR, 'wout_ITERModel_reference.nc')
        vmec = Vmec(filename)
        jdotB, details = vmec_j_dot_B_Redl(vmec, surfaces, ne, Te, Ti, Zeff, helicity_N, plot=False)

        if False:
            import matplotlib.pyplot as plt
            nrows = 2
            ncols = 2
            for j in range(4):
                plt.subplot(nrows, ncols, j + 1)
                plt.contourf(details.phi1d, details.theta1d, details.modB[:, :, j], 25)
                plt.ylabel('theta')
                plt.xlabel('phi')
                plt.title(f'modB, s={surfaces[j]}')
                plt.colorbar()
            plt.tight_layout()
            plt.show()
