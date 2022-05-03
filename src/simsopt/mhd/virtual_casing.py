# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides routines for interacting with the
``virtual_casing`` package by D Malhotra et al, for e.g. computing the
magnetic field on a surface due to currents outside the surface.

For details of the algorithm, see
D Malhotra, A J Cerfon, M O'Neil, and E Toler,
"Efficient high-order singular quadrature schemes in magnetic fusion",
Plasma Physics and Controlled Fusion 62, 024004 (2020).
"""

import os
import logging
from datetime import datetime

import numpy as np
from scipy.io import netcdf_file

from .vmec_diagnostics import B_cartesian
from .vmec import Vmec
from ..geo.surfacerzfourier import SurfaceRZFourier
from ..geo.surface import best_nphi_over_ntheta
from ..util.fourier_interpolation import fourier_interpolation

logger = logging.getLogger(__name__)


class VirtualCasing:
    r"""
    Use the virtual casing principle to compute the contribution to
    the total magnetic field due to current outside a bounded surface.

    Usually, an instance of this class is created using the
    :func:`from_vmec()` class method, which also drives the
    computationally demanding part of the calculation (solving the
    integral equation).  In the future, interfaces to other
    equilibrium codes may be added.

    In the standard 2-stage approach to stellarator optimization, the
    virtual casing calculation is run once, at the end of stage 1
    (optimizing the plasma shape), with the result provided as input
    to stage 2 (optimizing the coil shapes). In this case, you can use
    the :func:`save()` function or the ``filename`` argument of
    :func:`from_vmec()` to save the results of the virtual casing
    calculation.  These saved results can then be loaded in later
    using the :func:`load()` class method, when needed for solving the
    stage-2 problem.

    A common situation is that you may wish to use a different number
    of grid points in :math:`\theta` and :math:`\phi` on the surface
    for the stage-2 optimization compared to the number of grid points
    used for the virtual casing calculation. In this situation, the
    :func:`resample()` function is provided to interpolate the virtual
    casing results onto whatever grid you wish to use for the stage-2
    problem.

    To set the grid resolutions ``nphi`` and ``ntheta``, it can be
    convenient to use the function
    :func:`simsopt.geo.surface.best_nphi_over_ntheta`.

    An instance of this class has the following attributes. For all
    vector quantites, Cartesian coordinates are used, corresponding to
    array dimensions of size 3:

    - ``nphi``: The number of grid points in the toroidal angle :math:`\phi`, for the full torus.
    - ``ntheta``: The number of grid points in the poloidal angle :math:`\theta`.
    - ``phi``: An array of size ``(nphi,)`` with the grid points of :math:`\phi`.
    - ``theta``: An array of size ``(ntheta,)`` with the grid points of :math:`\theta`.
    - ``gamma``: An array of size ``(nphi, ntheta, 3)`` with the position vector on the surface.
    - ``unit_normal``: An array of size ``(nphi, ntheta, 3)`` with the unit normal vector on the surface.
    - ``B_total``: An array of size ``(nphi, ntheta, 3)`` with the total magnetic field vector on the surface.
    - ``B_external``: An array of size ``(nphi, ntheta, 3)`` with the contribution
      to the magnetic field due to current outside the surface.
    - ``B_external_normal``: An array of size ``(nphi, ntheta)`` with the contribution
      to the magnetic field due to current outside the surface, taking just the component
      normal to the surface.
    """

    @classmethod
    def from_vmec(cls, vmec, src_nphi, src_ntheta=None, digits=6, filename="auto", trgt_nphi=None, trgt_ntheta=None):
        """
        Given a :obj:`~simsopt.mhd.vmec.Vmec` object, compute the
        contribution to the total magnetic field due to currents outside
        the plasma.

        This function requires the python ``virtual_casing`` package to be
        installed.

        The argument ``nphi`` refers to the number of points around the
        full torus. It must be a multiple of ``2 * nfp``, so there is an
        integer number of points per half field period.

        To set the grid resolutions ``nphi`` and ``ntheta``, it can be
        convenient to use the function
        :func:`simsopt.geo.surface.best_nphi_over_ntheta`. This is
        done automatically if you omit the ``ntheta`` argument.

        For now, this routine only works for stellarator symmetry.

        Args:
            vmec: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`, or the name of a
              Vmec ``input.*`` or ``wout*`` file.
            nphi: Number of grid points toroidally for the calculation.
            ntheta: Number of grid points poloidally for the calculation. If ``None``,
              the number of grid points will be calculated automatically using
              :func:`simsopt.geo.surface.best_nphi_over_ntheta()` to minimize
              the grid anisotropy, given the specified ``nphi``.
            digits: Approximate number of digits of precision for the calculation.
            filename: If not ``None``, the results of the virtual casing calculation
              will be saved in this file. For the default value of ``"auto"``, the
              filename will automatically be set to ``"vcasing_<extension>.nc"``
              where ``<extension>`` is the string associated with Vmec input and output
              files, analogous to the Vmec output file ``"wout_<extension>.nc"``.
        """
        import virtual_casing as vc_module

        if not isinstance(vmec, Vmec):
            vmec = Vmec(vmec)

        vmec.run()
        nfp = vmec.wout.nfp
        stellsym = not bool(vmec.wout.lasym)
        if vmec.wout.lasym:
            raise RuntimeError('virtual casing presently only works for stellarator symmetry')
        if src_nphi % (2) != 0:
            raise ValueError(f'nphi must be a multiple of 2. nphi={nphi}')

        if src_ntheta is None:
            src_ntheta = int(nfp * src_nphi / best_nphi_over_ntheta(vmec.boundary))
            logger.debug(f'new src_ntheta: {src_ntheta}')

        # The requested nphi and ntheta may not match the quadrature
        # points in vmec.boundary, and the range may not be "full torus",
        # so generate a SurfaceRZFourier with the desired resolution:
        if stellsym:
            ran = "half period"
        else:
            ran = "field period"
        surf = SurfaceRZFourier(mpol=vmec.wout.mpol, ntor=vmec.wout.ntor, nfp=nfp,
                                nphi=src_nphi, ntheta=src_ntheta, range=ran)
        for jmn in range(vmec.wout.mnmax):
            surf.set_rc(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.rmnc[jmn, -1])
            surf.set_zs(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.zmns[jmn, -1])
        Bxyz = B_cartesian(vmec, nphi=src_nphi, ntheta=src_ntheta, range=ran)
        gamma = surf.gamma()
        logger.debug(f'gamma.shape: {gamma.shape}')
        logger.debug(f'Bxyz[0].shape: {Bxyz[0].shape}')

        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta
        trgt_surf = SurfaceRZFourier(mpol=vmec.wout.mpol, ntor=vmec.wout.ntor, nfp=nfp,
                                       nphi=trgt_nphi, ntheta=trgt_ntheta, range=ran)
        trgt_surf.x = surf.x

        unit_normal = trgt_surf.unitnormal()
        logger.debug(f'unit_normal.shape: {unit_normal.shape}')

        # virtual_casing wants all input arrays to be 1D. The order is
        # {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}
        # where Nt is toroidal (not theta!) and Np is poloidal (not phi!)
        gamma1d = np.zeros(src_nphi * src_ntheta * 3)
        B1d = np.zeros(src_nphi * src_ntheta * 3)
        B3d = np.zeros((src_nphi, src_ntheta, 3))
        for jxyz in range(3):
            gamma1d[jxyz * src_nphi * src_ntheta: (jxyz + 1) * src_nphi * src_ntheta] = gamma[:, :, jxyz].flatten(order='C')
            B1d[jxyz * src_nphi * src_ntheta: (jxyz + 1) * src_nphi * src_ntheta] = Bxyz[jxyz].flatten(order='C')
            B3d[:, :, jxyz] = Bxyz[jxyz]

        """
        # Check order:
        index = 0
        for jxyz in range(3):
            for jphi in range(src_nphi):
                for jtheta in range(src_ntheta):
                    np.testing.assert_allclose(gamma1d[index], gamma[jphi, jtheta, jxyz])
                    np.testing.assert_allclose(B1d[index], Bxyz[jxyz][jphi, jtheta])
                    index += 1
        """

        vcasing = vc_module.VirtualCasing()
        vcasing.setup(
            digits, nfp, stellsym,
            src_nphi, src_ntheta, gamma1d,
            src_nphi, src_ntheta,
            trgt_nphi, trgt_ntheta)
        # This next line launches the main computation:
        Bexternal1d = np.array(vcasing.compute_external_B(B1d))

        # Unpack 1D array results:
        Bexternal3d = np.zeros((trgt_nphi, trgt_ntheta, 3))
        for jxyz in range(3):
            Bexternal3d[:, :, jxyz] = Bexternal1d[jxyz * trgt_nphi * trgt_ntheta: (jxyz + 1) * trgt_nphi * trgt_ntheta].reshape((trgt_nphi, trgt_ntheta), order='C')

        """
        # Check order:
        index = 0
        for jxyz in range(3):
            for jphi in range(trgt_nphi):
                for jtheta in range(trgt_ntheta):
                    np.testing.assert_allclose(Bexternal1d[index], Bexternal3d[jphi, jtheta, jxyz])
                    index += 1
        """

        Bexternal_normal = np.sum(Bexternal3d * unit_normal, axis=2)

        vc = cls()
        vc.src_ntheta = src_ntheta
        vc.src_nphi = src_nphi
        vc.src_theta = surf.quadpoints_theta
        vc.src_phi = surf.quadpoints_phi

        vc.trgt_ntheta = trgt_ntheta
        vc.trgt_nphi = trgt_nphi
        vc.trgt_theta = trgt_surf.quadpoints_theta
        vc.trgt_phi = trgt_surf.quadpoints_phi

        vc.nfp = nfp
        vc.B_total = B3d
        vc.gamma = gamma
        vc.unit_normal = unit_normal
        vc.B_external = Bexternal3d
        vc.B_external_normal = Bexternal_normal

        # if filename is not None:
        #     if filename == 'auto':
        #         directory, basefile = os.path.split(vmec.output_file)
        #         filename = os.path.join(directory, 'vcasing' + basefile[4:])
        #         logger.debug(f'New filename: {filename}')
        #     vc.save(filename)

        return vc

    def save(self, filename="vcasing.nc"):
        """
        Save the results of a virtual casing calculation in a NetCDF file.

        Args:
            filename: Name of the file to create.
        """
        with netcdf_file(filename, 'w') as f:
            f.history = 'This file created by simsopt on ' + datetime.now().strftime("%B %d %Y, %H:%M:%S")
            f.createDimension('ntheta', self.ntheta)
            f.createDimension('nphi', self.nphi)
            f.createDimension('xyz', 3)

            ntheta = f.createVariable('ntheta', 'i', tuple())
            ntheta.assignValue(self.ntheta)
            ntheta.description = 'Number of grid points in the poloidal angle theta'
            ntheta.units = 'Dimensionless'

            nphi = f.createVariable('nphi', 'i', tuple())
            nphi.assignValue(self.nphi)
            nphi.description = 'Number of grid points in the toroidal angle phi, covering the full torus'
            nphi.units = 'Dimensionless'

            nphi = f.createVariable('nfp', 'i', tuple())
            nphi.assignValue(self.nfp)
            nphi.description = 'Periodicity in toroidal direction'
            nphi.units = 'Dimensionless'

            theta = f.createVariable('theta', 'd', ('ntheta',))
            theta[:] = self.theta
            theta.description = 'Grid points in the poloidal angle theta. Note that theta extends over [0, 1) not [0, 2pi).'
            theta.units = 'Dimensionless'

            phi = f.createVariable('phi', 'd', ('nphi',))
            phi[:] = self.phi
            phi.description = 'Grid points in the toroidal angle phi. Note that phi extends over [0, 1) not [0, 2pi).'
            phi.units = 'Dimensionless'

            gamma = f.createVariable('gamma', 'd', ('nphi', 'ntheta', 'xyz'))
            gamma[:, :, :] = self.gamma
            gamma.description = 'Position vector on the boundary surface'
            gamma.units = 'meter'

            unit_normal = f.createVariable('unit_normal', 'd', ('nphi', 'ntheta', 'xyz'))
            unit_normal[:, :, :] = self.unit_normal
            unit_normal.description = 'Unit-length normal vector on the boundary surface'
            unit_normal.units = 'Dimensionless'

            B_total = f.createVariable('B_total', 'd', ('nphi', 'ntheta', 'xyz'))
            B_total[:, :, :] = self.B_total
            B_total.description = 'Total magnetic field vector on the surface, including currents both inside and outside of the surface'
            B_total.units = 'Tesla'

            B_external = f.createVariable('B_external', 'd', ('nphi', 'ntheta', 'xyz'))
            B_external[:, :, :] = self.B_external
            B_external.description = 'Contribution to the magnetic field vector on the surface due only to currents outside the surface'
            B_external.units = 'Tesla'

            B_external_normal = f.createVariable('B_external_normal', 'd', ('nphi', 'ntheta'))
            B_external_normal[:, :] = self.B_external_normal
            B_external_normal.description = 'Component of B_external normal to the surface'
            B_external_normal.units = 'Tesla'

    @classmethod
    def load(cls, filename):
        """
        Load in the results of a previous virtual casing calculation,
        previously saved in NetCDF format.

        Args:
            filename: Name of the file to load.
        """
        vc = cls()
        with netcdf_file(filename, mmap=False) as f:
            for key, val in f.variables.items():
                val2 = val[()]  # Convert to numpy array
                vc.__setattr__(key, val2)
        return vc

    def plot(self, ax=None, show=True):
        """
        Plot ``B_external_normal``, the component normal to the surface of
        the magnetic field generated by currents outside the surface.
        This routine requires ``matplotlib``.

        Args:
            ax: The axis object on which to plot. This argument is useful when plotting multiple
              objects on the same axes. If equal to the default ``None``, a new axis will be created.
            show: Whether to call matplotlib's ``show()`` function.

        Returns:
            An axis which could be passed to a further call to matplotlib if desired.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        contours = ax.contourf(self.trgt_phi, self.trgt_theta, self.B_external_normal.T, 25)
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        ax.set_title('B_external_normal [Tesla]')
        fig.colorbar(contours)
        fig.tight_layout()
        if show:
            plt.show()
        return ax
