# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
from typing import Union
from datetime import datetime

import numpy as np
from scipy.io import netcdf

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    import vmec
except ImportError as e:
    vmec = None
    logger.debug(str(e))

from .._core.graph_optimizable import Optimizable
from .._core.util import Struct, ObjectiveFailure
from ..geo.surfacerzfourier import SurfaceRZFourier

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None

# Flags used by runvmec():
restart_flag = 1
readin_flag = 2
timestep_flag = 4
output_flag = 8
cleanup_flag = 16
reset_jacdt_flag = 32


def to_namelist_bool(bool_in):
    """ Convert a boolean to a format suitable for fortran namelist input """
    return "T" if bool_in else "F"


def array_to_namelist(arr, aux_s=False):
    if aux_s:
        if np.all(arr < 0):
            index = 0
        else:
            index = np.max(np.where(arr >= 0))
    else:
        if np.all(arr == 0):
            index = 0
        else:
            index = np.max(np.nonzero(arr))
    nml = ''
    for j in range(index + 1):
        nml += f'{arr[j]} '
    nml += '\n'
    return nml

# Documentation of flags for runvmec() from the VMEC source code:
#
#value flag-name         calls routines to...
#----- ---------         ---------------------
#  1   restart_flag      reset internal run-control parameters
#                        (for example, if jacobian was bad, to try a smaller
#                        time-step)
#  2   readin_flag       read in data from input_file and initialize parameters
#                        or arrays which do not dependent on radial grid size
#                        allocate internal grid-dependent arrays used by vmec;
#                        initialize internal grid-dependent vmec profiles (xc,
#                        iota, etc);
#                        setup loop for radial multi-grid meshes or, if
#                        ns_index = ictrl_array(4) is > 0, use radial grid
#                        points specified by ns_array[ns_index]
#  4   timestep_flag     iterate vmec either by "niter" time steps or until ftol
#                        satisfied, whichever comes first.
#                        If numsteps (see below) > 0, vmec will return
#                        to caller after numsteps, rather than niter, steps.
#  8   output_flag       write out output files (wout, jxbout)
# 16   cleanup_flag      cleanup (deallocate arrays) - this terminates present
#                        run of the sequence
#                        This flag will be ignored if the run might be continued.
#                        For example, if ier_flag (see below) returns the value
#                        more_iter_flag, the cleanup code will be skipped even if
#                        cleanup_flag is set, so that the run could be continued
#                        on the next call to runvmec.
# 32   reset_jacdt_flag  Resets ijacobian flag and time step to delt0
#                        thus, setting ictrl_flag = 1+2+4+8+16 will perform ALL
#                        the tasks thru cleanup_flag in addition,
#                        if ns_index = 0 and numsteps = 0 (see below), vmec will
#                        control its own run history


class Vmec(Optimizable):
    r"""
    This class represents the VMEC equilibrium code.

    You can initialize this class either from a VMEC
    ``input.<extension>`` file or from a ``wout_<extension>.nc`` output
    file. If neither is provided, a default input file is used. When
    this class is initialized from an input file, it is possible to
    modify the input parameters and run the VMEC code. When this class
    is initialized from a ``wout`` file, all the data from the
    ``wout`` file is available in memory but the VMEC code cannot be
    re-run, since some of the input data (e.g. radial multigrid
    parameters) is not available in the wout file.

    The input parameters to VMEC are all accessible as attributes of
    the ``indata`` attribute. For example, if ``vmec`` is an instance
    of ``Vmec``, then you can read or write the input resolution
    parameters using ``vmec.indata.mpol``, ``vmec.indata.ntor``,
    ``vmec.indata.ns_array``, etc. However, the boundary surface is
    different: ``rbc``, ``rbs``, ``zbc``, and ``zbs`` from the
    ``indata`` attribute are always ignored, and these arrays are
    instead taken from the simsopt surface object associated to the
    ``boundary`` attribute. If ``boundary`` is a surface based on some
    other representation than VMEC's Fourier representation, the
    surface will automatically be converted to VMEC's representation
    (:obj:`~simsopt.geo.surfacerzfourier.SurfaceRZFourier`) before
    each run of VMEC. You can replace ``boundary`` with a new surface
    object, of any type that implements the conversion function
    ``to_RZFourier()``.

    VMEC is run either when the :meth:`run()` function is called, or when
    any of the output functions like :meth:`aspect()` or :meth:`iota_axis()`
    are called.

    A caching mechanism is implemented, using the attribute
    ``need_to_run_code``. Whenever VMEC is run, or if the class is
    initialized from a ``wout`` file, this attribute is set to
    ``False``. Subsequent calls to :meth:`run()` or output functions
    like :meth:`aspect()` will not actually run VMEC again, until
    ``need_to_run_code`` is changed to ``True``. The attribute
    ``need_to_run_code`` is automatically set to ``True`` whenever the
    state vector ``.x`` is changed, and when dofs of the ``boundary``
    are changed. However, ``need_to_run_code`` is not automatically
    set to ``True`` when entries of ``indata`` are modified.

    Once VMEC has run at least once, or if the class is initialized
    from a ``wout`` file, all of the quantities in the ``wout`` output
    file are available as attributes of the ``wout`` attribute.  For
    example, if ``vmec`` is an instance of ``Vmec``, then the flux
    surface shapes can be obtained from ``vmec.wout.rmnc`` and
    ``vmec.wout.zmns``.

    Since the underlying fortran implementation of VMEC uses global
    module variables, it is not possible to have more than one python
    Vmec object with different parameters; changing the parameters of
    one would change the parameters of the other.

    An instance of this class owns just a few optimizable degrees of
    freedom, particularly ``phiedge`` and ``curtor``. The optimizable
    degrees of freedom associated with the boundary surface are owned
    by that surface object.

    When VMEC is run multiple times, the default behavior is that all
    ``wout`` output files will be deleted except for the first and
    most recent iteration on worker group 0. If you wish to keep all
    the ``wout`` files, you can set ``keep_all_files = True``. If you
    want to save the ``wout`` file for a certain intermediate
    iteration, you can set the ``files_to_delete`` attribute to ``[]``
    after that run of VMEC.

    Args:
        filename: Name of a VMEC ``input.<extension>`` file or ``wout_<extension>.nc``
          output file to use for loading the
          initial parameters. If ``None``, default parameters will be used.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for VMEC calculations. If ``None``,
          each MPI process will run VMEC independently.
        keep_all_files: If ``False``, all ``wout`` output files will be deleted
          except for the first and most recent ones from worker group 0. If
          ``True``, all ``wout`` files will be kept.

    Attributes:
        iter: Number of times VMEC has run.
        s_full_grid: The "full" grid in the radial coordinate s (normalized
          toroidal flux), including points at s=0 and s=1. Used for the output
          arrays and ``zmns``.
        s_half_grid: The "half" grid in the radial coordinate s, used for
          ``bmnc``, ``lmns``, and other output arrays. In contrast to
          wout files, this array has only ns-1 entries, so there is no
          leading 0.
        ds: The spacing between grid points for the radial coordinate s.
    """

    def __init__(self,
                 filename: Union[str, None] = None,
                 mpi: Union[MpiPartition, None] = None,
                 keep_all_files: bool = False,
                 ntheta=50,
                 nphi=50):

        if filename is None:
            # Read default input file, which should be in the same
            # directory as this file:
            filename = os.path.join(os.path.dirname(__file__), 'input.default')
            logger.info(f"Initializing a VMEC object from defaults in {filename}")

        basename = os.path.basename(filename)
        if basename[:5] == 'input':
            logger.info(f"Initializing a VMEC object from input file: {filename}")
            self.input_file = filename
            self.runnable = True
        elif basename[:4] == 'wout':
            logger.info(f"Initializing a VMEC object from wout file: {filename}")
            self.runnable = False
        else:
            raise ValueError('Invalid filename')

        self.wout = Struct()

        # Get MPI communicator:
        if (mpi is None and MPI is not None):
            self.mpi = MpiPartition(ngroups=1)
        else:
            self.mpi = mpi

        if self.runnable:
            if MPI is None:
                raise RuntimeError("mpi4py needs to be installed for running VMEC")
            if vmec is None:
                raise RuntimeError(
                    "Running VMEC from simsopt requires VMEC python extension. "
                    "Install the VMEC python extension from "
                    "https://https://github.com/hiddenSymmetries/VMEC2000")

            comm = self.mpi.comm_groups
            self.fcomm = comm.py2f()

            self.ictrl = np.zeros(5, dtype=np.int32)
            self.iter = -1
            self.keep_all_files = keep_all_files
            self.files_to_delete = []

            self.indata = vmec.vmec_input  # Shorthand
            vi = vmec.vmec_input  # Shorthand

            self.ictrl[0] = restart_flag + readin_flag
            self.ictrl[1] = 0  # ierr
            self.ictrl[2] = 0  # numsteps
            self.ictrl[3] = 0  # ns_index
            self.ictrl[4] = 0  # iseq
            verbose = True
            reset_file = ''
            logger.info('About to call runvmec to readin')
            vmec.runvmec(self.ictrl, filename, verbose, self.fcomm, reset_file)
            ierr = self.ictrl[1]
            logger.info('Done with runvmec. ierr={}. Calling cleanup next.'.format(ierr))
            # Deallocate arrays allocated by VMEC's fixaray():
            vmec.cleanup(False)
            if ierr != 0:
                raise RuntimeError("Failed to initialize VMEC from input file {}. "
                                   "error code {}".format(filename, ierr))

            objstr = " for Vmec " + str(hex(id(self)))

            # A vmec object has mpol and ntor attributes independent of
            # the boundary. The boundary surface object is initialized
            # with mpol and ntor values that match those of the vmec
            # object, but the mpol/ntor values of either the vmec object
            # or the boundary surface object can be changed independently
            # by the user.
            quadpoints_theta = np.linspace(0, 1., ntheta, endpoint=False)
            quadpoints_phi = np.linspace(0, 1., nphi, endpoint=False)
            self._boundary = SurfaceRZFourier(nfp=vi.nfp,
                                              stellsym=not vi.lasym,
                                              mpol=vi.mpol,
                                              ntor=vi.ntor,
                                              quadpoints_theta=quadpoints_theta,
                                              quadpoints_phi=quadpoints_phi)
            self.free_boundary = bool(vi.lfreeb)

            # Transfer boundary shape data from fortran to the ParameterArray:
            for m in range(vi.mpol + 1):
                for n in range(-vi.ntor, vi.ntor + 1):
                    self._boundary.rc[m, n + vi.ntor] = vi.rbc[101 + n, m]
                    self._boundary.zs[m, n + vi.ntor] = vi.zbs[101 + n, m]
                    if vi.lasym:
                        self._boundary.rs[m, n + vi.ntor] = vi.rbs[101 + n, m]
                        self._boundary.zc[m, n + vi.ntor] = vi.zbc[101 + n, m]
            self._boundary.local_full_x = self._boundary.get_dofs()

            self.need_to_run_code = True
        else:
            # Initialized from a wout file, so not runnable.
            self._boundary = SurfaceRZFourier.from_wout(filename)
            self.output_file = filename
            self.load_wout()

        # Handle a few variables that are not Parameters:
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ['delt', 'tcon0', 'phiedge', 'curtor', 'gamma']
        super().__init__(x0=x0, fixed=fixed, names=names,
                         depends_on=[self._boundary],
                         external_dof_setter=Vmec.set_dofs)

        if not self.runnable:
            # This next line must come after Optimizable.__init__
            # since that calls recompute_bell()
            self.need_to_run_code = False

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if not boundary is self._boundary:
            logging.debug('Replacing surface in boundary setter')
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.need_to_run_code = True

    def get_dofs(self):
        if not self.runnable:
            # Use default values from vmec_input
            return np.array([1, 1, 1, 0, 0])
        else:
            return np.array([self.indata.delt, self.indata.tcon0,
                             self.indata.phiedge, self.indata.curtor,
                             self.indata.gamma])

    def set_dofs(self, x):
        if self.runnable:
            self.need_to_run_code = True
            self.indata.delt = x[0]
            self.indata.tcon0 = x[1]
            self.indata.phiedge = x[2]
            self.indata.curtor = x[3]
            self.indata.gamma = x[4]

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def set_indata(self):
        """
        Transfer data from simsopt objects to Vmec's fortran module data.
        Presently, this function sets the boundary shape and magnetic
        axis shape.  In the future, the input profiles will be set
        here as well. This data transfer is performed before writing a
        Vmec input file or running Vmec. The boundary surface object
        converted to ``SurfaceRZFourier`` is returned.
        """
        if not self.runnable:
            raise RuntimeError('Cannot access indata for a Vmec object that was initialized from a wout file.')
        vi = vmec.vmec_input  # Shorthand
        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        # VMEC does not allow mpol or ntor above 101:
        if vi.mpol > 101:
            raise ValueError("VMEC does not allow mpol > 101")
        if vi.ntor > 101:
            raise ValueError("VMEC does not allow ntor > 101")
        vi.rbc[:, :] = 0
        vi.zbs[:, :] = 0
        mpol_capped = np.min([boundary_RZFourier.mpol, 101])
        ntor_capped = np.min([boundary_RZFourier.ntor, 101])
        # Transfer boundary shape data from the surface object to VMEC:
        for m in range(mpol_capped + 1):
            for n in range(-ntor_capped, ntor_capped + 1):
                vi.rbc[101 + n, m] = boundary_RZFourier.get_rc(m, n)
                vi.zbs[101 + n, m] = boundary_RZFourier.get_zs(m, n)

        # Set axis shape to something that is obviously wrong (R=0) to
        # trigger vmec's internal guess_axis.f to run. Otherwise the
        # initial axis shape for run N will be the final axis shape
        # from run N-1, which makes VMEC results depend slightly on
        # the history of previous evaluations, confusing the finite
        # differencing.
        vi.raxis_cc[:] = 0
        vi.raxis_cs[:] = 0
        vi.zaxis_cc[:] = 0
        vi.zaxis_cs[:] = 0

        return boundary_RZFourier

    def get_input(self):
        """
        Generate a VMEC input file. The result will be returned as a
        string. To save a file, see the ``write_input()`` function.
        """
        boundary_RZFourier = self.set_indata()  # Transfer the boundary from simsopt to fortran.
        vi = vmec.vmec_input  # Shorthand
        nml = '&INDATA\n'
        nml += '! This file created by simsopt on ' + datetime.now().strftime("%B %d %Y, %H:%M:%S") + '\n\n'
        nml += '! ---- Geometric parameters ----\n'
        nml += f'NFP = {vi.nfp}\n'
        nml += f'LASYM = {to_namelist_bool(vi.lasym)}\n'

        nml += '\n! ---- Resolution parameters ----\n'
        nml += f'MPOL = {vi.mpol}\n'
        nml += f'NTOR = {vi.ntor}\n'
        index = np.max(np.nonzero(vi.ns_array))
        nml += f'NS_ARRAY    ='
        for j in range(index + 1):
            nml += f'{vi.ns_array[j]:7}'
        nml += '\n'
        index = np.max(np.where(vi.niter_array > 0))
        nml += f'NITER_ARRAY ='
        for j in range(index + 1):
            nml += f'{vi.niter_array[j]:7}'
        nml += '\n'
        index = np.max(np.nonzero(vi.ftol_array))
        nml += f'FTOL_ARRAY  ='
        for j in range(index + 1):
            nml += f'{vi.ftol_array[j]:7}'
        nml += '\n'

        nml += '\n! ---- Boundary toroidal flux ----\n'
        nml += f'PHIEDGE = {vi.phiedge}\n'

        nml += '\n! ---- Pressure profile specification ----\n'
        profile_type = vi.pmass_type.decode().strip()
        nml += f'PMASS_TYPE = "{profile_type}"\n'
        nml += 'AM = ' + array_to_namelist(vi.am)
        if np.any(vi.am_aux_s >= 0):
            nml += 'AM_AUX_S = ' + array_to_namelist(vi.am_aux_s, True)
            nml += 'AM_AUX_F = ' + array_to_namelist(vi.am_aux_f)
        nml += f'PRES_SCALE = {vi.pres_scale}\n'

        nml += '\n! ---- Profile specification of iota or current ----\n'
        nml += f'NCURR = {vi.ncurr}\n'
        if vi.ncurr == 0:
            # Iota profile specified
            profile_type = vi.piota_type.decode().strip()
            nml += f'PIOTA_TYPE = "{profile_type}"\n'
            nml += 'AI = ' + array_to_namelist(vi.ai)
            if np.any(vi.ai_aux_s >= 0):
                nml += 'AI_AUX_S = ' + array_to_namelist(vi.ai_aux_s, True)
                nml += 'AI_AUX_F = ' + array_to_namelist(vi.ai_aux_f)
        else:
            # Current profile specified
            nml += f'CURTOR = {vi.curtor}\n'
            profile_type = vi.pcurr_type.decode().strip()
            nml += f'PCURR_TYPE = "{profile_type}"\n'
            nml += 'AC = ' + array_to_namelist(vi.ac)
            if np.any(vi.ac_aux_s >= 0):
                nml += 'AC_AUX_S = ' + array_to_namelist(vi.ac_aux_s, True)
                nml += 'AC_AUX_F = ' + array_to_namelist(vi.ac_aux_f)

        nml += '\n! ---- Other numerical parameters ----\n'
        nml += f'DELT = {vi.delt}\n'
        nml += f'NSTEP = {vi.nstep}\n'

        nml += '\n! ---- Boundary shape. Array index order is (n, m) ----\n'
        surf_str = boundary_RZFourier.get_nml().split('\n')
        for j in range(3, len(surf_str)):
            nml += surf_str[j] + '\n'

        return nml

    def write_input(self, filename):
        """
        Write a VMEC input file. To just get the result as a string
        without saving a file, see the ``get_input()`` function.

        Args:
            filename: Name of the file to write.
        """
        if self.mpi.proc0_groups:
            with open(filename, 'w') as f:
                f.write(self.get_input())

    def run(self):
        """
        Run VMEC, if ``need_to_run_code`` is ``True``.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run VMEC.")
            return

        if not self.runnable:
            raise RuntimeError('Cannot run a Vmec object that was initialized from a wout file.')

        logger.info("Preparing to run VMEC.")

        self.iter += 1
        base_filename = self.input_file + '_{:03d}_{:06d}'.format(
            self.mpi.group, self.iter)
        input_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename))
        self.output_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace('input.', 'wout_') + '.nc')
        mercier_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace('input.', 'mercier.'))
        jxbout_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace('input.', 'jxbout_') + '.nc')

        file_to_write = input_file if self.mpi.proc0_world else None
        # This next line also calls set_indata():
        self.write_input(file_to_write)

        logger.info("Calling VMEC reinit().")
        vmec.reinit()

        logger.info("Calling runvmec().")
        self.ictrl[0] = restart_flag + reset_jacdt_flag \
            + timestep_flag + output_flag
        self.ictrl[1] = 0  # ierr
        self.ictrl[2] = 0  # numsteps
        self.ictrl[3] = 0  # ns_index
        self.ictrl[4] = 0  # iseq
        verbose = True
        reset_file = ''
        vmec.runvmec(self.ictrl, input_file, verbose, self.fcomm, reset_file)
        ierr = self.ictrl[1]

        # Deallocate arrays, even if vmec did not converge:
        logger.info("Calling VMEC cleanup().")
        vmec.cleanup(True)

        # See VMEC2000/Sources/General/vmec_params.f for ierr codes.
        # 11 = successful_term_flag.
        # Error codes that are expected to occur due to lack of
        # convergence cause ObjectiveFailure, which the optimizer
        # handles gracefully by treating the point as bad. But the
        # user/developer should know if an error codes arises that
        # should logically never occur, so these codes raise a
        # different exception.
        if ierr in [0, 5]:
            raise RuntimeError(f"runvmec returned an error code that should " \
                               "never occur: ierr={ierr}")
        if ierr != 11:
            raise ObjectiveFailure(f"VMEC did not converge. ierr={ierr}")

        logger.info("VMEC run complete. Now loading output.")
        self.load_wout()
        logger.info("Done loading VMEC output.")

        # Group leaders handle deletion of files:
        if self.mpi.proc0_groups:
            # Delete some files produced by VMEC that we never care
            # about. For some reason the os.remove statements give a 'file
            # not found' error in the CI, hence the try-except blocks.
            try:
                os.remove(mercier_file)
            except FileNotFoundError:
                logger.debug(f'Tried to delete the file {mercier_file} but it was not found')
                raise

            try:
                os.remove(jxbout_file)
            except FileNotFoundError:
                logger.debug(f'Tried to delete the file {jxbout_file} but it was not found')
                raise

            try:
                os.remove("fort.9")
            except FileNotFoundError:
                logger.debug('Tried to delete the file fort.9 but it was not found')

            # If the worker group is not 0, delete all wout files, unless
            # keep_all_files is True:
            if (not self.keep_all_files) and (self.mpi.group > 0):
                os.remove(self.output_file)

            # Delete the previous output file, if desired:
            for filename in self.files_to_delete:
                os.remove(filename)
            self.files_to_delete = []

            # Record the latest output file to delete if we run again:
            if (self.mpi.group == 0) and (self.iter > 0) and (not self.keep_all_files):
                self.files_to_delete += [input_file, self.output_file]

        self.need_to_run_code = False

    def load_wout(self):
        """
        Read in the most recent ``wout`` file created, and store all the
        data in a ``wout`` attribute of this Vmec object.
        """
        ierr = 0
        logger.info(f"Attempting to read file {self.output_file}")

        with netcdf.netcdf_file(self.output_file, mmap=False) as f:
            for key, val in f.variables.items():
                # 2D arrays need to be transposed.
                val2 = val[()]  # Convert to numpy array
                val3 = val2.T if len(val2.shape) == 2 else val2
                self.wout.__setattr__(key, val3)

            if self.wout.ier_flag != 0:
                logger.info("VMEC did not succeed!")
                raise ObjectiveFailure("VMEC did not succeed")

            # Shorthand for a long variable name:
            self.wout.lasym = f.variables['lasym__logical__'][()]
            self.wout.volume = self.wout.volume_p

        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds

        return ierr

    def aspect(self):
        """
        Return the plasma aspect ratio.
        """
        self.run()
        return self.wout.aspect

    def volume(self):
        """
        Return the volume inside the VMEC last closed flux surface.
        """
        self.run()
        return self.wout.volume

    def iota_axis(self):
        """
        Return the rotational transform on axis
        """
        self.run()
        return self.wout.iotaf[0]

    def iota_edge(self):
        """
        Return the rotational transform at the boundary
        """
        self.run()
        return self.wout.iotaf[-1]

    def mean_iota(self):
        """
        Return the mean rotational transform. The average is taken over
        the normalized toroidal flux s.
        """
        self.run()
        return np.mean(self.wout.iotas[1:])

    def mean_shear(self):
        """
        Return an average magnetic shear, d(iota)/ds, where s is the
        normalized toroidal flux. This is computed by fitting the
        rotational transform to a linear (plus constant) function in
        s. The slope of this fit function is returned.
        """
        self.run()

        # Fit a linear polynomial:
        poly = np.polynomial.Polynomial.fit(self.s_half_grid,
                                            self.wout.iotas[1:], deg=1)
        # Return the slope:
        return poly.deriv()(0)

    def get_max_mn(self):
        """
        Look through the rbc and zbs data in fortran to determine the
        largest m and n for which rbc or zbs is nonzero.
        """
        max_m = 0
        max_n = 0
        for m in range(1, 101):
            for n in range(1, 101):
                if np.abs(vmec.vmec_input.rbc[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbs[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbs[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbc[101 + n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbc[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbs[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.rbs[101 - n, m]) > 0 \
                        or np.abs(vmec.vmec_input.zbc[101 - n, m]) > 0:
                    max_m = np.max((max_m, m))
                    max_n = np.max((max_n, n))
        # It may happen that mpol or ntor exceed the max_m or max_n
        # according to rbc/zbs. In this case, go with the larger
        # value.
        max_m = np.max((max_m, vmec.vmec_input.mpol))
        max_n = np.max((max_n, vmec.vmec_input.ntor))
        return (max_m, max_n)

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        return f"{self.name} (nfp={self.indata.nfp} mpol={self.indata.mpol}" + \
               f" ntor={self.indata.ntor})"

    def vacuum_well(self):
        """
        Compute a single number W that summarizes the vacuum magnetic well,
        given by the formula

        W = (dV/ds(s=0) - dV/ds(s=1)) / (dV/ds(s=0)

        where dVds is the derivative of the flux surface volume with
        respect to the radial coordinate s. Positive values of W are
        favorable for stability to interchange modes. This formula for
        W is motivated by the fact that

        d^2 V / d s^2 < 0

        is favorable for stability. Integrating over s from 0 to 1
        and normalizing gives the above formula for W. Notice that W
        is dimensionless, and it scales as the square of the minor
        radius. To compute dV/ds, we use

        dV/ds = 4 * pi**2 * abs(sqrt(g)_{0,0})

        where sqrt(g) is the Jacobian of (s, theta, phi) coordinates,
        computed by VMEC in the gmnc array, and _{0,0} indicates the
        m=n=0 Fourier component. Since gmnc is reported by VMEC on the
        half mesh, we extrapolate by half of a radial grid point to s
        = 0 and 1.
        """
        self.run()

        # gmnc is on the half mesh, so drop the 0th radial entry:
        dVds = 4 * np.pi * np.pi * np.abs(self.wout.gmnc[0, 1:])

        # To get from the half grid to s=0 and s=1, we must
        # extrapolate by 1/2 of a radial grid point:
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]

        well = (dVds_s0 - dVds_s1) / dVds_s0
        return well

    return_fn_map = {'aspect': aspect, 'volume': volume, 'iota_axis': iota_axis,
                     'iota_edge': iota_edge, 'mean_iota': mean_iota,
                     'mean_shear': mean_shear, 'vacuum_well': vacuum_well}
