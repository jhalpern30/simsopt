#!/usr/bin/env python

import os
import numpy as np
from simsopt.util.mpi import MpiPartition, log
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve

"""
This example shows how scripting can be used to increase the size
of the parameter space and refine the resolution of the calculations
during an optimization.

The objective function for this example targets quasi-axisymmetry and
the iota profile.  First we optimize in a small parameter space, with
m and |n| values up through 1.  Then the parameter space is widened to
include m and |n| values up through 2, with the resolution of VMEC and
booz_xform increased at the same time.  Then the parameter space is
widened again to include m and |n| values up through 3, and again the
resolution for VMEC and booz_xform is increased.
"""

#log()

print("Running 2_Intermediate/resolution_increase.py")
print("=============================================")

mpi = MpiPartition()
mpi.write()

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA')
vmec = Vmec(filename, mpi=mpi)
vmec.verbose = mpi.proc0_world
surf = vmec.boundary

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=0)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 6, 1),
                                        (vmec.mean_iota, 0.42, 1),
                                        (qs.residuals, 0, 1)])

# Fourier modes of the boundary with m <= max_mode and |n| <= max_mode
# will be varied in the optimization. A larger range of modes are
# included in the VMEC and booz_xform calculations.
for step in range(3):
    max_mode = step + 1

    # VMEC's mpol & ntor will be 3, 4, 5:
    vmec.indata.mpol = 3 + step
    vmec.indata.ntor = vmec.indata.mpol

    if mpi.proc0_world:
        print("Beginning optimization with max_mode =", max_mode, \
              ", vmec mpol=ntor=", vmec.indata.mpol, \
              ". Previous vmec iteration = ", vmec.iter)

    # Define parameter space:
    surf.local_fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, 
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Major radius

    # For the test to run quickly, we stop after the first function
    # evaluation, by passing max_nfev=1 to scipy.optimize. For a
    # "real" optimization, remove the max_nfev parameter below.
    least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=1)

    # Preserve the output file from the last iteration, so it is not
    # deleted when vmec runs again:
    vmec.files_to_delete = []

    if mpi.proc0_world:
        print(f"Done optimization with max_mode ={max_mode}. "
              f"Final vmec iteration = {vmec.iter}")

print("Good bye")

print("End of 2_Intermediate/resolution_increase.py")
print("=============================================")
