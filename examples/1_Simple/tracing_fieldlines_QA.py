#!/usr/bin/env python3

"""
This example demonstrates how to use SIMSOPT to compute Poincare plots.

This example also illustrates how the coil shapes resulting from a
simsopt stage-2 optimization can be loaded in to another script for
analysis.  The coil shape data used in this script,
``inputs/biot_savart_opt.json``, can be re-generated by running the
example 2_Intermediate/stage_two_optimization.py.
"""

import time
import os
import logging
import sys
from pathlib import Path
import numpy as np
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

import simsopt
from simsopt.field import InterpolatedField
from simsopt.geo import SurfaceRZFourier
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data

print("Running 1_Simple/tracing_fieldlines_QA.py")
print("=========================================")

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nfieldlines = 3 if ci else 10
tmax_fl = 10000 if ci else 20000
degree = 2 if ci else 4

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# Load in the boundary surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'
# Note that the range must be "full torus"!
quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(nphi=200, ntheta=30, range="full torus")
surf = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi, quadpoints_theta)
nfp = surf.nfp

# Load in the optimized coils from stage_two_optimization.py:
coils_filename = Path(__file__).parent / "inputs" / "biot_savart_opt.json"
bs = simsopt.load(coils_filename)

surf.to_vtk(OUT_DIR + 'surface')
sc_fieldline = SurfaceClassifier(surf, h=0.03, p=2)
sc_fieldline.to_vtk(OUT_DIR + 'levelset', h=0.02)


def trace_fieldlines(bfield, label):
    t1 = time.time()
    # Set initial grid of points for field line tracing, going from
    # the magnetic axis to the surface. The actual plasma boundary is
    # at R=1.300425, but the outermost initial point is a bit inward
    # from that, R = 1.295, so the SurfaceClassifier does not think we
    # have exited the surface
    R0 = np.linspace(1.2125346, 1.295, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm is None or comm.rank == 0:
        particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)


# uncomment this to run tracing using the biot savart field (very slow!)
# trace_fieldlines(bs, 'bs')


# Bounds for the interpolated magnetic field chosen so that the surface is
# entirely contained in it
n = 20
rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
zs = surf.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/nfp, n*2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n//2)


def skip(rs, phis, zs):
    # The RegularGrindInterpolant3D class allows us to specify a function that
    # is used in order to figure out which cells to be skipped.  Internally,
    # the class will evaluate this function on the nodes of the regular mesh,
    # and if *all* of the eight corners are outside the domain, then the cell
    # is skipped.  Since the surface may be curved in a way that for some
    # cells, all mesh nodes are outside the surface, but the surface still
    # intersects with a cell, we need to have a bit of buffer in the signed
    # distance (essentially blowing up the surface a bit), to avoid ignoring
    # cells that shouldn't be ignored
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
    print("Skip", sum(skip), "cells out of", len(skip), flush=True)
    return skip


print('Initializing InterpolatedField')
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
)
print('Done initializing InterpolatedField.')

bsh.set_points(surf.gamma().reshape((-1, 3)))
bs.set_points(surf.gamma().reshape((-1, 3)))
Bh = bsh.B()
B = bs.B()
print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))

print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))

print('Beginning field line tracing')
trace_fieldlines(bsh, 'bsh')

print("End of 1_Simple/tracing_fieldlines_QA.py")
print("========================================")
