#!/usr/bin/env python
r"""
This example script optimizes a set of relatively simple toroidal field coils
and passive superconducting coils (PSCs)
for an ARIES-CS reactor-scale version of the precise-QH stellarator from 
Landreman and Paul. 

The script should be run as:
    mpirun -n 1 python QH_psc_example.py
on a cluster machine but 
    python QH_psc_example.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

"""
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.geo.psc_grid import PSCgrid
from simsopt.objectives import SquaredFlux
from simsopt.util import in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *
import time

np.random.seed(1)  # set a seed so that the same PSCs are initialized each time

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
else:
    # Resolution needs to be reasonably high if you are doing permanent magnets
    # or small coils because the fields are quite local
    nphi = 32  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    # Make higher resolution surface for plotting Bnormal
    qphi = nphi * 4
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta * 4, endpoint=True)

poff = 2.0  # PSC grid will be offset 'poff' meters from the plasma surface
coff = 0.5  # PSC grid will be initialized between 1 m and 2 m from plasma

# Read in the plasma equilibrium file
input_name = 'input.LandremanPaul2021_QH_reactorScale_lowres'
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
range_param = 'half period'
s = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi, ntheta=ntheta
)
# Print major and minor radius
print('s.R = ', s.get_rc(0, 0))
print('s.r = ', s.get_rc(1, 0))

# Make inner and outer toroidal surfaces very high resolution,
# which helps to initialize coils precisely between the surfaces. 
s_inner = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4
)
s_outer = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 4, ntheta=ntheta * 4
)

# Make the inner and outer surfaces by extending the plasma surface
s_inner.extend_via_normal(poff)
s_outer.extend_via_normal(poff + coff)

# Make the output directory
out_str = "QH_psc_output/"
out_dir = Path("QH_psc_output")
out_dir.mkdir(parents=True, exist_ok=True)

# Save the inner and outer surfaces for debugging purposes
s_inner.to_vtk(out_str + 'inner_surf')
s_outer.to_vtk(out_str + 'outer_surf')

# initialize the coils
base_curves, curves, coils = initialize_coils('qh', TEST_DIR, s, out_dir)
currents = np.array([coil.current.get_value() for coil in coils])

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_on_axis_B(bs, s)

# Make high resolution, full torus version of the plasma boundary for plotting
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, 
    quadpoints_phi=quadpoints_phi, 
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# optimize the currents in the TF coils and plot results
# fix all the coil shapes so only the currents are optimized
# for i in range(ncoils):
#     base_curves[i].fix_all()
# bs = coil_optimization(s, bs, base_curves, curves, out_dir)
curves_to_vtk(curves, out_dir / "TF_coils", close=True)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized")

# check after-optimization average on-axis magnetic field strength
B_axis = calculate_on_axis_B(bs, s)
# B_axis = 1.0  # Don't rescale
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized", B_axis)

# Finally, initialize the psc class
kwargs_geo = {"Nx": 10, "out_dir": out_str, "initialization": "random",
               "plasma_boundary_full": s_plot} 
psc_array = PSCgrid.geo_setup_between_toroidal_surfaces(
    s, coils, s_inner, s_outer,  **kwargs_geo
)
print('Number of PSC locations = ', len(psc_array.grid_xyz))

currents = []
for i in range(psc_array.num_psc):
    currents.append(Current(psc_array.I[i]))
all_coils = coils_via_symmetries(
    psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
)
B_PSC = BiotSavart(all_coils)
# Plot initial errors from only the PSCs, and then together with the TF coils
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
make_Bnormal_plots(bs + B_PSC, s_plot, out_dir, "PSC_and_TF_initial", B_axis)

currents = []
for i in range(psc_array.num_psc * psc_array.symmetry):
    currents.append(Current(psc_array.I_all_with_sign_flips[i]))

all_coils = coils_via_symmetries(
    psc_array.all_curves, currents, nfp=1, stellsym=False
)
B_PSC = BiotSavart(all_coils)
# Plot initial errors from only the PSCs, and then together with the TF coils
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_direct", B_axis)

from simsopt.field import BiotSavart, InterpolatedField
Bn = psc_array.Bn_PSC_full.reshape(qphi, 4 * ntheta)[:, :, None] * 1e-7 / B_axis
pointData = {"B_N": Bn}
s_plot.to_vtk(out_dir / "direct_Bn_PSC", extra_data=pointData)
# Bn = psc_array.b_opt.reshape(nphi, ntheta)[:, :, None] * 1e-7 / B_axis
# pointData = {"B_N": Bn}
# s.to_vtk(out_dir / "direct_Bn_TF", extra_data=pointData)

# Check SquaredFlux values using different ways to calculate it
x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
fB = SquaredFlux(s, bs, np.zeros((nphi, ntheta))).J()
print('fB only TF coils = ', fB / (B_axis ** 2 * s.area()))
bs.set_points(s.gamma().reshape(-1, 3))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
print('fB only TF direct = ', np.sum(Bnormal.reshape(-1) ** 2 * psc_array.grid_normalization ** 2
                                    ) / (2 * B_axis ** 2 * s.area()))
make_Bnormal_plots(B_PSC, s_plot, out_dir, "biot_savart_PSC_initial", B_axis)
fB = SquaredFlux(s, B_PSC, np.zeros((nphi, ntheta))).J()
print(fB/ (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, B_PSC + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, before opt = ', fB / (B_axis ** 2 * s.area()))
fB = SquaredFlux(s, B_PSC, -Bnormal).J()
print('fB with both (minus sign), before opt = ', fB / (B_axis ** 2 * s.area()))

# exit()
# Actually do the minimization now
from scipy.optimize import minimize
print('beginning optimization: ')
opt_bounds = tuple([(0, 2 * np.pi) for i in range(psc_array.num_psc * 2)])
options = {"disp": True, "maxiter": 100}
# print(opt_bounds)
# x0 = np.random.rand(2 * psc_array.num_psc) * 2 * np.pi
verbose = True

# Run STLSQ with BFGS in the loop
kwargs_manual = {
                 "out_dir": out_str, 
                 "plasma_boundary" : s,
                 "coils_TF" : coils
                 }
I_threshold = 0.0
STLSQ_max_iters = 10
for k in range(STLSQ_max_iters):
    # x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
    print('Number of PSCs = ', len(x0) // 2, ' in iteration ', k)
    x_opt = minimize(psc_array.least_squares, x0, args=(verbose,),
                     method='L-BFGS-B',
                     # bounds=opt_bounds,
                        # jac=psc_array.least_squares_jacobian, 
                     options=options,
                     tol=1e-10,
                     )
    from matplotlib import pyplot as plt
    plt.figure()
    plt.semilogy(psc_array.BdotN2_list)
    plt.show()
    I = psc_array.I
    small_I_inds = np.ravel(np.where(np.abs(I) < I_threshold))
    grid_xyz = psc_array.grid_xyz
    alphas = psc_array.alphas
    deltas = psc_array.deltas
    if len(small_I_inds) > 0:
        grid_xyz = grid_xyz[small_I_inds, :]
        alphas = alphas[small_I_inds]
        deltas = deltas[small_I_inds]
    else:
        print('STLSQ converged, breaking out of loop')
        break
    kwargs_manual["alphas"] = alphas
    kwargs_manual["deltas"] = deltas
    # Initialize new PSC array with coils only at the remaining locations
    # with initial orientations from the solve using BFGS
    psc_array = PSCgrid.geo_setup_manual(
        grid_xyz, psc_array.R, **kwargs_manual
    )
    
# psc_array.setup_orientations(x_opt.x[:len(x_opt) // 2], x_opt.x[len(x_opt) // 2:])
psc_array.setup_curves()
psc_array.plot_curves('final_')
currents = []
for i in range(psc_array.num_psc):
    currents.append(Current(psc_array.I[i]))
all_coils = coils_via_symmetries(
    psc_array.curves, currents, nfp=psc_array.nfp, stellsym=psc_array.stellsym
)
B_PSC = BiotSavart(all_coils)

# Check that direct Bn calculation agrees with optimization calculation
fB = SquaredFlux(s, B_PSC + bs, np.zeros((nphi, ntheta))).J()
print('fB with both, after opt = ', fB / (B_axis ** 2 * s.area()))
make_Bnormal_plots(B_PSC, s_plot, out_dir, "PSC_final", B_axis)
make_Bnormal_plots(bs + B_PSC, s_plot, out_dir, "PSC_and_TF_final", B_axis)
print('end')
