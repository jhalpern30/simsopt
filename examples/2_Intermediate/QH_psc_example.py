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

# np.random.seed(1)  # set a seed so that the same PSCs are initialized each time

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 4  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
else:
    # Resolution needs to be reasonably high if you are doing permanent magnets
    # or small coils because the fields are quite local
    nphi = 64  # nphi = ntheta >= 64 needed for accurate full-resolution runs
    ntheta = nphi
    # Make higher resolution surface for plotting Bnormal
    qphi = nphi * 4
    quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta * 4, endpoint=True)

poff = 1.0  # PSC grid will be offset 'poff' meters from the plasma surface
coff = 0.7  # PSC grid will be initialized between 1 m and 2 m from plasma

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
    surface_filename, range=range_param, nphi=nphi * 8, ntheta=ntheta * 8
)
s_outer = SurfaceRZFourier.from_vmec_input(
    surface_filename, range=range_param, nphi=nphi * 8, ntheta=ntheta * 84
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

def coil_optimization_QH(s, bs, base_curves, curves):
    from scipy.optimize import minimize
    from simsopt.geo import CurveLength, CurveCurveDistance, \
        MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
    from simsopt.objectives import QuadraticPenalty
    from simsopt.geo import curves_to_vtk
    from simsopt.objectives import SquaredFlux

    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    ncoils = len(base_curves)

    # Weight on the curve lengths in the objective function:
    LENGTH_WEIGHT = 1e-3

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 1

    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = 2.0
    CS_WEIGHT = 1e-1

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = 0.1
    CURVATURE_WEIGHT = 1e-9

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = 0.1
    MSC_WEIGHT = 1e-9

    MAXITER = 500  # number of iterations for minimize

    # Define the objective function:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Form the total objective function.
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)

    def fun(dofs):
        """ Function for coil optimization grabbed from stage_two_optimization.py """
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad

    print("""
    ################################################################################
    ### Perform a Taylor test ######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    # np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)

    J0, dJ0 = f(dofs)
    dJh = sum(dJ0 * h)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        J1, _ = f(dofs + eps*h)
        J2, _ = f(dofs - eps*h)
        print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    curves_to_vtk(curves, out_dir / "curves_opt")
    bs.set_points(s.gamma().reshape((-1, 3)))
    return bs

bs = coil_optimization_QH(s, bs, base_curves, curves)
curves_to_vtk(curves, out_dir / "TF_coils", close=True)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized")

# check after-optimization average on-axis magnetic field strength
B_axis = calculate_on_axis_B(bs, s)
# B_axis = 1.0  # Don't rescale
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_TF_optimized", B_axis)

# Finally, initialize the psc class
kwargs_geo = {"Nx": 13, "out_dir": out_str, "poff": poff,
               "plasma_boundary_full": s_plot} 
psc_array = PSCgrid.geo_setup_between_toroidal_surfaces(
    s, coils, s_inner, s_outer,  **kwargs_geo
)
L_orig = psc_array.L
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

# from simsopt.field import BiotSavart, InterpolatedField
# Bn = psc_array.Bn_PSC_full.reshape(qphi, 4 * ntheta)[:, :, None] * 1e-7 / B_axis
# pointData = {"B_N": Bn}
# s_plot.to_vtk(out_dir / "direct_Bn_PSC", extra_data=pointData)
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

from scipy.optimize import approx_fprime, check_grad, basinhopping, dual_annealing, direct, differential_evolution, OptimizeResult
# from scipy.optimize import lbfgsb
def callback(x):
    print('fB: ', psc_array.least_squares(x))
    print('approx: ', approx_fprime(x, psc_array.least_squares, 1E-3))
    print('exact: ', psc_array.least_squares_jacobian(x))
    print('-----')
    print(check_grad(psc_array.least_squares, psc_array.least_squares_jacobian, x) / np.linalg.norm(psc_array.least_squares_jacobian(x)))

# exit()
# Actually do the minimization now
from scipy.optimize import minimize
print('beginning optimization: ')
eps = 1e-6
# opt_bounds1 = tuple([(-np.pi / 2.0 + eps, np.pi / 2.0 - eps) for i in range(psc_array.num_psc)])
# opt_bounds2 = tuple([(-np.pi + eps, np.pi - eps) for i in range(psc_array.num_psc)])
# # print(opt_bounds1, opt_bounds2)
# opt_bounds = np.vstack((opt_bounds1, opt_bounds2))
# opt_bounds = tuple(map(tuple, opt_bounds))
options = {"disp": True, "maxiter": 50}
# print(opt_bounds)
# x0 = np.hstack(((np.random.rand(psc_array.num_psc) - 0.5) * np.pi, (np.random.rand(psc_array.num_psc) - 0.5) * 2 * np.pi))
# x0 = np.zeros(x0.shape)
verbose = True

# Run STLSQ with BFGS in the loop
kwargs_manual = {
                 "out_dir": out_str, 
                 "plasma_boundary" : s,
                 "coils_TF" : coils
                 }
I_threshold = 2e5
STLSQ_max_iters = 1
BdotN2_list = []
num_pscs = []
for k in range(STLSQ_max_iters):
    x0 = np.ravel(np.array([psc_array.alphas, psc_array.deltas]))
    num_pscs.append(len(x0) // 2)
    print('Number of PSCs = ', len(x0) // 2, ' in iteration ', k)
    opt_bounds1 = tuple([(-np.pi / 2.0 + eps, np.pi / 2.0 - eps) for i in range(psc_array.num_psc)])
    opt_bounds2 = tuple([(-np.pi + eps, np.pi - eps) for i in range(psc_array.num_psc)])
    opt_bounds = np.vstack((opt_bounds1, opt_bounds2))
    opt_bounds = tuple(map(tuple, opt_bounds))
    x_opt = minimize(psc_array.least_squares, 
                     x0, 
                     args=(verbose,),
                     method='L-BFGS-B',
                     bounds=opt_bounds,
                     jac=psc_array.least_squares_jacobian, 
                     options=options,
                     tol=1e-20,
                     # callback=callback
                     )
    I = psc_array.I
    grid_xyz = psc_array.grid_xyz
    alphas = psc_array.alphas
    deltas = psc_array.deltas
    if len(BdotN2_list) > 0:
        print(BdotN2_list, np.array(psc_array.BdotN2_list))
        BdotN2_list = np.hstack((BdotN2_list, np.array(psc_array.BdotN2_list)))
    else:
        BdotN2_list = np.array(psc_array.BdotN2_list)
    big_I_inds = np.ravel(np.where(np.abs(I) > I_threshold))
    if len(big_I_inds) != psc_array.num_psc:
        grid_xyz = grid_xyz[big_I_inds, :]
        alphas = alphas[big_I_inds]
        deltas = deltas[big_I_inds]
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
BdotN2_list = np.ravel(BdotN2_list)
L_final = psc_array.L
print('L1, L2 = ', L_orig, L_final)
    
from matplotlib import pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.semilogy(BdotN2_list)
plt.subplot(1, 2, 2)
plt.plot(num_pscs)
    
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

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
vmec_flag = True 
if vmec_flag:
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    from simsopt.util import comm_world
    mpi = MpiPartition(ngroups=4)
    comm = comm_world

    # # Make the QFM surfaces
    # t1 = time.time()
    # Bfield = bs + B_PSC
    # Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    # qfm_surf = make_qfm(s_plot, Bfield)
    # qfm_surf = qfm_surf.surface
    # t2 = time.time()
    # print("Making the QFM surface took ", t2 - t1, " s")

    # # Run VMEC with new QFM surface
    # t1 = time.time()

    # ### Always use the QA VMEC file and just change the boundary
    # vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    # equil = Vmec(vmec_input, mpi)
    # equil.boundary = qfm_surf
    # equil.run()
    
    from simsopt.field.magneticfieldclasses import InterpolatedField

    out_dir = Path(out_dir)
    n = 20
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 2  # 2 is sufficient sometimes
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    B_PSC.set_points(s_plot.gamma().reshape((-1, 3)))
    bsh = InterpolatedField(
        bs + B_PSC, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    from simsopt.field.tracing import compute_fieldlines, \
        plot_poincare_data, \
        IterationStoppingCriterion, SurfaceClassifier, \
        LevelsetStoppingCriterion
    from simsopt.util import proc0_print


    # set fieldline tracer parameters
    nfieldlines = 8
    tmax_fl = 10000

    R0 = np.linspace(16.5, 17.5, nfieldlines)  # np.linspace(s_plot.get_rc(0, 0) - s_plot.get_rc(1, 0) / 2.0, s_plot.get_rc(0, 0) + s_plot.get_rc(1, 0) / 2.0, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s_plot.nfp) for i in range(4)]
    print(rrange, zrange, phirange)
    print(R0, Z0)

    t1 = time.time()
    # compute the fieldlines from the initial locations specified above
    sc_fieldline = SurfaceClassifier(s_plot, h=0.03, p=2)
    sc_fieldline.to_vtk(str(out_dir) + 'levelset', h=0.02)

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bsh, R0, Z0, tmax=tmax_fl, tol=1e-20, comm=comm,
        phis=phis,
        # phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        stopping_criteria=[IterationStoppingCriterion(50000)])
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    # make the poincare plots
    if comm is None or comm.rank == 0:
        plot_poincare_data(fieldlines_phi_hits, phis, out_dir / 'poincare_fieldline.png', dpi=100, surf=s_plot)

plt.show()
# t_end = time.time()
# print('Total time = ', t_end - t_start)
print('end')
