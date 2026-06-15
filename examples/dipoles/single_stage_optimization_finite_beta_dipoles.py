#!/usr/bin/env python3
r"""
In this example we both a stage 1 and stage 2 optimization problems
using the generalization to finite beta of the
single stage approach of R. Jorge et al in https://arxiv.org/abs/2302.10622
The objective function in this case is J = J_stage1 + coils_objective_weight*J_stage2
To accelerate convergence, a stage 2 optimization is done before the single stage one.
This script requires the VirtualCasing module to be installed.
Rogerio Jorge, April 2023
"""
import os
import numpy as np
from math import isnan
from pathlib import Path
from scipy.optimize import minimize
from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt._core.util import ObjectiveFailure
from simsopt._core.optimizable import load
from simsopt import make_optimizable
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, VirtualCasing
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.geo import SurfaceRZFourier
from helper_functions import *

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)

mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
##########################################################################################
############## Input parameters
##########################################################################################
MAXITER_stage_2 = 300
MAXITER_single_stage = 50
max_mode = 6 # maximum poloidal and toroidal modes on the surface being optimized with VMEC

vacuum_dir = "../single_stage_scans_epsilon_constraint_updated/wout_nfp22ginsburg_000_000281_init_dir90/iota_tar0.15/stage02_cw1/mpol6_ntor6"
bs = load(os.path.join(vacuum_dir, "bs_opt.json"))
surf = load(os.path.join(vacuum_dir, "surf_opt.json"))
results = load(os.path.join(vacuum_dir, "results.json"))
tf = ToroidalFlux(surf, bs)

VV = SurfaceRZFourier(nfp=results["surf_nfp"])
VV.set_rc(0, 0, results["VV_R0"])
VV.set_rc(1, 0, results["VV_a"])
VV.set_zs(1, 0, results["VV_b"])

# Initial vmec input file - take boozer surface and fit to boundar and add HBT profiles
vmec_input_filename = os.path.join(parent_path, "equilibria/input.hbt_finite_beta_test")

nphi_VMEC = 128
ntheta_VMEC = 64
vc_src_nphi = ntheta_VMEC
coils_objective_weight = 1e+3
aspect_ratio_weight = 1
diff_method = "forward"
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 0
JACOBIAN_THRESHOLD = 100
CP_WEIGHT = 50
CP_THRESHOLD = 100000
##########################################################################################
##########################################################################################
directory = '../single_stage_finite_beta'
vmec_verbose = False
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Load in VMEC input file with HBT profiles and set surface data
proc0_print(f' Using vmec input file {vmec_input_filename}')
proc0_print(f' Loading in surface from {vacuum_dir}')
vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
vmec.boundary = surf # Boozer surface boundary
vmec.indata.phiedge = tf.J() # Toroidal flux
vmec.indata.mpol = max_mode
vmec.indata.ntor = max_mode
vmec.indata.nzeta = 4 * max_mode
vmec.indata.ntheta = 4 * max_mode
vmec.indata.nfp = surf.nfp
proc0_print(f"Toroidal flux: {tf.J()}")

##########################################################################################
##########################################################################################
#Stage 2
coils = bs.coils
ntf = 4 * 2 * surf.nfp
tf_coils = coils[0:ntf]
dipole_coils = coils[ntf:]

##########################################################################################
proc0_print('  Starting optimization')
##########################################################################################
# Initial stage 2 optimization
##########################################################################################

## The function fun_coils defined below is used to only optimize the coils at the beginning
## and then optimize the coils and the surface together. This makes the overall optimization
## more efficient as the number of iterations needed to achieve a good solution is reduced.
def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    # BiotSavart caches can become inconsistent after updating coil DOFs; ensure
    # the field is evaluated on the same surface quadrature grid each call.
    bs.set_points(surf.gamma().reshape((-1, 3)))
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
        BdotN = np.mean(np.abs(BdotN_surf))
        # BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, max I = {np.max([c.current.get_value() for c in dipole_coils])}"
        print(outstr)
    return J, grad

##########################################################################################
##########################################################################################

## The function fun_J defined below is used to calculate the objective function value
## from the coils and the surface using the virtual casing principle. As this function
## is used also to calculate the derivatives using finite difference, it was separated
## from the function fun below.
def fun_J(prob, coils_prob):
    global previous_surf_dofs
    J_stage_1 = prob.objective()
    if np.any(previous_surf_dofs != prob.x):  # Only run virtual casing if surface dofs have changed
        previous_surf_dofs = prob.x
        try:
            vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
            Jf.target = vc.B_external_normal
        except ObjectiveFailure:
            pass

    bs.set_points(surf.gamma().reshape((-1, 3)))
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    return J

##########################################################################################
##########################################################################################
## The function fun defined below is used to optimize the coils and the surface together.
def fun(dofss, prob_jacobian, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)
    prob.x = dofss[-number_vmec_dofs:]
    coil_dofs = dofss[:-number_vmec_dofs]
    # Un-fix the desired coil dofs so they can be updated:
    JF.full_unfix(free_coil_dofs)
    JF.x = coil_dofs
    J = fun_J(prob, JF)
    if J > JACOBIAN_THRESHOLD or isnan(J):
        proc0_print(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(coil_dofs)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        JF.fix_all()  # Must re-fix the coil dofs before beginning the finite differencing.
        grad_with_respect_to_surface = prob_jacobian.jac(prob.x)[0]

    JF.fix_all()
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))

    return J, grad

##########################################################################################
#############################################################
## Perform optimization
#############################################################
##########################################################################################
# Switch to SurfaceRZFourier since SurfaceXYZTensorFourier doesn't have the correct dof access
surf = SurfaceRZFourier.from_wout("wout_hbt_finite_beta_000_000000.nc", nphi=nphi_VMEC, ntheta=ntheta_VMEC)
# surf.fix_all()
# surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(vmec.x))
qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=0)
aspect_ratio_target = vmec.aspect()
aspect_ratio_weight = 1
objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1)]
prob = LeastSquaresProblem.from_tuples(objective_tuple)
previous_surf_dofs = prob.x
bs.set_points(surf.gamma().reshape((-1, 3)))
vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
total_current_vmec = vmec.external_current() / (2 * surf.nfp)
proc0_print(f"Total current: {total_current_vmec}")
Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
JF = Jf
dofs = np.concatenate((JF.x, vmec.x))

proc0_print(f"Aspect ratio before optimization: {vmec.aspect()}")
proc0_print(f"Mean iota before optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective before optimization: {qs.total()}")
proc0_print(f"Magnetic well before optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux before optimization: {Jf.J()}")
proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')

if comm_world.rank == 0:
    plot_relBfinal_norm_modB(bs, vmec.boundary, coils_results_path, "initial", plot_config, vc = vc)
    plot_cross_section(
        surf, VV, coils_results_path, "initial", plot_config,
        base_dipole_coils=dipole_coils,
    )
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, coils_results_path, "initial", plot_config)

res = minimize(fun_coils, JF.x, jac=True, args=({'Nfeval': 0}), method='BFGS', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
if comm_world.rank == 0:
    plot_relBfinal_norm_modB(bs, vmec.boundary, coils_results_path, "stage2", plot_config, vc = vc)
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, coils_results_path, "stage2", plot_config)

proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
dofs[:-number_vmec_dofs] = res.x
JF.x = dofs[:-number_vmec_dofs]
mpi.comm_world.Bcast(dofs, root=0)
opt = make_optimizable(fun_J, prob, JF)
free_coil_dofs = JF.dofs_free_status
JF.fix_all()

with MPIFiniteDifference(
    opt.J,
    mpi,
    diff_method=diff_method,
    abs_step=finite_difference_abs_step,
    rel_step=finite_difference_rel_step,
) as prob_jacobian:
    if mpi.proc0_world:
        res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)

Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
if comm_world.rank == 0:
    plot_relBfinal_norm_modB(bs, vmec.boundary, coils_results_path, "single_stage", plot_config, vc = vc)
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, coils_results_path, "single_stage", plot_config)
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_opt"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, 'input.final'))
proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
proc0_print(f"Magnetic well after optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux after optimization: {Jf.J()}")
JF.full_unfix(free_coil_dofs)  # Needed to evaluate JF.dJ
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
BdotN = np.mean(np.abs(BdotN_surf))
BdotNmax = np.max(np.abs(BdotN_surf))
outstr = f"Coil parameters: ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
proc0_print(outstr)
