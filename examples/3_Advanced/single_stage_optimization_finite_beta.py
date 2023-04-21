#!/usr/bin/env python
import os
import time
import numpy as np
from mpi4py import MPI
from math import isnan
from pathlib import Path
from scipy.optimize import minimize
from simsopt.util import MpiPartition
from simsopt._core.optimizable import make_optimizable
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, VirtualCasing
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)
comm = MPI.COMM_WORLD


def pprint(*args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)


mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
start = time.time()
##########################################################################################
############## Input parameters
##########################################################################################
max_mode = 1
MAXITER_stage_2 = 50
MAXITER_single_stage = 20
vmec_input_filename = os.path.join(parent_path, 'inputs', 'input.QH_finitebeta')
ncoils = 3
aspect_ratio_target = 7.0
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
nphi_VMEC = 34
ntheta_VMEC = 34
vc_src_nphi = ntheta_VMEC
nmodes_coils = 7
coils_objective_weight = 1e+3
aspect_ratio_weight = 1
diff_method = "forward"
R0 = 1.0
R1 = 0.6
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 0
JACOBIAN_THRESHOLD = 100
LENGTH_CON_WEIGHT = 0.1  # Weight on the quadratic penalty for the curve length
LENGTH_WEIGHT = 1e-8  # Weight on the curve lengths in the objective function
CC_WEIGHT = 1e+0  # Weight for the coil-to-coil distance penalty in the objective function
CURVATURE_WEIGHT = 1e-3  # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-3  # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-9  # Weight for the arclength variation penalty in the objective function
##########################################################################################
##########################################################################################
directory = f'optimization_QH_finitebeta'
vmec_verbose = False
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
OUT_DIR = os.path.join(this_path, "output")
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm.rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
pprint(f' Using vmec input file {vmec_input_filename}')
vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary
##########################################################################################
##########################################################################################
# Finite Beta Virtual Casing Principle
vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
total_current_vmec = vmec.external_current() / (2 * surf.nfp)
##########################################################################################
##########################################################################################
#Stage 2
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
total_current = Current(total_current_vmec)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]
##########################################################################################
##########################################################################################
# Save initial surface and coil data
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
if comm.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)
##########################################################################################
##########################################################################################
Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])
JF = Jf + J_CC + J_LENGTH + J_LENGTH_PENALTY + J_CURVATURE + J_MSC
##########################################################################################
pprint(f'  Starting optimization')
# Initial stage 2 optimization


def fun_coils(dofss, info, oustr_dict=[]):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
        BdotN = np.mean(np.abs(BdotN_surf))
        BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
        print(outstr)
    return J, grad
##########################################################################################


def fun_J(dofs_vmec, dofs_coils):
    run_vcasing = False
    if np.sum(prob.x != dofs_vmec) > 0:
        prob.x = dofs_vmec
        run_vcasing = True
    J_stage_1 = prob.objective()
    dofs_coils = np.ravel(dofs_coils)
    if np.sum(JF.x != dofs_coils) > 0:
        JF.x = dofs_coils
    if run_vcasing:
        try:
            vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
            Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
            if np.sum(Jf.x != dofs_coils) > 0: Jf.x = dofs_coils
            JF.opts[0].opts[0].opts[0].opts[0].opts[0] = Jf
            if np.sum(JF.x != dofs_coils) > 0: JF.x = dofs_coils
        except Exception as e:
            J = JACOBIAN_THRESHOLD
            Jf = JF.opts[0].opts[0].opts[0].opts[0].opts[0]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    return J
##########################################################################################


def fun(dofss, prob_jacobian=None, info={'Nfeval': 0}, max_mode=1, oustr_dict=[]):
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)
    dofs_vmec = dofss[-number_vmec_dofs:]
    dofs_coils = dofss[:-number_vmec_dofs]
    J = fun_J(dofs_vmec, dofs_coils)
    if J > JACOBIAN_THRESHOLD or isnan(J):
        pprint(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(dofs_coils)
    else:
        pprint(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        grad_with_respect_to_surface = prob_jacobian.jac(dofs_vmec, dofs_coils)[0]
        fun_J(dofs_vmec, dofs_coils)
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return J, grad


#############################################################
## Perform optimization
#############################################################
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))
qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1)]
prob = LeastSquaresProblem.from_tuples(objective_tuple)
dofs = np.concatenate((JF.x, vmec.x))
bs.set_points(surf.gamma().reshape((-1, 3)))
vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
pprint(f"Aspect ratio before optimization: {vmec.aspect()}")
pprint(f"Mean iota before optimization: {vmec.mean_iota()}")
pprint(f"Quasisymmetry objective before optimization: {qs.total()}")
pprint(f"Magnetic well before optimization: {vmec.vacuum_well()}")
pprint(f"Squared flux before optimization: {Jf.J()}")
pprint(f'  Performing stage 2 optimization with {MAXITER_stage_2} iterations')
oustr_dict = []
res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}, oustr_dict), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
if comm.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_after_stage2"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)
pprint(f'  Performing single stage optimization with {MAXITER_single_stage} iterations')
dofs[:-number_vmec_dofs] = res.x
JF.x = dofs[:-number_vmec_dofs]
Jf = JF.opts[0].opts[0].opts[0].opts[0].opts[0]
mpi.comm_world.Bcast(dofs, root=0)
opt = make_optimizable(fun_J, dofs[-number_vmec_dofs:], dofs[:-number_vmec_dofs], dof_indicators=["dof", "non-dof"])
oustr_dict_inner = []
with MPIFiniteDifference(opt.J, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
    if mpi.proc0_world:
        res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}, max_mode, oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)
        dofs = res.x
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
if comm.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_opt"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, f'input.final'))
pprint(f"Aspect ratio after optimization: {vmec.aspect()}")
pprint(f"Mean iota after optimization: {vmec.mean_iota()}")
pprint(f"Quasisymmetry objective after optimization: {qs.total()}")
pprint(f"Magnetic well after optimization: {vmec.vacuum_well()}")
pprint(f"Squared flux after optimization: {Jf.J()}")