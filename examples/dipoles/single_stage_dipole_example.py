import os
import io
import sys
import numpy as np
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk
from simsopt.geo.surfaceobjectives import Volume, BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.field import BiotSavart, Coil, Current, coils_to_vtk, CurrentPenalty
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load, save
import matplotlib.pyplot as plt
from simsopt._core.derivative import derivative_dec
from helper_functions import *

# Create plot configuration
plot_config = PlotConfig(
    dpi=100,
    titlefontsize=16,
    axisfontsize=16,
    legendfontsize=14,
    ticklabelfontsize=14,
    cbarfontsize=16
)

class BoozerResidualExact(Optimizable):
    r"""
    This term returns the Boozer residual penalty term
    
    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.
    
    where
    
    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]
    
    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        # same number of points as on the solved surface
        nphis = in_surface.quadpoints_phi.size
        phis = np.linspace(0,1./in_surface.nfp,nphis*4,endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0,1,nthetas*4,endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        print("warning: constraint weight set to 0")
        self.constraint_weight = 0.0
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """
        
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
 
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True)
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)
        
        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil
        
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil
        
    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        
        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=True)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)
        
        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB

def initialize_boozer_surface(surf_prev, mpol, ntor, bs, vol_target, constraint_weight, iota, G0):
    """
    This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm
    
    surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    bs: simsopt.field.BiotSavart instance
    vol_target: target volume to be enclosed by the boozer surface
    constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    iota: initial guess for iota value on the surface
    G0: Value of net current going through the torus hole
    """
    surf = SurfaceXYZTensorFourier(
          mpol=mpol,ntor=ntor,nfp=surf_prev.nfp,stellsym=True,
          quadpoints_theta=surf_prev.quadpoints_theta,
          quadpoints_phi=surf_prev.quadpoints_phi
          )
    surf.least_squares_fit(surf_prev.gamma())

    if constraint_weight:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose':True})
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
              mpol=mpol,ntor=ntor,nfp=surf.nfp,stellsym=True,
              quadpoints_theta=np.linspace(0,1,2*mpol+1,endpoint=False),
              quadpoints_phi=np.linspace(0,1./surf.nfp,2*mpol+1,endpoint=False),
              dofs=surf.dofs
              )
    
        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose':True})

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)
    print(f"G0 from solve: {res['G']}")
    print(f"iota from solve: {res['iota']}")
    print(f"Volume from solve: {boozer_surface.surface.volume()}")

    # Check if boozer algo is successful
    success1 = res['success'] # True if the boozer surface algo converged
    success2 = not boozer_surface.surface.is_self_intersecting() # True if surface is not self intersecting
    success = success1 and success2
    if not success:
        raise RuntimeError("Something went wrong with the Boozer solve...")

    return boozer_surface

def normPlot(surf, bs, filename):
    # Plot the normal magnetic field on the plasma surface
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1./absn)[:,:,None]
    sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bfinal = bs.B().reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    mean_abs_relBfinal_norm = np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area)
    max_rnorm = np.max(np.abs(relBfinal_norm))
    relBfinal_norm = np.sum(bs.B().reshape((len(phi), len(theta), 3)) * surf.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((len(phi), len(theta), 3))**2, axis=2))[:, :, None]
    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBfinal_norm).T, levels=50, cmap='seismic', vmin=-max_rnorm, vmax=max_rnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_title(f'Surface-averaged \n |Bn|/|B| = {mean_abs_relBfinal_norm:.4e}', fontsize=18, fontweight='bold')
    plt.savefig(f"{filename}.png")
    plt.close()

def fun(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns the
    last accepted objective value with negated gradient to reject the step.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        J: Objective function value
        dJ: Gradient of objective function
    """
    dx = np.linalg.norm(x - run_dict['x_prev'])
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount']+=1

    # initialize to last accepted surface values
    boozer_surface.surface.x = run_dict['sdofs']
    boozer_surface.res['iota'] = run_dict['iota']
    boozer_surface.res['G'] = run_dict['G']

    # Set new coil dofs
    JF.x = x

    # Run boozer surface
    res = boozer_surface.run_code(run_dict['iota'], run_dict['G'])

    # Check success
    try:
        success1 = boozer_surface.res['success']
        success2 = not boozer_surface.surface.is_self_intersecting()
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2

    if success:
        J = JF.J()
        dJ = JF.dJ()

        print(f"Volume: {boozer_surface.surface.volume()}")
        print(f"Iota: {Iotas(boozer_surface).J()}")

    else:
        print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not success1:
            print("Boozer solver failed")
        if not success2:
            print("Surface is self-intersecting")

        J = run_dict['J']
        dJ = -run_dict['dJ']
        boozer_surface.surface.x = run_dict['sdofs']
        boozer_surface.res['iota'] = run_dict['iota']
        boozer_surface.res['G'] = run_dict['G']

    print(f"Objective J: {J:.6e}, ||∇J||: {np.linalg.norm(dJ):.6e}")
    return J, dJ

def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0

    # Store last accepted state
    run_dict['sdofs'] = boozer_surface.surface.x.copy()
    run_dict['iota'] = boozer_surface.res['iota']
    run_dict['G'] = boozer_surface.res['G']
    run_dict['J'] = JF.J()
    run_dict['dJ'] = JF.dJ().copy()

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']
    
    J_QS = JnonQSRatio.J()
    dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
    J_Boozer = JBoozerResidual.J()
    dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
    J_iota = Jiota.J()
    dJ_iota = np.linalg.norm(Jiota.dJ())
    J_curr = Jcurrent.J()
    dJ_curr = np.linalg.norm(Jcurrent.dJ())

    iota_str = f"{iota.J():.4f}"
    volume_str = f"{boozer_surface.surface.volume():.4f}"

    nphi = boozer_surface.surface.quadpoints_phi.size
    ntheta = boozer_surface.surface.quadpoints_theta.size
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))

    currents = np.array([abs(c.current.get_value()) for c in dipole_coils])
    num_over = np.sum(currents > CURRENT_THRESHOLD)
    max_current = np.max(currents)

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Current Penalty':{width}} = {J_curr:.6e} (dJ = {dJ_curr:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    print(f"{'Max current':{width}} = {max_current:.2f} A", file=buffer)
    print(f"{'# currents over threshold':{width}} = {num_over}", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    filename = OUT_DIR_ITER + "/log.txt"
    with open(filename, "a") as f:
        f.write(output_str + "\n")

    # Advance iteration counter
    run_dict['it'] += 1

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
mpol = 8
ntor = 8
INIT_DIR = "../single_stage_outputs/wout_nfp22ginsburg_000_000281/mpol=5-ntor=5_current_penalty_200000"

# This is created by and contains the results from stage 2, which we use to initialize single stage
results = load(os.path.join(INIT_DIR, 'results.json'))

# Optimization targets and weights
CONSTRAINT_WEIGHT = 1.0
MAXITER = 300
iota_target = 0.15

# Objective function weights and parameters
RES_WEIGHT = 1e3
IOTAS_WEIGHT = 1e2
CURRENT_THRESHOLD = 200000
CURRENT_WEIGHT = 1e-14

# Convergence tolerances for different mpol values
ftol_by_mpol = {5: 1e-8, 8: 5e-9, 10: 1e-6, 11: 5e-7, 12: 1e-7, 13: 5e-8, 14: 1e-8, 15: 5e-9, 16: 1e-9, 17: 5e-10, 18: 1e-10}
gtol_by_mpol = {5: 1e-8, 8: 5e-9, 10: 1e-3, 11: 5e-4, 12: 1e-4, 13: 5e-5, 14: 1e-5, 15: 5e-6, 16: 1e-6, 17: 5e-7, 18: 1e-7}
    
# Output directory setup
eq_name = results["eq_name"]
OUT_DIR = f"../single_stage_outputs/{eq_name}"
os.makedirs(OUT_DIR, exist_ok=True)
boozer_type = {'initial': 'least_squares', 'final': 'exact'}  # example
stage = 'initial'  # or 'final', depending on what you want

# ==============================================================================
# SURFACE GEOMETRY DEFINITIONS
# ==============================================================================
# Solely for visualization purposes
VV = SurfaceRZFourier(nfp=results["surf_nfp"])
VV.set_rc(0, 0, results["VV_R0"])
VV.set_rc(1, 0, results["VV_a"])
VV.set_zs(1, 0, results["VV_b"])

# ==============================================================================
# LOAD EQUILIBRIUM AND COILS
# ==============================================================================
print(f"\n===== Loading in equilibrium and coils =====")

bs = load(os.path.join(INIT_DIR, 'bs_opt.json'))

# Initialize the boundary magnetic surface and scale it to the same as stage 2
# Initialize the boundary magnetic surface and scale it to the same as stage 2
surf_opt_path = os.path.join(INIT_DIR, 'surf_opt.json')
if os.path.exists(surf_opt_path): # load from single-stage
    surf = load(surf_opt_path)
else: # load from stage 2
    eq_name_full = os.path.join(results["eq_dir"], results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_name_full, s=results["surf_s"], range="half period", nphi=results["plas_nPhi"], ntheta=results["plas_nTheta"]
    )
    surf.set_dofs(results["surf_dof_scale"] * surf.get_dofs())
vol_target = surf.volume()
print("Starting equilibrium = ", eq_name)
print(f"Target volume: {vol_target}")

# Extract coil information
num_tf_coils = results["ntf"] * 2 * results["surf_nfp"]  # ntf is the number of TF coils per half-period, so this is the total number
coils = bs.coils
curves = [c.curve for c in coils]
tf_coils = coils[:num_tf_coils]
tf_curves = [c.curve for c in tf_coils]
dipole_coils = coils[num_tf_coils:]
dipole_curves = [c.curve for c in dipole_coils]

# Just triple make sure they're fixed
for c in dipole_curves:
    c.fix_all()

print(f"# of TF coils: {len(tf_coils)}")
print(f"# of Dipole coils: {len(dipole_coils)}")

current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

# ==============================================================================
# OPTIMIZATION SETUP
# ==============================================================================
print(f"\n===== Starting single stage optimization for mpol = {mpol} and ntor = {ntor} =====")

OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}_current_penalty_{CURRENT_THRESHOLD}"
os.makedirs(OUT_DIR_ITER, exist_ok=True)

# Initialize Boozer surface with target parameters
boozer_surface = initialize_boozer_surface(surf, mpol, ntor, bs, vol_target, CONSTRAINT_WEIGHT, iota_target, G0)

# ==============================================================================
# SAVE INITIAL STATE
# ==============================================================================
# Save initial coil configurations
coils_to_vtk(coils, filename=OUT_DIR_ITER + "/coils_init", close=True)
bs.save(OUT_DIR_ITER + f"/bs_init.json")

# Save initial surface with magnetic field normal component data
pointData = {"B_N/B": np.sum(bs.B().reshape((results["plas_nPhi"], results["plas_nTheta"], 3)) *
    boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((results["plas_nPhi"], results["plas_nTheta"], 3))**2, axis=2))[:, :, None]}
boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_init", extra_data=pointData)
boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_init.json")
print(f"Volume: {boozer_surface.surface.volume()}")

# Generate initial diagnostic plots
normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotInitial")
plot_cross_section(boozer_surface.surface, VV, OUT_DIR_ITER, "BoozerCrossSectionInitial", plot_config)

# ==============================================================================
# DEFINE OBJECTIVE FUNCTION COMPONENTS
# ==============================================================================
# Biot-Savart field calculation
bs_obj = BiotSavart(coils)

# Quasi-symmetry and Boozer coordinate residuals
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
if boozer_type[stage]=='exact':
    brs = [BoozerResidualExact(boozer_surface, bs_obj)]
else:
    brs = [BoozerResidual(boozer_surface, bs_obj)]

# Individual objective terms
iota = Iotas(boozer_surface)

Jiota = QuadraticPenalty(iota, iota_target)
JnonQSRatio = sum(nonQSs)
JBoozerResidual = sum(brs)
Jcurrent = CurrentPenalty([c.current for c in dipole_coils], CURRENT_THRESHOLD)

# Combined objective function
JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiota + CURRENT_WEIGHT * Jcurrent

# Extract degrees of freedom
dofs = JF.x

# ==============================================================================
# INITIALIZE OPTIMIZATION STATE
# ==============================================================================
# Initialize run_dict after JF and boozer_surface are ready
run_dict = {
    'sdofs': boozer_surface.surface.x.copy(),
    'iota': boozer_surface.res['iota'],
    'G': boozer_surface.res['G'],
    'J': JF.J(),
    'dJ': JF.dJ().copy(),
    'it': 1,
    'lscount': 0,
    'x_prev': dofs.copy()
}

# ==============================================================================
# RUN OPTIMIZATION
# ==============================================================================
# Get convergence tolerances for current mpol
ftol = ftol_by_mpol.get(mpol)
gtol = gtol_by_mpol.get(mpol)

# Run L-BFGS-B optimization
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', callback=callback, options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': ftol, 'gtol': gtol})
print(res.message)

# ==============================================================================
# SAVE OPTIMIZED STATE AND PERFORM POSTPROCESSING
# ==============================================================================
# Save optimized coil configurations
coils_to_vtk(coils, filename=OUT_DIR_ITER + "/coils_opt", close=True)
bs.save(OUT_DIR_ITER + "/bs_opt.json")

# Save vacuum vessel for visualization
VV.to_vtk(os.path.join(OUT_DIR_ITER, "vacuum_vessel"))

# Save optimized surface with magnetic field normal component data
pointData = {"B_N/B": np.sum(bs.B().reshape((results["plas_nPhi"], results["plas_nTheta"], 3)) *
    boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((results["plas_nPhi"], results["plas_nTheta"], 3))**2, axis=2))[:, :, None]}

# Print final results
boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_opt", extra_data=pointData)
boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_opt.json")
print(f"Volume: {boozer_surface.surface.volume()}")
print(f"Iota: {Iotas(boozer_surface).J()}")

# Generate final diagnostic plots
normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotOptimized")
plot_cross_section(boozer_surface.surface, VV, OUT_DIR_ITER, "BoozerCrossSectionOptimized", plot_config)

# plots currents on surface
wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(dipole_coils, VV)
plot_coil_currents_on_theta_phi_grid(
    wp_currents_phis_thetas,
    OUT_DIR_ITER,
    plot_config,
)

# Save results dictionary for reproducibility and downstream use
results_output = {
    # Stage 2 directory
    "init_dir": INIT_DIR,

    # Optimization configuration
    "mpol": mpol,
    "ntor": ntor,
    "maxiter": MAXITER,
    "constraint_weight": CONSTRAINT_WEIGHT,
    "iota_target": iota_target,
    "res_weight": RES_WEIGHT,
    "iotas_weight": IOTAS_WEIGHT,
    "current_threshold": CURRENT_THRESHOLD,
    "current_weight": CURRENT_WEIGHT,
    
    # Convergence tolerances used
    "ftol": ftol_by_mpol.get(mpol),
    "gtol": gtol_by_mpol.get(mpol),
    
    # Optimization results
    "optimization_success": res.success,
    "optimization_message": res.message,
    "final_objective": JF.J(),
    "final_iota": float(Iotas(boozer_surface).J()),
    "final_volume": float(boozer_surface.surface.volume()),
    
    # Diagnostic metrics
    "nonQS_ratio": float(JnonQSRatio.J()),
    "boozer_residual": float(JBoozerResidual.J()),
    "iota_penalty": float(Jiota.J()),
    "current_penalty": float(Jcurrent.J()),
    "max_current": float(np.max([abs(c.current.get_value()) for c in dipole_coils])),
    "num_currents_over_threshold": int(np.sum([abs(c.current.get_value()) > CURRENT_THRESHOLD for c in dipole_coils])),
    
    # Coil information
    "num_tf_coils": len(tf_coils),
    "num_dipole_coils": len(dipole_coils),
    
    # Inherited from stage 2
    "eq_name": results["eq_name"],
    "eq_dir": results["eq_dir"],
    "surf_nfp": results["surf_nfp"],
    "surf_s": results["surf_s"],
    "plas_nPhi": results["plas_nPhi"],
    "plas_nTheta": results["plas_nTheta"],
    "VV_R0": results["VV_R0"],
    "VV_a": results["VV_a"],
    "VV_b": results["VV_b"],
    "ntf": results["ntf"],
}

# Save to file
save(results_output, os.path.join(OUT_DIR_ITER, 'results.json'))
print(f"Results saved to {os.path.join(OUT_DIR_ITER, 'results.json')}")