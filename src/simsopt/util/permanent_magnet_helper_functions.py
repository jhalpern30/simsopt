"""
This module contains the a number of useful functions for using 
the permanent magnets functionality in the SIMSOPT code.
"""
__all__ = ['read_focus_coils', 'coil_optimization', 
           'trace_fieldlines', 'make_qfm', 
           'initialize_coils', 'calculate_on_axis_B',
           'make_optimization_plots', 'run_Poincare_plots',
           'make_curve_at_theta0', 'make_filament_from_voxels',
           'make_Bnormal_plots', 'initialize_default_kwargs',
           'perform_filament_optimization'
           ]

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LpCurveTorsion
from simsopt.objectives import Weight, QuadraticPenalty
from simsopt.objectives import SquaredFlux
from pathlib import Path


def read_focus_coils(filename):
    """
    Reads in the coils from a FOCUS file. For instance, this is
    used for loading in the MUSE phased TF coils.

    Args:
        filename: String denoting the name of the coils file. 

    Returns:
        coils: List of CurveXYZFourier class objects.
        base_currents: List of Current class objects.
        ncoils: Integer representing the number of coils.
    """
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import Current

    ncoils = np.loadtxt(filename, skiprows=1, max_rows=1, dtype=int)
    order = int(np.loadtxt(filename, skiprows=8, max_rows=1, dtype=int))
    coilcurrents = np.zeros(ncoils)
    xc = np.zeros((ncoils, order + 1))
    xs = np.zeros((ncoils, order + 1))
    yc = np.zeros((ncoils, order + 1))
    ys = np.zeros((ncoils, order + 1))
    zc = np.zeros((ncoils, order + 1))
    zs = np.zeros((ncoils, order + 1))
    # load in coil currents and fourier representations of (x, y, z)
    for i in range(ncoils):
        coilcurrents[i] = np.loadtxt(filename, skiprows=6 + 14 * i, max_rows=1, usecols=1)
        xc[i, :] = np.loadtxt(filename, skiprows=10 + 14 * i, max_rows=1, usecols=range(order + 1))
        xs[i, :] = np.loadtxt(filename, skiprows=11 + 14 * i, max_rows=1, usecols=range(order + 1))
        yc[i, :] = np.loadtxt(filename, skiprows=12 + 14 * i, max_rows=1, usecols=range(order + 1))
        ys[i, :] = np.loadtxt(filename, skiprows=13 + 14 * i, max_rows=1, usecols=range(order + 1))
        zc[i, :] = np.loadtxt(filename, skiprows=14 + 14 * i, max_rows=1, usecols=range(order + 1))
        zs[i, :] = np.loadtxt(filename, skiprows=15 + 14 * i, max_rows=1, usecols=range(order + 1))

    # CurveXYZFourier wants data in order sin_x, cos_x, sin_y, cos_y, ...
    coil_data = np.zeros((order + 1, ncoils * 6))
    for i in range(ncoils):
        coil_data[:, i * 6 + 0] = xs[i, :]
        coil_data[:, i * 6 + 1] = xc[i, :]
        coil_data[:, i * 6 + 2] = ys[i, :]
        coil_data[:, i * 6 + 3] = yc[i, :]
        coil_data[:, i * 6 + 4] = zs[i, :]
        coil_data[:, i * 6 + 5] = zc[i, :]

    # Set the degrees of freedom in the coil objects
    base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
    ppp = 20
    coils = [CurveXYZFourier(order*ppp, order) for i in range(ncoils)]
    for ic in range(ncoils):
        dofs = coils[ic].dofs_matrix
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(order, coil_data.shape[0]-1)):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return coils, base_currents, ncoils


def coil_optimization(s, bs, base_curves, curves, out_dir=''):
    """
    Optimize the coils for the QA, QH, or other configurations.

    Args:
        s: plasma boundary.
        bs: Biot Savart class object, presumably representing the
          magnetic fields generated by the coils.
        base_curves: List of CurveXYZFourier class objects.
        curves: List of Curve class objects.
        out_dir: Path or string for the output directory for saved files.

    Returns:
        bs: Biot Savart class object, presumably representing the
          OPTIMIZED magnetic fields generated by the coils.
    """

    from simsopt.geo import CurveLength, CurveCurveDistance, \
        MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance
    from simsopt.objectives import QuadraticPenalty
    from simsopt.geo import curves_to_vtk
    from simsopt.objectives import SquaredFlux

    out_dir = Path(out_dir)
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    ncoils = len(base_curves)

    # Weight on the curve lengths in the objective function:
    LENGTH_WEIGHT = 1e-4

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 1e-1

    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = 0.1
    CS_WEIGHT = 1e-2

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
    np.random.seed(1)
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
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    curves_to_vtk(curves, out_dir / "curves_opt")
    bs.set_points(s.gamma().reshape((-1, 3)))
    return bs


def trace_fieldlines(bfield, label, s, comm, R0, out_dir=''):
    """
    Make Poincare plots on a surface as in the trace_fieldlines
    example in the examples/1_Simple/ directory.

    Args:
        bfield: MagneticField or InterpolatedField class object.
        label: Name of the file to write to.
        s: plasma boundary surface.
        comm: MPI COMM_WORLD object for using MPI for tracing.
        R0: 1D numpy array containing the cylindrical R locations
          that should be sampled by the field tracing (usually the
          points are from the magnetic axis to the plasma boundary
          at the phi = 0 plane).
        out_dir: Path or string for the output directory for saved files.
    """
    from simsopt.field.tracing import particles_to_vtk, compute_fieldlines, \
        LevelsetStoppingCriterion, plot_poincare_data, \
        IterationStoppingCriterion, SurfaceClassifier

    out_dir = Path(out_dir)

    # set fieldline tracer parameters
    nfieldlines = len(R0)
    tmax_fl = 60000

    Z0 = np.zeros(nfieldlines)
    print('R0s = ', R0)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]

    # compute the fieldlines from the initial locations specified above
    sc_fieldline = SurfaceClassifier(s, h=0.01, p=2)
    sc_fieldline.to_vtk(str(out_dir) + 'levelset', h=0.02)

    # See field tracing example in examples/1_Simple for description
    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])

    # make the poincare plots
    if comm is None or comm.rank == 0:
        plot_poincare_data(fieldlines_phi_hits, phis, out_dir / f'poincare_fieldline_{label}.png', dpi=100, surf=s, ylims=(-0.375, 0.375), xlims=(0.575, 1.35))


def make_qfm(s, Bfield):
    """
    Given some Bfield generated by dipoles AND a set of TF coils, 
    compute a quadratic flux-minimizing surface (QFMS)
    for the total field configuration on the surface s.

    Args:
        s: plasma boundary surface.
        Bfield: MagneticField or inherited MagneticField class object.

    Returns:
        qfm_surface: The identified QfmSurface class object.
    """
    from simsopt.geo.qfmsurface import QfmSurface
    from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume

    # weight for the optimization
    constraint_weight = 1e0

    # First optimize at fixed volume
    qfm = QfmResidual(s, Bfield)
    qfm.J()

    # s.change_resolution(16, 16)
    vol = Volume(s)
    vol_target = vol.J()
    qfm_surface = QfmSurface(Bfield, s, vol, vol_target)

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=50,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    # repeat the optimization for further convergence
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=200,
                                                             constraint_weight=constraint_weight)
    print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    return qfm_surface


def initialize_coils(config_flag, TEST_DIR, s, out_dir=''):
    """
    Initializes coils for each of the target configurations that are
    used for permanent magnet optimization.

    Args:
        config_flag: String denoting the stellarator configuration 
          being initialized.
        TEST_DIR: String denoting where to find the input files.
        out_dir: Path or string for the output directory for saved files.
        s: plasma boundary surface.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk

    out_dir = Path(out_dir)
    if 'muse' in config_flag:
        # Load in pre-optimized coils
        coils_filename = TEST_DIR / 'muse_tf_coils.focus'
        base_curves, base_currents, ncoils = read_focus_coils(coils_filename)
        coils = []
        for i in range(ncoils):
            coils.append(Coil(base_curves[i], base_currents[i]))
        base_currents[0].fix_all()

        # fix all the coil shapes
        for i in range(ncoils):
            base_curves[i].fix_all()
    elif config_flag == 'qh':
        # generate planar TF coils
        ncoils = 4
        R0 = s.get_rc(0, 0)
        R1 = s.get_rc(1, 0) * 2
        order = 5

        # qh needs to be scaled to 0.1 T on-axis magnetic field strength
        from simsopt.mhd.vmec import Vmec
        vmec_file = 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc'
        total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 8.75 / 5.69674966667
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()
    elif config_flag == 'qa':
        # generate planar TF coils
        ncoils = 8
        R0 = 1.0
        R1 = 0.65
        order = 5

        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        from simsopt.mhd.vmec import Vmec
        vmec_file = 'wout_LandremanPaul2021_QA_lowres.nc'
        total_current = Vmec(TEST_DIR / vmec_file).external_current() / (2 * s.nfp) / 7.131
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir / "curves_init")
    return base_curves, curves, coils


def calculate_on_axis_B(bs, s):
    """
    Check the average, approximate, on-axis
    magnetic field strength to make sure the
    configuration is scaled correctly.

    Args:
        bs: MagneticField or BiotSavart class object.
        s: plasma boundary surface.

    Returns:
        B0avg: Average magnetic field strength along 
          the major radius of the device.
    """
    nphi = len(s.quadpoints_phi)
    bspoints = np.zeros((nphi, 3))

    # rescale phi from [0, 1) to [0, 2 * pi)
    phi = s.quadpoints_phi * 2 * np.pi

    R0 = s.get_rc(0, 0)
    for i in range(nphi):
        bspoints[i] = np.array([R0 * np.cos(phi[i]),
                                R0 * np.sin(phi[i]),
                                0.0]
                               )
    bs.set_points(bspoints)
    B0 = np.linalg.norm(bs.B(), axis=-1)
    B0avg = np.mean(np.linalg.norm(bs.B(), axis=-1))
    surface_area = s.area()
    bnormalization = B0avg * surface_area
    print("Bmag at R = ", R0, ", Z = 0: ", B0)
    print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg)
    return B0avg


def make_optimization_plots(RS_history, m_history, m_proxy_history, pm_opt, out_dir=''):
    """
    Make line plots of the algorithm convergence and make histograms
    of m, m_proxy, and if available, the FAMUS solution.

    Args:
        RS_history: List of the relax-and-split optimization progress.
        m_history: List of the permanent magnet solutions generated
          as the relax-and-split optimization progresses.
        m_proxy_history: Same but for the 'proxy' variable in the
          relax-and-split optimization.
        pm_opt: PermanentMagnetGrid class object that was optimized.
        out_dir: Path or string for the output directory for saved files.
    """
    out_dir = Path(out_dir)

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(out_dir / 'RS_objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    m0_abs = np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    mproxy_abs = np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima

    # get FAMUS rho values for making comparison histograms
    if hasattr(pm_opt, 'famus_filename'):
        m0, p = np.loadtxt(
            pm_opt.famus_filename, skiprows=3,
            usecols=[7, 8],
            delimiter=',', unpack=True
        )
        # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
        momentq = np.loadtxt(pm_opt.famus_filename, skiprows=1, max_rows=1, usecols=[1])
        rho = p ** momentq
        rho = rho[pm_opt.Ic_inds]
        x_multi = [m0_abs, mproxy_abs, abs(rho)]
    else:
        x_multi = [m0_abs, mproxy_abs]
        rho = None
    plt.hist(x_multi, bins=np.linspace(0, 1, 40), log=True, histtype='bar')
    plt.grid(True)
    plt.legend(['m', 'w', 'FAMUS'])
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(out_dir / 'm_histograms.png')

    # If relax-and-split was used with l0 norm,
    # make a nice histogram of the algorithm progress
    if len(RS_history) != 0:

        m_history = np.array(m_history)
        m_history = m_history.reshape(m_history.shape[0] * m_history.shape[1], pm_opt.ndipoles, 3)
        m_proxy_history = np.array(m_proxy_history).reshape(m_history.shape[0], pm_opt.ndipoles, 3)

        for i, datum in enumerate([m_history, m_proxy_history]):
            # Code from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
            def prepare_animation(bar_container):
                def animate(frame_number):
                    plt.title(frame_number)
                    data = np.sqrt(np.sum(datum[frame_number, :] ** 2, axis=-1)) / pm_opt.m_maxima
                    n, _ = np.histogram(data, np.linspace(0, 1, 40))
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    return bar_container.patches
                return animate

            # make histogram animation of the dipoles at each relax-and-split save
            fig, ax = plt.subplots()
            data = np.sqrt(np.sum(datum[0, :] ** 2, axis=-1)) / pm_opt.m_maxima
            if rho is not None:
                ax.hist(abs(rho), bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='g')
            _, _, bar_container = ax.hist(data, bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='r')
            ax.set_ylim(top=1e5)  # set safe limit to ensure that all data is visible.
            plt.grid(True)
            if rho is not None:
                plt.legend(['FAMUS', np.array([r'm$^*$', r'w$^*$'])[i]])
            else:
                plt.legend([np.array([r'm$^*$', r'w$^*$'])[i]])

            plt.xlabel('Normalized magnitudes')
            plt.ylabel('Number of dipoles')
            ani = animation.FuncAnimation(
                fig, prepare_animation(bar_container),
                range(0, m_history.shape[0], 2),
                repeat=False, blit=True
            )
            # ani.save(out_dir / 'm_history' + str(i) + '.mp4')


def run_Poincare_plots(s_plot, bs, b_dipole, comm, filename_poincare, R0, out_dir=''):
    """
    Wrapper function for making Poincare plots.

    Args:
        s_plot: plasma boundary surface with range = 'full torus'.
        bs: MagneticField class object.
        b_dipole: DipoleField class object.
        comm: MPI COMM_WORLD object for using MPI for tracing.
        filename_poincare: Filename for the output poincare picture.
        R0: 1D numpy array containing the cylindrical R locations
          that should be sampled by the field tracing (usually the
          points are from the magnetic axis to the plasma boundary
          at the phi = 0 plane).
        out_dir: Path or string for the output directory for saved files.
    """
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.objectives import SquaredFlux

    out_dir = Path(out_dir)

    n = 20
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    r_margin = 0.05
    rrange = (np.min(rs) - r_margin, np.max(rs) + r_margin, n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 4  # 2 is sufficient sometimes
    nphi = len(s_plot.quadpoints_phi)
    ntheta = len(s_plot.quadpoints_theta)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    Bnormal_dipole = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    f_B = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_pre_poincare_check")
    make_Bnormal_plots(b_dipole, s_plot, out_dir, "dipole_pre_poincare_check")
    make_Bnormal_plots(bs + b_dipole, s_plot, out_dir, "total_pre_poincare_check")

    bsh = InterpolatedField(
        bs + b_dipole, degree, rrange, phirange, zrange, True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    trace_fieldlines(bsh, 'bsh_PMs_' + filename_poincare, s_plot, comm, out_dir, R0)


def make_Bnormal_plots(bs, s_plot, out_dir='', bs_filename="Bnormal"):
    """
    Plot Bnormal on plasma surface from a MagneticField object.
    Do this quite a bit in the permanent magnet optimization
    and initialization so this is a wrapper function to reduce
    the amount of code.

    Args:
        bs: MagneticField class object.
        s_plot: plasma boundary surface with range = 'full torus'.
        out_dir: Path or string for the output directory for saved files.
        bs_filename: String denoting the name of the output file. 
    """
    out_dir = Path(out_dir)
    nphi = len(s_plot.quadpoints_phi)
    ntheta = len(s_plot.quadpoints_theta)
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
    s_plot.to_vtk(out_dir / bs_filename, extra_data=pointData)


def initialize_default_kwargs(algorithm='RS'):
    """
    Keywords to the permanent magnet optimizers are now passed
    by the kwargs dictionary. Default dictionaries are initialized
    here.

    Args:
        algorithm: String denoting which algorithm is being used
          for permanent magnet optimization. Options are 'RS'
          (relax and split), 'GPMO' (greedy placement), 
          'backtracking' (GPMO with backtracking), 
          'multi' (GPMO, placing multiple magnets each iteration),
          'ArbVec' (GPMO with arbitrary dipole orientiations),
          'ArbVec_backtracking' (GPMO with backtracking and 
          arbitrary dipole orientations).

    Returns:
        kwargs: Dictionary of keywords to pass to a permanent magnet 
          optimizer.
    """

    kwargs = {}
    kwargs['verbose'] = True   # print out errors every few iterations
    if algorithm == 'RS':
        kwargs['nu'] = 1e100  # Strength of the "relaxation" part of relax-and-split
        kwargs['max_iter'] = 100  # Number of iterations to take in a convex step
        kwargs['reg_l0'] = 0.0
        kwargs['reg_l1'] = 0.0
        kwargs['alpha'] = 0.0
        kwargs['min_fb'] = 0.0
        kwargs['epsilon'] = 1e-3
        kwargs['epsilon_RS'] = 1e-3
        kwargs['max_iter_RS'] = 2  # Number of total iterations of the relax-and-split algorithm
        kwargs['reg_l2'] = 0.0
    elif 'GPMO' in algorithm or 'ArbVec' in algorithm:
        kwargs['K'] = 1000
        kwargs["reg_l2"] = 0.0 
        kwargs['nhistory'] = 500  # K > nhistory and nhistory must be divisor of K
    return kwargs


def make_curve_at_theta0(s, numquadpoints):
    from simsopt.geo import CurveRZFourier

    order = s.ntor + 1
    quadpoints = np.linspace(0, 1, numquadpoints, endpoint=True)
    curve = CurveRZFourier(quadpoints, order, nfp=s.nfp, stellsym=True)
    r_mn = np.zeros((s.mpol + 1, 2 * s.ntor + 1))
    z_mn = np.zeros((s.mpol + 1, 2 * s.ntor + 1))
    for m in range(s.mpol + 1):
        if m == 0:
            nmin = 0
        else: 
            nmin = -s.ntor
        for n in range(nmin, s.ntor + 1):
            r_mn[m, n + s.ntor] = s.get_rc(m, n)
            z_mn[m, n + s.ntor] = s.get_zs(m, n)
    r_n = np.sum(r_mn, axis=0)
    z_n = np.sum(z_mn, axis=0)
    for n in range(s.ntor + 1):
        if n == 0:
            curve.rc[n] = r_n[n + s.ntor]
        else:
            curve.rc[n] = r_n[n + s.ntor] + r_n[-n + s.ntor]
            curve.zs[n - 1] = -z_n[n + s.ntor] + z_n[-n + s.ntor]

    curve.x = curve.get_dofs()
    curve.x = curve.x  # need to do this to transfer data to C++
    return curve


def make_filament_from_voxels(current_voxels_grid, final_threshold, truncate=False, num_fourier=16):
    """
    Take an optimized CurrentVoxelGrid class object and the largest
    threshold used to optimize it, and produce a filamentary curve from
    the solution. This function assumes that the current voxel solution
    is curve-like, i.e. that it makes sense to try and fit this solution
    with a Fourier series.
    """
    from simsopt.geo import CurveXYZFourier

    # Find where voxels are nonzero
    alphas = current_voxels_grid.alphas.reshape(current_voxels_grid.N_grid, current_voxels_grid.n_functions)
    nonzero_inds = (np.linalg.norm(alphas, axis=-1) > final_threshold)
    xyz_curve = current_voxels_grid.XYZ_flat[nonzero_inds, :]
    nfp = current_voxels_grid.plasma_boundary.nfp
    stellsym = current_voxels_grid.plasma_boundary.stellsym
    n = xyz_curve.shape[0]

    # complete the curve via the symmetries 
    xyz_total = np.zeros((n * nfp * (stellsym + 1), 3)) 
    if stellsym:
        stell_list = [1, -1]
    else:
        stell_list = [1]
    index = 0
    for stell in stell_list:
        for fp in range(nfp):
            phi0 = (2 * np.pi / nfp) * fp
            xyz_total[index:index + n, 0] = xyz_curve[:, 0] * np.cos(phi0) - xyz_curve[:, 1] * np.sin(phi0) * stell 
            xyz_total[index:index + n, 1] = xyz_curve[:, 0] * np.sin(phi0) + xyz_curve[:, 1] * np.cos(phi0) * stell 
            xyz_total[index:index + n, 2] = xyz_curve[:, 2] * stell 
            index += n

    # number of fourier modes to use for the fourier expansion in each coordinate
    ax = plt.figure().add_subplot(projection='3d')
    xyz_curve = xyz_total
    plt.plot(xyz_curve[:, 0], xyz_curve[:, 1], xyz_curve[:, 2])

    # Let's assume curve is unique w/ respect to theta or phi
    # in the 1 / 2 * nfp part of the surface
    phi, unique_inds = np.unique(np.arctan2(xyz_curve[:, 1], xyz_curve[:, 0]), return_index=True)
    xyz_curve = xyz_curve[unique_inds, :]
    plt.plot(xyz_curve[:, 0], xyz_curve[:, 1], xyz_curve[:, 2])

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    x_curve = moving_average(xyz_curve[:, 0], 10)
    y_curve = moving_average(xyz_curve[:, 1], 10)
    z_curve = moving_average(xyz_curve[:, 2], 10)

    # Stitch together missing points lost during the moving average
    xn = len(x_curve)
    extra_x = -x_curve[xn // 2:xn // 2 + 6]
    extra_y = -y_curve[xn // 2:xn // 2 + 6]
    extra_z = z_curve[xn // 2:xn // 2 + 6]
    x_curve = np.concatenate((extra_x, x_curve))
    y_curve = np.concatenate((extra_y, y_curve))
    z_curve = np.concatenate((extra_z, z_curve))
    x_curve = np.append(x_curve, np.flip(extra_x))
    y_curve = np.append(y_curve, -np.flip(extra_y))
    z_curve = np.append(z_curve, -np.flip(extra_z))
    xyz_curve = np.transpose([x_curve, y_curve, z_curve])
    plt.plot(xyz_curve[:, 0], xyz_curve[:, 1], xyz_curve[:, 2])

    # get phi and reorder it from -pi to pi (sometimes arctan returns different quadrant)
    phi = np.arctan2(xyz_curve[:, 1], xyz_curve[:, 0])
    phi_inds = np.argsort(phi)
    phi = phi[phi_inds] + np.pi
    xyz_curve = xyz_curve[phi_inds, :]
    plt.figure()
    plt.plot(phi, xyz_curve[:, 0])
    plt.plot(phi, xyz_curve[:, 1])
    plt.plot(phi, xyz_curve[:, 2])
    # plt.show()

    # Initialize CurveXYZFourier object
    quadpoints = 2000
    coil = CurveXYZFourier(quadpoints, num_fourier)
    dofs = coil.dofs_matrix
    dofs[0][0] = np.trapz(xyz_curve[:, 0], phi) / np.pi

    if truncate and abs(dofs[0][0]) < 1e-2:
        dofs[0][0] = 0.0
    for f in range(num_fourier):
        for i in range(3):
            if i == 0:
                dofs[i][f * 2 + 2] = np.trapz(xyz_curve[:, i] * np.cos((f + 1) * phi), phi) / np.pi
                if truncate and abs(dofs[i][f * 2 + 2]) < 1e-2:
                    dofs[i][f * 2 + 2] = 0.0
            else:
                dofs[i][f * 2 + 1] = np.trapz(xyz_curve[:, i] * np.sin((f + 1) * phi), phi) / np.pi
                if truncate and abs(dofs[i][f * 2 + 1]) < 1e-2:
                    dofs[i][f * 2 + 1] = 0.0
    coil.local_x = np.concatenate(dofs)

    # now make the stellarator symmetry fixed in the optimization
    for f in range(num_fourier):
        coil.fix(f'xs({f + 1})')
    for f in range(num_fourier + 1):
        coil.fix(f'yc({f})')
        coil.fix(f'zc({f})')

    # attempt to fix all the non-nfp-symmetric coefficients to zero during optimiziation
    for f in range(num_fourier * 2 + 1):
        for i in range(3):
            if i == 0:
                if np.isclose(dofs[i][f], 0.0) and (f % 2) == 0 and (f // 2) % 2 == 0:
                    coil.fix(f'xc({f // 2})')
            elif i == 1: 
                if np.isclose(dofs[i][f], 0.0) and (f % 2) != 0 and (f // 2) % 2 != 0:
                    coil.fix(f'ys({f // 2 + 1})')
            if i == 2: 
                if np.isclose(dofs[i][f], 0.0) and (f % 2) != 0:
                    coil.fix(f'zs({f // 2 + 1})')
    return coil


def perform_filament_optimization(s, bs, curves, **kwargs):
    """
    Wrapper function for performing filament optimization with a given BiotSavart
    object, a plasma boundary, and a set of curves. 
    Follows closely the syntax and functionality
    in the stage_two_optimization.py script in examples/2_Intermediate.

    Args:
        s: SurfaceRZFourier object representing the plasma boundary
        bs: BiotSavart class object representing the fields from the filaments.
        curves: List of Curve class objects representing the filaments to be optimized.
    """
    # Weight on the curve lengths in the objective function
    LENGTH_WEIGHT = Weight(kwargs.pop("length_weight", 1e-7))

    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = kwargs.pop("cs_threshold", 0.1)
    CS_WEIGHT = kwargs.pop("cs_weight", 1e-12)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = kwargs.pop("cc_threshold", 0.1)
    CC_WEIGHT = kwargs.pop("cc_weight", 1e-12)

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = kwargs.pop("curvature_threshold", 0.1)
    CURVATURE_WEIGHT = kwargs.pop("curvature_weight", 1e-12)
    TORSION_WEIGHT = kwargs.pop("torsion_weight", 1e-12)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = kwargs.pop("msc_threshold", 0.1)
    MSC_WEIGHT = kwargs.pop("msc_weight", 1e-14)

    # Define the individual terms objective function:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(curve) for curve in curves]
    if len(curves) > 1:
        Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(curve, 2, CURVATURE_THRESHOLD) for curve in curves]
    Jts = [LpCurveTorsion(curve, 2, CURVATURE_THRESHOLD) for curve in curves]
    Jmscs = [MeanSquaredCurvature(curve) for curve in curves]

    # Form the total objective function
    JF = Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + TORSION_WEIGHT * sum(Jts) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)

    if len(curves) > 1:
        JF += CC_WEIGHT * Jccdist

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)))
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in curves) 
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        if len(curves) > 1:
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
    np.random.seed(1)
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
    MAXITER = 2000
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 200}, tol=1e-15)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_super_long")
    pointData = {"B_N": np.sum(bs.B().reshape(s.gamma().shape) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_super_long", extra_data=pointData)
    calculate_on_axis_B(bs, s)
    bs.save(OUT_DIR + "biot_savart_opt.json")
