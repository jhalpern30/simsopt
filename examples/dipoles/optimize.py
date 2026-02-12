from simsopt.geo import SurfaceRZFourier
from simsopt.field import coils_to_vtk
from simsopt.objectives import SquaredFlux
import numpy as np
import os
import json
from helper_functions import *

def optimize(
    fil_distance,
    half_per_distance,
    dipole_radius,
    numquadpoints,  # dipole parameters
    VV_a,
    VV_b,
    VV_R0,  # vessel parameters
    plas_nPhi,
    plas_nTheta,
    surf_s,
    surf_dof_scale,
    eq_dir,
    eq_name,  # equilibrium parameters
    ntf,
    num_fixed,
    field_on_axis,
    TF_R0,
    TF_a,
    TF_b,
    fixed_geo_TFs,
    CC_THRESHOLD,
    CC_WEIGHT,
    CS_THRESHOLD,
    CS_WEIGHT,  # TF parameters
    definition,
    precomputed,
    MAXITER,
    CURRENT_THRESHOLD,
    CURRENT_WEIGHT,
    dpi,
    titlefontsize,
    axisfontsize,
    legendfontsize,
    ticklabelfontsize,
    cbarfontsize,
    output_dir,
    verbose=False,
):
    
    # Create plot configuration
    plot_config = PlotConfig(
        dpi=dpi,
        titlefontsize=titlefontsize,
        axisfontsize=axisfontsize,
        legendfontsize=legendfontsize,
        ticklabelfontsize=ticklabelfontsize,
        cbarfontsize=cbarfontsize
    )

    # ============================================================================
    # Initialization
    # ============================================================================
    # Create the plasma surface
    eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
    surf = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range="half period", nphi=plas_nPhi, ntheta=plas_nTheta)
    surf.set_dofs(surf_dof_scale * surf.get_dofs())

    # Create a surface representing the vacuum vessel that dipoles will be placed on
    VV = SurfaceRZFourier(nfp=surf.nfp)
    VV.set_rc(0, 0, VV_R0)
    VV.set_rc(1, 0, VV_a)
    VV.set_zs(1, 0, VV_b)
    plot_cross_section(surf, VV, output_dir, plot_config)

    # Coil regularization radii (meters)
    tf_coil_radius = 0.05  # TF coil filament radius
    wp_coil_radius = 0.025  # WP coil filament radius, gives 5cm spacing in between coils

    # Initialize TF Coils
    # Compute I from toroidal solenoid approximation, I = B_T * 2 * pi * R0 / mu0 / (2 * nfp * ntf)
    mu0 = 4.0 * np.pi * 1e-7
    TF_current = 2.0 * np.pi * surf.get_rc(0, 0) * field_on_axis / mu0 / (2 * ntf * surf.nfp)
    base_tf_curves, base_tf_coils = generate_tf_array(
        winding_surface=VV,
        ntf=ntf,
        TF_R0=TF_R0,
        TF_a=TF_a,
        TF_b=TF_b,
        TF_current=TF_current,
        fixed_geo_tfs=fixed_geo_TFs,
        numquadpoints=numquadpoints,
        tf_coil_radius=tf_coil_radius,
    )
    tf_regularizations = [c.regularization for c in base_tf_coils] if hasattr(base_tf_coils[0], "regularization") else None
    tf_coils = coils_via_symmetries(
        [c.curve for c in base_tf_coils],
        [c.current for c in base_tf_coils],
        surf.nfp,
        True,
        regularizations=tf_regularizations,
    )
    bs_tf = BiotSavart(tf_coils)
    plot_relBfinal_norm_modB(bs_tf, surf, output_dir, plot_config, "Initial")
    if not fixed_geo_TFs:
        optimize_tfs(
            base_tf_coils=base_tf_coils,
            surf_plasma=surf,
            winding_surface=VV,
            CC_THRESHOLD=CC_THRESHOLD,
            CC_WEIGHT=CC_WEIGHT,
            CS_THRESHOLD=CS_THRESHOLD,
            CS_WEIGHT=CS_WEIGHT,
            num_fixed=num_fixed,
            definition=definition,
            maxiter=MAXITER,
            verbose=verbose,
        )
        plot_relBfinal_norm_modB(bs_tf, surf, output_dir, plot_config, "Post TF Optimization")
        
    # Initialize dipoles
    base_wp_coils = generate_windowpane_array(
        winding_surface=VV,
        inboard_radius=dipole_radius,
        wp_fil_spacing=fil_distance,
        half_per_spacing=half_per_distance,
        wp_n=4,
        numquadpoints=numquadpoints,
        order=12,
        verbose=verbose,
        wp_coil_radius=wp_coil_radius,
    )
    nwptot = len(base_wp_coils * 2 * surf.nfp)

    # ============================================================================
    # Optimization
    # ============================================================================
    res, bs = optimize_windowpane_currents(
        base_wp_coils=base_wp_coils,
        base_tf_coils=base_tf_coils,
        surf_plasma=surf,
        definition=definition,
        precomputed=precomputed,
        current_threshold=CURRENT_THRESHOLD,
        current_weight=CURRENT_WEIGHT,
        maxiter=MAXITER,
        num_fixed=num_fixed,
        verbose=verbose,
    )

    # ============================================================================
    # Post-processing
    # ============================================================================
    # Final Bnormal
    relBfinal_norm, mean_abs_relBfinal_norm, max_relBfinal_norm = plot_relBfinal_norm_modB(bs, surf, output_dir, plot_config, "Final")
    Jf = SquaredFlux(surf, bs, definition=definition)
    # plots currents on surface
    wp_currents_phis_thetas = coil_currents_on_theta_phi_grid(base_wp_coils, VV)
    plot_coil_currents_on_theta_phi_grid(
        wp_currents_phis_thetas,
        output_dir,
        plot_config,
    )

    # Prep coil data
    tf_regularizations = [c.regularization for c in base_tf_coils] if hasattr(base_tf_coils[0], "regularization") else None
    wp_regularizations = [c.regularization for c in base_wp_coils] if hasattr(base_wp_coils[0], "regularization") else None
    tf_coils = coils_via_symmetries(
        [c.curve for c in base_tf_coils],
        [c.current for c in base_tf_coils],
        surf.nfp,
        True,
        regularizations=tf_regularizations,
    )
    wp_coils = coils_via_symmetries(
        [c.curve for c in base_wp_coils],
        [c.current for c in base_wp_coils],
        surf.nfp,
        True,
        regularizations=wp_regularizations,
    )
    coils = tf_coils + wp_coils
    tf_currents = [c.current.get_value() for c in tf_coils]
    wp_currents = [c.current.get_value() for c in wp_coils]
    
    # Save various files
    VV.to_vtk(os.path.join(output_dir, "vacuum_vessel"))
    # Need to include all coils together in output dump for force/torque calcs
    # THIS TAKES A WHILE! But uncomment if you want the vtk output
    # TODO: make this a flag of whether to do the short curves_to_vtk with current output or
    # the full coils_to_vtk output with forces and torques, which is more expensive
    # coils_to_vtk(coils, filename=os.path.join(output_dir, "coils"), close=True)
    bs.save(os.path.join(output_dir, "bs_opt.json"))
    # BdotN on the full torus surface
    surf_full = SurfaceRZFourier.from_wout(
        eq_name_full,
        s=surf_s,
        range="full torus",
        nphi=2 * surf.nfp * plas_nPhi,
        ntheta=plas_nTheta,
    )
    bs.set_points(surf_full.gamma().reshape(-1, 3))
    Bdotn = np.sum(bs.B().reshape(surf_full.unitnormal().shape) * surf_full.unitnormal(), axis=2)
    modB = bs.AbsB().reshape((2 * surf.nfp * plas_nPhi, plas_nTheta))
    BdotN_norm = Bdotn / modB
    surf_full.to_vtk(os.path.join(output_dir, "surf_full"), extra_data={"B_N": BdotN_norm[:, :, None]})

    # Set points back for Jf.J in results section
    bs.set_points(surf.gamma().reshape(-1, 3))

    # Compute forces and torques for TF and WP coils
    # For each coil, compute force/torque from all coils, then extract statistics
    # max_tf_forces = []
    # min_tf_forces = []
    # mean_tf_forces = []
    # max_tf_torques = []
    # min_tf_torques = []
    # mean_tf_torques = []
    
    # This takes a while, so commenting out for now
    # for c in tf_coils:
    #     force_per_length = c.force(coils)  # Force per unit length (N/m)
    #     torque_per_length = c.torque(coils)  # Torque per unit length (N)
    #     force_mag = np.linalg.norm(force_per_length, axis=1)
    #     torque_mag = np.linalg.norm(torque_per_length, axis=1)
    #     max_tf_forces.append(np.max(force_mag))
    #     min_tf_forces.append(np.min(force_mag))
    #     mean_tf_forces.append(np.mean(force_mag))
    #     max_tf_torques.append(np.max(torque_mag))
    #     min_tf_torques.append(np.min(torque_mag))
    #     mean_tf_torques.append(np.mean(torque_mag))
    
    # max_wp_forces = []
    # min_wp_forces = []
    # mean_wp_forces = []
    # max_wp_torques = []
    # min_wp_torques = []
    # mean_wp_torques = []
    
    # This takes a while, so commenting out for now
    # for c in wp_coils:
    #     force_per_length = c.force(coils)
    #     torque_per_length = c.torque(coils)
    #     force_mag = np.linalg.norm(force_per_length, axis=1)
    #     torque_mag = np.linalg.norm(torque_per_length, axis=1)
    #     max_wp_forces.append(np.max(force_mag))
    #     min_wp_forces.append(np.min(force_mag))
    #     mean_wp_forces.append(np.mean(force_mag))
    #     max_wp_torques.append(np.max(torque_mag))
    #     min_wp_torques.append(np.min(torque_mag))
    #     mean_wp_torques.append(np.mean(torque_mag))

    # Extract TF optimized geometric dofs
    if not fixed_geo_TFs:
        R0s = np.zeros_like(base_tf_curves)
        r_rotations = np.zeros_like(base_tf_curves)
        for i, c in enumerate(base_tf_curves):
            R0s[i] = c.get("R0")
            r_rotations[i] = c.get("r_rotation")
    else:
        R0s = None
        r_rotations = None

    results = {
        # input parameters
        "filament_distance": fil_distance,
        "half_period_distance": half_per_distance,
        "inboard_radius": dipole_radius,
        "numquadpoints": numquadpoints,
        "VV_a": VV_a,
        "VV_b": VV_b,
        "VV_R0": VV_R0,
        "plas_nPhi": plas_nPhi,
        "plas_nTheta": plas_nTheta,
        "surf_s": surf_s,
        "surf_dof_scale": surf_dof_scale,
        "eq_dir": eq_dir,
        "eq_name": eq_name,
        "ntf": ntf,
        "num_fixed": num_fixed,
        "TF_R0": TF_R0,
        "TF_a": TF_a,
        "TF_b": TF_b,
        "fixed_geo_TFs": fixed_geo_TFs,
        "CC_THRESHOLD": CC_THRESHOLD,
        "CC_WEIGHT": CC_WEIGHT,
        "CS_THRESHOLD": CS_THRESHOLD,
        "CS_WEIGHT": CS_WEIGHT,
        "field_on_axis": field_on_axis,
        "squared_flux_def": definition,
        "max_iterations": MAXITER,
        "current_threshold": CURRENT_THRESHOLD,
        "current_weight": CURRENT_WEIGHT,
        # derived quantities
        "surf_nfp": surf.nfp,
        "surf_major_radius": surf.major_radius(),
        "surf_minor_radius": surf.minor_radius(),
        "surf_aspect_ratio": surf.aspect_ratio(),
        "surf_volume": surf.volume(),
        "initial_tf_current": TF_current,
        "initial_wp_current": wp_current_init,
        "num_wps": nwptot,
        "ntoroidal": int((np.pi/surf.nfp*(VV_R0 - VV_a) - half_per_distance + fil_distance) / (2 * dipole_radius + fil_distance)),
        "npoloidal": int(len(base_wp_coils) / int((np.pi/surf.nfp*(VV_R0 - VV_a) - half_per_distance + fil_distance) / (2 * dipole_radius + fil_distance))),
        # optimization results
        "message":                  res.message,
        "success":                  res.success,
        "iterations":               res.nit,
        "function_evaluations":     res.nfev,
        "max_tf_current": np.max(np.abs(np.array(tf_currents))),
        "min_tf_current": np.min(np.abs(np.array(tf_currents))),
        "max_wp_current": np.max(np.abs(np.array(wp_currents))),
        "min_wp_current": np.min(np.abs(np.array(wp_currents))),
        # "tf_max_max_force": max(float(f) for f in max_tf_forces),
        # "tf_min_min_force": min(float(f) for f in min_tf_forces),
        # "tf_mean_mean_force": float(np.mean([f for f in mean_tf_forces])),
        # "wp_max_max_force": max(float(f) for f in max_wp_forces),
        # "wp_min_min_force": min(float(f) for f in min_wp_forces),
        # "wp_mean_mean_force": float(np.mean([f for f in mean_wp_forces])),
        # "tf_max_max_torque": max(float(f) for f in max_tf_torques),
        # "tf_min_min_torque": min(float(f) for f in min_tf_torques),
        # "tf_mean_mean_torque": float(np.mean([f for f in mean_tf_torques])),
        # "wp_max_max_torque": max(float(f) for f in max_wp_torques),
        # "wp_min_min_torque": min(float(f) for f in min_wp_torques),
        # "wp_mean_mean_torque": float(np.mean([f for f in mean_wp_torques])),
        "final_squared_flux": Jf.J(),
        "avg_Bnormal": mean_abs_relBfinal_norm,
        "max_Bnormal": max_relBfinal_norm,
        "peak_wp_field": np.max(np.abs(np.array(wp_currents))) * mu0 / 2 / dipole_radius,
        "MA_meters": get_total_amp_meters(base_tf_coils, base_wp_coils, VV) / 1e6,
        "maxR0": np.max(R0s) if R0s is not None else None,
        "minR0": np.min(R0s) if R0s is not None else None,
        "avgR0": np.mean(R0s) if R0s is not None else None,
        "maxtilt": np.max(r_rotations) if r_rotations is not None else None,
        "mintilt": np.min(r_rotations) if r_rotations is not None else None,
        "avgtilt": np.mean(np.abs(r_rotations)) if r_rotations is not None else None,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)