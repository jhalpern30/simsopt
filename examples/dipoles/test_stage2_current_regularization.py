"""
Isolated study of L1 total-current regularization in stage-II windowpane optimization.

Runs the standard equilibrium / coil setup used in optimize.py, adds an optional
penalty proportional to the sum of absolute windowpane currents, and produces
plots that show whether regularization is needed as the plasma-surface
quadrature resolution varies. The windowpane count is fixed throughout.

Usage (from examples/dipoles):
    conda run -n simsopt python test_stage2_current_regularization.py
"""

from __future__ import annotations

import os

# Cray MPICH may abort on login nodes when GPU support is enabled but GTL is absent.
os.environ.setdefault("MPICH_GPU_SUPPORT_ENABLED", "0")

import argparse
import json
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from helper_functions import (  # noqa: E402
    apply_symmetries_to_curves,
    apply_symmetries_to_currents,
    classify_current,
    derivativeJcp,
    generate_tf_array,
    generate_windowpane_array,
    surf_int,
    surf_int_3D,
)


class TotalCurrentMagnitude(Optimizable):
    """
    L1 norm of coil currents (total current magnitude):

        J = sum_i |I_i|

    Differentiable everywhere except I_i = 0, where the subgradient is used.
    """

    def __init__(self, currents):
        super().__init__(depends_on=list(currents))
        self.currents = list(currents)
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        values = np.asarray([c.get_value() for c in self.currents], dtype=float)
        self._J = float(np.sum(np.abs(values)))
        dJ_val = np.sign(values)
        dJ_val[values == 0.0] = 0.0
        deriv = Derivative({})
        for i, current in enumerate(self.currents):
            deriv += current.vjp(np.atleast_1d(dJ_val[i]))
        self._dJ = deriv


@dataclass
class Stage2Config:
    eq_name: str = "wout_nfp22ginsburg_000_000281"
    eq_dir: str | None = None
    surf_s: float = 1.0
    surf_dof_scale: float = 1.0
    nphi: int = 128
    ntheta: int = 64
    fil_distance: float = 0.05
    half_per_distance: float = 0.05
    dipole_radius: float | None = 0.05
    wp_npol_target: int | None = 12
    wp_ntor_target: int | None = 12
    ntf: int = 4
    num_fixed: int = 4
    field_on_axis: float = 0.5
    tf_a: float = 0.40


@dataclass
class Stage2Setup:
    surf: SurfaceRZFourier
    vv: SurfaceRZFourier
    base_tf_coils: list
    base_wp_coils: list
    eq_name: str
    ntf: int
    num_fixed: int
    nwps_poloidal: int
    nwps_toroidal: int


def _vv_geometry_from_equilibrium(eq_path: str, surf_s: float, surf_dof_scale: float):
    surf_full = SurfaceRZFourier.from_wout(
        eq_path, s=surf_s, range="full torus", nphi=128, ntheta=64
    )
    surf_full.set_dofs(surf_dof_scale * surf_full.get_dofs())
    gamma = surf_full.gamma()
    r = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    rmin, rmax = np.min(r), np.max(r)
    zmin, zmax = np.min(gamma[:, :, 2]), np.max(gamma[:, :, 2])
    vv_r0 = (rmin + rmax) / 2 + 0.03
    vv_a = (rmax - rmin) / 2 + 0.12
    vv_b = (zmax - zmin) / 2 + 0.12
    return vv_r0, vv_a, vv_b


def build_standard_stage2_setup(config: Stage2Config | None = None, **kwargs) -> Stage2Setup:
    """Standard equilibrium + TF + windowpane configuration from optimize.py."""
    if config is None:
        config = Stage2Config(**kwargs)
    else:
        for key, value in kwargs.items():
            setattr(config, key, value)

    eq_name = config.eq_name
    eq_dir = config.eq_dir or os.path.join(script_dir, "equilibria")
    eq_path = os.path.join(eq_dir, eq_name + ".nc")
    if not os.path.isfile(eq_path):
        raise FileNotFoundError(f"Equilibrium not found: {eq_path}")

    surf = SurfaceRZFourier.from_wout(
        eq_path, s=config.surf_s, range="half period", nphi=config.nphi, ntheta=config.ntheta
    )
    surf.set_dofs(config.surf_dof_scale * surf.get_dofs())

    vv_r0, vv_a, vv_b = _vv_geometry_from_equilibrium(
        eq_path, config.surf_s, config.surf_dof_scale
    )
    vv = SurfaceRZFourier(nfp=surf.nfp)
    vv.set_rc(0, 0, vv_r0)
    vv.set_rc(1, 0, vv_a)
    vv.set_zs(1, 0, vv_b)

    mu0 = 4.0 * np.pi * 1e-7
    tf_current = (
        2.0 * np.pi * surf.get_rc(0, 0) * config.field_on_axis / mu0 / (2 * config.ntf * surf.nfp)
    )
    tf_b = config.tf_a * vv_b / vv_a
    _, base_tf_coils = generate_tf_array(
        winding_surface=vv,
        ntf=config.ntf,
        TF_R0=vv_r0,
        TF_a=config.tf_a,
        TF_b=tf_b,
        TF_current=tf_current,
        fixed_geo_tfs=True,
        numquadpoints=64,
        tf_coil_radius=0.0,
    )
    base_wp_coils, _, _, _, nwps_poloidal, nwps_toroidal = generate_windowpane_array(
        winding_surface=vv,
        inboard_radius=config.dipole_radius,
        wp_fil_spacing=config.fil_distance,
        half_per_spacing=config.half_per_distance,
        wp_n=4,
        numquadpoints=64,
        order=12,
        verbose=False,
        wp_coil_radius=0.0,
        nwps_poloidal_target=config.wp_npol_target,
        nwps_toroidal_target=config.wp_ntor_target,
    )
    return Stage2Setup(
        surf=surf,
        vv=vv,
        base_tf_coils=base_tf_coils,
        base_wp_coils=base_wp_coils,
        eq_name=eq_name,
        ntf=config.ntf,
        num_fixed=config.num_fixed,
        nwps_poloidal=nwps_poloidal,
        nwps_toroidal=nwps_toroidal,
    )


def _field_error_metrics(bs: BiotSavart, surf_plasma: SurfaceRZFourier):
    n = surf_plasma.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1.0 / absn)[:, :, None]
    bs.set_points(surf_plasma.gamma().reshape((-1, 3)))
    bfinal = bs.B().reshape(n.shape)
    bfinal_norm = np.sum(bfinal * unitn, axis=2)
    mod_b = np.linalg.norm(bfinal, axis=2)
    rel_bn = bfinal_norm / mod_b
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    surf_area = sqrt_area ** 2
    abs_rel = np.abs(rel_bn.reshape((-1, 1))) * surf_area
    mean_abs_rel_bn = float(np.sum(abs_rel) / np.sum(surf_area))
    max_abs_rel_bn = float(np.max(np.abs(rel_bn)))
    jf = SquaredFlux(surf_plasma, bs, definition="local")
    return {
        "squared_flux": float(jf.J()),
        "mean_abs_rel_bn": mean_abs_rel_bn,
        "max_abs_rel_bn": max_abs_rel_bn,
    }


def _wp_current_stats(base_wp_coils):
    currents = np.array([c.current.get_value() for c in base_wp_coils], dtype=float)
    return {
        "max_abs_wp_current": float(np.max(np.abs(currents))),
        "mean_abs_wp_current": float(np.mean(np.abs(currents))),
        "total_current_magnitude": float(np.sum(np.abs(currents))),
        "wp_currents": currents,
    }


def optimize_stage2_with_regularization(
    setup: Stage2Setup,
    regularization_weight: float = 0.0,
    current_threshold: float = 1e12,
    current_weight: float = 0.0,
    definition: str = "local",
    precomputed: bool = True,
    maxiter: int = 1500,
    verbose: bool = False,
):
    """
    Stage-II windowpane current optimization with optional L1 current regularization.

    When regularization_weight > 0, adds regularization_weight * TotalCurrentMagnitude
    to the squared-flux objective. The threshold penalty from optimize.py remains
    available but defaults to zero here so the study isolates L1 regularization.
    """
    base_wp_coils = setup.base_wp_coils
    base_tf_coils = setup.base_tf_coils
    surf_plasma = setup.surf
    num_fixed = setup.num_fixed

    for i in range(num_fixed):
        base_tf_coils[i].current.fix_all()
    for coil in base_wp_coils:
        coil.curve.fix_all()
        coil.current.unfix_all()
    for coil in base_tf_coils:
        coil.curve.fix_all()

    base_coils = base_tf_coils + base_wp_coils
    wp_currents = [c.current for c in base_wp_coils]
    j_reg = TotalCurrentMagnitude(wp_currents)

    if precomputed:
        jf_temp = SquaredFlux(surf_plasma, BiotSavart(base_wp_coils[0:1]), definition=definition)
        dof_start_num = int(jf_temp.dof_names[0].split(":")[0].replace("Current", ""))
        ndofs = len(base_coils) - num_fixed
        numbers = list(range(dof_start_num, ndofs + dof_start_num))
        sorted_indices = [int(num) for num in sorted(str(n) for n in numbers)]

        surf_plasma_nphi = len(surf_plasma.quadpoints_phi)
        surf_plasma_ntheta = len(surf_plasma.quadpoints_theta)

        bdotn_coil = np.zeros((ndofs, surf_plasma_nphi, surf_plasma_ntheta))
        bdotn_coil_fixed = np.zeros((num_fixed, surf_plasma_nphi, surf_plasma_ntheta))
        if definition != "quadratic flux":
            b_coil = np.zeros((ndofs, surf_plasma_nphi, surf_plasma_ntheta, 3))
            b_coil_fixed = np.zeros((num_fixed, surf_plasma_nphi, surf_plasma_ntheta, 3))

        coils = []
        for ii, coil in enumerate(base_tf_coils[0:num_fixed]):
            paired_curves = apply_symmetries_to_curves(
                base_curves=[coil.curve], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym
            )
            paired_currents = apply_symmetries_to_currents(
                base_currents=[coil.current], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym
            )
            paired = [Coil(curve, current) for curve, current in zip(paired_curves, paired_currents)]
            bs_fixed = BiotSavart(paired)
            bs_fixed.set_points(surf_plasma.gamma().reshape((-1, 3)))
            bdotn_coil_fixed[ii, :, :] = np.sum(
                bs_fixed.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3))
                * surf_plasma.unitnormal(),
                axis=2,
            )
            if definition in ("local", "normalized"):
                b_coil_fixed[ii, :, :, :] = bs_fixed.B().reshape(
                    (surf_plasma_nphi, surf_plasma_ntheta, 3)
                )
            coils += paired

        for ii, coil in enumerate(base_coils[num_fixed:]):
            paired_curves = apply_symmetries_to_curves(
                base_curves=[coil.curve], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym
            )
            paired_currents = apply_symmetries_to_currents(
                base_currents=[coil.current], nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym
            )
            paired = [Coil(curve, current) for curve, current in zip(paired_curves, paired_currents)]
            bs_coil = BiotSavart(paired)
            bs_coil.set_points(surf_plasma.gamma().reshape((-1, 3)))
            idx = sorted_indices.index(ii + dof_start_num)
            bdotn_coil[idx, :, :] = np.sum(
                bs_coil.B().reshape((surf_plasma_nphi, surf_plasma_ntheta, 3))
                * surf_plasma.unitnormal(),
                axis=2,
            )
            if definition in ("local", "normalized"):
                b_coil[idx, :, :, :] = bs_coil.B().reshape(
                    (surf_plasma_nphi, surf_plasma_ntheta, 3)
                )
            coils += paired

        bs = BiotSavart(coils)
        jf = SquaredFlux(surf_plasma, bs, definition=definition)
        dofs = jf.x.copy()
        wp_scale_factor = base_wp_coils[0].current.get_value()
        target_normal = np.zeros((surf_plasma_nphi, surf_plasma_ntheta))
        dJscale = [classify_current(name, len(base_tf_coils) + 1) for name in jf.dof_names]
        wp_mask = np.asarray(dJscale, dtype=bool)

        def fun(dofs_local, info={"Nfeval": 0}):
            info["Nfeval"] += 1
            n_norm = np.linalg.norm(surf_plasma.normal(), axis=2)
            phi = surf_plasma.quadpoints_phi
            theta = surf_plasma.quadpoints_theta
            bdotn = np.sum(dofs_local[:, None, None] * bdotn_coil, axis=0) + np.sum(
                bdotn_coil_fixed, axis=0
            )
            if definition in ("local", "normalized"):
                b_field = np.sum(dofs_local[:, None, None, None] * b_coil, axis=0) + np.sum(
                    b_coil_fixed, axis=0
                )
                mod_b = np.linalg.norm(b_field, axis=2)
                b_coil_dot_b = np.sum(b_coil * b_field, axis=3)
            if definition == "local":
                sf = 0.5 * surf_int((bdotn / mod_b) ** 2, n_norm, theta, phi)
                grad_sf = surf_int_3D(
                    (mod_b[None, :, :] ** 2 * bdotn_coil * bdotn[None, :, :]
                     - b_coil_dot_b * bdotn[None, :, :] ** 2)
                    / mod_b[None, :, :] ** 4,
                    n_norm,
                    theta,
                    phi,
                )
            elif definition == "normalized":
                sf = 0.5 * surf_int(bdotn ** 2, n_norm, theta, phi) / surf_int(
                    mod_b ** 2, n_norm, theta, phi
                )
                grad_sf = (
                    surf_int_3D(mod_b[None, :, :] ** 2, n_norm, theta, phi)
                    * surf_int_3D(bdotn_coil * bdotn[None, :, :], n_norm, theta, phi)
                    - surf_int_3D(b_coil_dot_b, n_norm, theta, phi)
                    * surf_int_3D(bdotn[None, :, :] ** 2, n_norm, theta, phi)
                ) / surf_int_3D(mod_b[None, :, :] ** 2, n_norm, theta, phi) ** 2
            else:
                sf = 0.5 * surf_int(bdotn ** 2, n_norm, theta, phi)
                grad_sf = surf_int_3D(bdotn_coil * bdotn[None, :, :], n_norm, theta, phi)

            jcp = np.sum(
                dJscale
                * np.array(
                    [
                        np.maximum(np.abs(dofs_local[i] * wp_scale_factor) - current_threshold, 0) ** 2
                        for i in range(len(dofs_local))
                    ]
                )
            )
            djcp = dJscale * np.array(
                [
                    derivativeJcp(dofs_local[i] * wp_scale_factor, current_threshold)
                    for i in range(len(dofs_local))
                ]
            )

            j_reg_val = float(np.sum(np.abs(dofs_local[wp_mask])))
            dj_reg = np.zeros_like(dofs_local)
            dj_reg[wp_mask] = np.sign(dofs_local[wp_mask])

            total_j = (
                sf
                + current_weight * jcp
                + regularization_weight * j_reg_val
            )
            grad = grad_sf + current_weight * djcp + regularization_weight * dj_reg
            if verbose and info["Nfeval"] % 10 == 0:
                print(
                    f"  iter {info['Nfeval']:4d}: Jf={sf:.3e}, "
                    f"Jreg={regularization_weight * j_reg_val:.3e}, "
                    f"max|I|={np.max(np.abs(dofs_local[wp_mask])):.3e}"
                )
            return total_j, grad

        res = minimize(
            fun,
            dofs,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxcor": 300},
            tol=1e-20,
        )
        jf.x = res.x
    else:
        from helper_functions import coils_via_symmetries

        coils = coils_via_symmetries(
            [c.curve for c in base_tf_coils + base_wp_coils],
            [c.current for c in base_tf_coils + base_wp_coils],
            surf_plasma.nfp,
            surf_plasma.stellsym,
        )
        bs = BiotSavart(coils)
        jf = SquaredFlux(surf_plasma, bs, definition=definition)
        dofs = jf.x.copy()

        def fun(dofs_local, info={"Nfeval": 0}):
            info["Nfeval"] += 1
            jf.x = dofs_local
            sf = jf.J()
            grad_sf = jf.dJ()
            j_reg_val = j_reg.J()
            grad_reg = j_reg.dJ()
            total_j = sf + regularization_weight * j_reg_val
            grad = grad_sf + regularization_weight * grad_reg
            if verbose and info["Nfeval"] % 5 == 0:
                print(
                    f"  iter {info['Nfeval']:4d}: Jf={sf:.3e}, "
                    f"Jreg={regularization_weight * j_reg_val:.3e}"
                )
            return total_j, grad

        res = minimize(
            fun,
            dofs,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxcor": 300},
            tol=1e-20,
        )
        jf.x = res.x

    return res, bs, j_reg


def run_single_case(
    config: Stage2Config,
    regularization_weight: float,
    maxiter: int,
    verbose: bool = False,
):
    setup = build_standard_stage2_setup(config)
    res, bs, j_reg = optimize_stage2_with_regularization(
        setup,
        regularization_weight=regularization_weight,
        maxiter=maxiter,
        verbose=verbose,
    )
    metrics = _field_error_metrics(bs, setup.surf)
    current_stats = _wp_current_stats(setup.base_wp_coils)
    return {
        "success": bool(res.success),
        "iterations": int(res.nit),
        "regularization_weight": regularization_weight,
        "regularization_term": regularization_weight * j_reg.J(),
        **metrics,
        **{k: v for k, v in current_stats.items() if k != "wp_currents"},
        "wp_currents": current_stats["wp_currents"],
        "nphi": len(setup.surf.quadpoints_phi),
        "ntheta": len(setup.surf.quadpoints_theta),
    }


def _pick_reference_weight(weight_results):
    """Choose a moderate regularization weight for the resolution comparison."""
    unreg = next(r for r in weight_results if r["regularization_weight"] == 0.0)
    for row in sorted(weight_results, key=lambda r: r["regularization_weight"]):
        if row["regularization_weight"] <= 0:
            continue
        current_drop = 1.0 - row["max_abs_wp_current"] / unreg["max_abs_wp_current"]
        flux_increase = row["squared_flux"] / unreg["squared_flux"] - 1.0
        if current_drop >= 0.15 and flux_increase <= 0.25:
            return row["regularization_weight"]
    nonzero = [r for r in weight_results if r["regularization_weight"] > 0]
    if not nonzero:
        return 0.0
    mid = nonzero[len(nonzero) // 2]
    return mid["regularization_weight"]


def _plot_grouped_resolution_comparison(
    ax_row,
    rows_unreg,
    rows_reg,
    labels,
    reference_weight,
    xlabel,
    title,
):
    x = np.arange(len(labels))
    width = 0.35
    keys = [
        ("squared_flux", r"$J_f$", 1.0),
        ("max_abs_wp_current", "Max $|I|$ [kA]", 1e-3),
        ("mean_abs_rel_bn", r"Mean $|\mathbf{B}\cdot\mathbf{n}|/|\mathbf{B}|$", 1.0),
    ]
    for ax, (key, ylabel, scale) in zip(ax_row, keys):
        ax.bar(x - width / 2, [r[key] * scale for r in rows_unreg], width, label="no regularization")
        ax.bar(
            x + width / 2,
            [r[key] * scale for r in rows_reg],
            width,
            label=f"L1 reg ($\\lambda={reference_weight:.1e}$)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    ax_row[0].legend()
    ax_row[0].figure.suptitle(title)


def make_plots(
    output_dir: str,
    weight_results: list[dict],
    surface_resolution_results: list[dict],
    reference_weight: float,
):
    os.makedirs(output_dir, exist_ok=True)

    weights = np.array([r["regularization_weight"] for r in weight_results])
    max_current = np.array([r["max_abs_wp_current"] for r in weight_results])
    total_current = np.array([r["total_current_magnitude"] for r in weight_results])
    mean_bn = np.array([r["mean_abs_rel_bn"] for r in weight_results])
    max_bn = np.array([r["max_abs_rel_bn"] for r in weight_results])
    sq_flux = np.array([r["squared_flux"] for r in weight_results])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(max_current / 1e3, mean_bn, "o-", label=r"mean $|\mathbf{B}\cdot\mathbf{n}|/|\mathbf{B}|$")
    ax.plot(max_current / 1e3, max_bn, "s--", label=r"max $|\mathbf{B}\cdot\mathbf{n}|/|\mathbf{B}|$")
    for row in weight_results:
        if row["regularization_weight"] == 0:
            ax.annotate("unreg", (row["max_abs_wp_current"] / 1e3, row["mean_abs_rel_bn"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Max windowpane current [kA]")
    ax.set_ylabel("Relative normal-field error")
    ax.set_title("Field error vs max current (regularization weight sweep)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "field_error_vs_max_current.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    panels = [
        (axes[0, 0], sq_flux, r"$J_f$ (local squared flux)"),
        (axes[0, 1], max_current / 1e3, "Max $|I|$ [kA]"),
        (axes[1, 0], total_current / 1e6, r"$\sum |I_i|$ [MA]"),
        (axes[1, 1], mean_bn, r"Mean $|\mathbf{B}\cdot\mathbf{n}|/|\mathbf{B}|$"),
    ]
    for ax, values, ylabel in panels:
        ax.plot(weights, values, "o-")
        ax.set_xscale("symlog", linthresh=1e-15)
        ax.set_xlabel("Regularization weight")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Stage-II metrics vs L1 regularization weight (fixed resolution)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_vs_regularization_weight.png"), dpi=150)
    plt.close(fig)

    if surface_resolution_results:
        surf_unreg = [
            r for r in surface_resolution_results if r["regularization_weight"] == 0.0
        ]
        surf_unreg = sorted(surf_unreg, key=lambda r: (r["nphi"], r["ntheta"]))
        surf_reg = [
            next(
                r for r in surface_resolution_results
                if r["nphi"] == base["nphi"]
                and r["ntheta"] == base["ntheta"]
                and r["regularization_weight"] > 0
            )
            for base in surf_unreg
        ]
        surf_labels = [f"{r['nphi']}x{r['ntheta']}" for r in surf_unreg]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        _plot_grouped_resolution_comparison(
            axes,
            surf_unreg,
            surf_reg,
            surf_labels,
            reference_weight,
            xlabel="Plasma surface quadrature (nphi x ntheta)",
            title="Surface resolution scan: with vs without L1 current regularization",
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "surface_resolution_scan_with_without_regularization.png"),
            dpi=150,
        )
        plt.close(fig)

    unreg = next(r for r in weight_results if r["regularization_weight"] == 0.0)
    reg_example = max(
        (r for r in weight_results if r["regularization_weight"] > 0),
        key=lambda r: r["regularization_weight"],
        default=unreg,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(
        0,
        1.05 * max(np.max(np.abs(unreg["wp_currents"])), np.max(np.abs(reg_example["wp_currents"]))) / 1e3,
        25,
    )
    axes[0].hist(unreg["wp_currents"] / 1e3, bins=bins, alpha=0.8)
    axes[0].set_title("Unregularized")
    axes[1].hist(reg_example["wp_currents"] / 1e3, bins=bins, alpha=0.8, color="C1")
    axes[1].set_title(f"Regularized ($\\lambda={reg_example['regularization_weight']:.1e}$)")
    for ax in axes:
        ax.set_xlabel("Windowpane current [kA]")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Windowpane current distributions (fixed resolution)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "current_histograms.png"), dpi=150)
    plt.close(fig)

    ratio = max_current / np.maximum(max_current[0], 1.0)
    flux_ratio = sq_flux / np.maximum(sq_flux[0], 1e-30)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(weights, ratio, "o-", label="max $|I|$/max $|I|_{\\lambda=0}$")
    ax.plot(weights, flux_ratio, "s--", label="$J_f/J_{f,\\lambda=0}$")
    ax.set_xscale("symlog", linthresh=1e-15)
    ax.set_xlabel("Regularization weight")
    ax.set_ylabel("Normalized value")
    ax.set_title("Regularization trade-off (fixed resolution)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regularization_tradeoff_normalized.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stage-II L1 current regularization study")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "../outputs/stage2_regularization_study"),
        help="Directory for plots and summary JSON",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    maxiter = 1500
    reg_weights = [0.0, 1e-14, 1e-12, 1e-10, 1e-8]
    surface_nphi_grid = [32, 64, 96, 128, 256]

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Building standard stage-II setup...")
    base_config = Stage2Config(
        wp_npol_target=10,
        wp_ntor_target=10,
        nphi=128,
        ntheta=64,
    )
    base_setup = build_standard_stage2_setup(base_config)
    print(
        f"  equilibrium={base_setup.eq_name}, "
        f"grid={base_setup.nwps_poloidal}x{base_setup.nwps_toroidal}, "
        f"nphi={len(base_setup.surf.quadpoints_phi)}, "
        f"ntheta={len(base_setup.surf.quadpoints_theta)}"
    )

    print("\nSweeping regularization weight at fixed resolution...")
    weight_results = []
    for weight in reg_weights:
        print(f"  lambda={weight:.1e}")
        row = run_single_case(base_config, weight, maxiter=maxiter, verbose=args.verbose)
        row["nwps_poloidal"] = base_setup.nwps_poloidal
        row["nwps_toroidal"] = base_setup.nwps_toroidal
        weight_results.append(row)
        print(
            f"    Jf={row['squared_flux']:.3e}, "
            f"max|I|={row['max_abs_wp_current']:.3e} A, "
            f"mean|Bn|/|B|={row['mean_abs_rel_bn']:.3e}"
        )

    reference_weight = _pick_reference_weight(weight_results)
    print(f"\nReference regularization weight for surface resolution scan: {reference_weight:.1e}")

    print("\nSweeping plasma surface quadrature resolution with and without regularization...")
    surface_resolution_results = []
    for nphi in surface_nphi_grid:
        ntheta = max(nphi // 2, 16)
        for weight in (0.0, reference_weight):
            label = "unreg" if weight == 0.0 else f"reg({weight:.1e})"
            print(f"  surf={nphi}x{ntheta}, {label}")
            config = Stage2Config(
                wp_npol_target=base_config.wp_npol_target,
                wp_ntor_target=base_config.wp_ntor_target,
                nphi=nphi,
                ntheta=ntheta,
            )
            row = run_single_case(config, weight, maxiter=maxiter, verbose=args.verbose)
            row["nwps_poloidal"] = base_setup.nwps_poloidal
            row["nwps_toroidal"] = base_setup.nwps_toroidal
            surface_resolution_results.append(row)
            print(
                f"    Jf={row['squared_flux']:.3e}, "
                f"max|I|={row['max_abs_wp_current']:.3e} A"
            )

    serializable_weight = [{k: v for k, v in r.items() if k != "wp_currents"} for r in weight_results]
    serializable_surface_resolution = [
        {k: v for k, v in r.items() if k != "wp_currents"} for r in surface_resolution_results
    ]
    summary = {
        "eq_name": base_setup.eq_name,
        "wp_grid": f"{base_setup.nwps_poloidal}x{base_setup.nwps_toroidal}",
        "reference_regularization_weight": reference_weight,
        "weight_sweep": serializable_weight,
        "surface_resolution_sweep": serializable_surface_resolution,
    }
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWriting plots to {output_dir} ...")
    make_plots(
        output_dir,
        weight_results,
        surface_resolution_results,
        reference_weight,
    )

    unreg = next(r for r in weight_results if r["regularization_weight"] == 0.0)
    strongest = max(weight_results, key=lambda r: r["regularization_weight"])
    current_reduction = 1.0 - strongest["max_abs_wp_current"] / unreg["max_abs_wp_current"]
    flux_penalty = strongest["squared_flux"] / unreg["squared_flux"] - 1.0
    print("\nSummary")
    print(f"  Unregularized: Jf={unreg['squared_flux']:.3e}, max|I|={unreg['max_abs_wp_current']:.3e} A")
    print(
        f"  Strongest reg (lambda={strongest['regularization_weight']:.1e}): "
        f"Jf={strongest['squared_flux']:.3e}, max|I|={strongest['max_abs_wp_current']:.3e} A"
    )
    print(f"  Max-current reduction at strongest weight: {100 * current_reduction:.1f}%")
    print(f"  Squared-flux increase at strongest weight: {100 * flux_penalty:.1f}%")
    print("Done.")


if __name__ == "__main__":
    main()
