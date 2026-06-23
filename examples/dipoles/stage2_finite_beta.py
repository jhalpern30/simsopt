#!/usr/bin/env python3
"""
Finite-beta dipole stage-2: optimise windowpane currents against a
VirtualCasing B.n target on the plasma surface.

Inputs:
  - ``equilibria/input.vacuum`` – VMEC solver template
  - ``equilibria/input.hbt_json_pI`` – pressure/current profiles (parsed only)
  - Single-stage run (under ``SINGLE_STAGE_DIR``): ``surf_opt.json``, ``bs_opt.json``, ``results.json``
    (from ``nersc_run/.../mpol12_ntor12``)

Steps:
  1. VMEC vacuum and finite-beta equilibria on the optimised boundary
  2. Quasisymmetry residual comparison (vacuum vs finite-beta)
  3. VirtualCasing on the finite-beta equilibrium
  4. Dipole-only stage-2 optimisation of (B_coils.n - B_external_normal)^2
  5. Initial/final field and coil-current diagnostics
  6. Vacuum vs finite-beta Boozer profile plots

Outputs (under ``ROOT_DIR``):
  - ``vacuum/``, ``finite_beta/`` – VMEC wout files
  - ``plots/`` – relBn/modB, coil currents, qs_error_comparison,
    hbt_profiles_and_rotational_transform (.png and .pdf)
  - ``coils/biot_savart_opt.json``, ``surf_initial.vts``, ``surf_final.vts``
  - ``vcasing_nersc_finite_beta.nc`` – VirtualCasing cache
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import booz_xform as bx
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

from simsopt._core.optimizable import load
from simsopt.field import (
    BiotSavart, Coil, apply_symmetries_to_curves, apply_symmetries_to_currents,
)
from simsopt.geo import SurfaceRZFourier
from simsopt.mhd import Vmec, VirtualCasing, QuasisymmetryRatioResidual
from simsopt.geo.surfaceobjectives import ToroidalFlux


HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent))
from helper_functions import (  # noqa: E402
    PlotConfig,
    optimize_windowpane_currents,
    plot_coil_currents_on_theta_phi_grid,
    plot_relBfinal_norm_modB,
)

ROOT_DIR = (HERE / ".." / "finite_beta_iota0.1_fc_150kA_mpol12_ntor12").resolve()
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VMEC_WORK_DIR = ROOT_DIR / "finite_beta"
VMEC_WORK_DIR.mkdir(parents=True, exist_ok=True)
VMEC_WORK_DIR_VAC = ROOT_DIR / "vacuum"
VMEC_WORK_DIR_VAC.mkdir(parents=True, exist_ok=True)
COILS_DIR = ROOT_DIR / "coils"
COILS_DIR.mkdir(parents=True, exist_ok=True)

# ---- What each input file contributes ----------------------------------------
#
# VMEC_INPUT  (input.nersc_vacuum)
#   Provides VMEC *solver* settings only:
#     NS_ARRAY, NITER_ARRAY, FTOL_ARRAY  – radial grid refinement ladder
#     DELT, NSTEP                         – time-step / convergence controls
#     NTHETA, NZETA                       – internal VMEC poloidal/toroidal grid
#   Everything else is replaced at runtime before vmec.run() is called:
#     boundary (RBC/ZBS)  ← surf_opt.json from SINGLE_STAGE_DIR
#     MPOL / NTOR         ← set to match surf_opt_rz.mpol / surf_opt_rz.ntor
#     PHIEDGE             ← recomputed via ToroidalFlux(boundary, coils)
#     pressure / current  ← overwritten from PROFILES_INPUT (finite-beta run only)
#   This file therefore does NOT need to be regenerated when switching runs.
#
# PROFILES_INPUT  (input.hbt_finite_beta_test)
#   Provides the finite-beta *plasma profiles* only (never run, only parsed):
#     PMASS_TYPE + AM_AUX_S/F  – pressure profile p(s)  [cubic spline]
#     PCURR_TYPE + AC_AUX_S/F  – current profile j(s)   [cubic spline_I]
#     CURTOR                   – total toroidal current  [scaled by CURTOR_SCALE]
#   Its boundary, PHIEDGE, and solver settings are never used.
#
# -------------------------------------------------------------------------------
VMEC_INPUT = (HERE / "equilibria" / "input.vacuum").resolve()
PROFILES_INPUT = (HERE / "equilibria" / "input.hbt_json_pI").resolve()

SINGLE_STAGE_DIR = (HERE / ".." / "nersc_run" / "iota0.1_fcp150kA_vt0.3" / "mpol12_ntor12").resolve()
BS_OPT_JSON = SINGLE_STAGE_DIR / "bs_opt.json"
RESULTS_JSON = SINGLE_STAGE_DIR / "results.json"

TRGT_NPHI = 48
TRGT_NTHETA = 48
SRC_NPHI = 48
SRC_NTHETA = 48

QS_HELICITY_M = 1   # target helicity for QuasisymmetryRatioResidual (m=1, n=0 → QA)
QS_HELICITY_N = 0
QS_SURFACES = np.linspace(0, 1, 11)

# Scale factor applied to the HBT toroidal current (curtor and current profile).
# Pair with PHIEDGE sign so vacuum and finite-beta ι have the same sign:
#   negative PHIEDGE          → CURTOR_SCALE = -1.0
#   positive PHIEDGE          → CURTOR_SCALE = +1.0
# 0.0 → vacuum-like (no plasma current).
CURTOR_SCALE = -1.0

BOOZMN_NAME = "boozmn_postprocessing.nc"
MBOZ = 48
NBOZ = 48

verbose = True


plot_config = PlotConfig(
    dpi=100, titlefontsize=14, axisfontsize=13,
    legendfontsize=12, ticklabelfontsize=12, cbarfontsize=13,
)
profile_plot_config = PlotConfig(
    dpi=300, titlefontsize=14, axisfontsize=13,
    legendfontsize=12, ticklabelfontsize=12, cbarfontsize=13,
)


def compute_residual_stats(bs, surf, vc, label=""):
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n / absn[:, :, None]
    bs.set_points(surf.gamma().reshape((-1, 3)))
    B = bs.B().reshape(n.shape)
    Bn = np.sum(B * unitn, axis=2)
    modB = np.linalg.norm(B, axis=2)
    target = np.asarray(vc.B_external_normal)
    residual = Bn - target
    mean_abs_resid = np.sum(np.abs(residual) * absn) / np.sum(absn)
    max_abs_resid = float(np.max(np.abs(residual)))
    mean_abs_Bn = np.sum(np.abs(Bn) * absn) / np.sum(absn)
    mean_modB = np.sum(modB * absn) / np.sum(absn)
    print(f"[{label}] <|B_coils.n - target|> = {mean_abs_resid:.4e} T   "
          f"max = {max_abs_resid:.4e} T")
    print(f"[{label}] <|B_coils.n|>          = {mean_abs_Bn:.4e} T")
    print(f"[{label}] <|B_coils|>            = {mean_modB:.4e} T")
    return mean_abs_resid, max_abs_resid


def _find_wout(run_dir: Path) -> Path:
    for pattern in ("wout*.nc", "**/wout*.nc"):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No wout*.nc found in {run_dir}")


def _get_or_create_boozmn(run_dir: Path, mboz: int, nboz: int) -> Path:
    boozmn_path = run_dir / BOOZMN_NAME
    wout_path = _find_wout(run_dir)
    if boozmn_path.exists() and boozmn_path.stat().st_mtime >= wout_path.stat().st_mtime:
        return boozmn_path

    b = bx.Booz_xform()
    b.read_wout(str(wout_path))
    b.mboz = mboz
    b.nboz = nboz
    b.run()
    b.write_boozmn(str(boozmn_path))
    return boozmn_path


def _read_iota_from_wout(wout_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotational transform with a sign convention consistent across vacuum and
    finite-beta runs (see finite_beta_test/plot_profiles.py).
    """
    f = netcdf_file(str(wout_path), mmap=False)
    ns = int(f.variables["ns"][()])
    iotaf = np.asarray(f.variables["iotaf"][()], dtype=float)
    signgs = float(f.variables["signgs"][()])
    phi = np.asarray(f.variables["phi"][()], dtype=float)
    f.close()
    s = np.linspace(0.0, 1.0, ns)
    iota = iotaf * signgs * np.sign(phi[-1])
    return s, iota


def _read_profiles_from_boozmn(boozmn_path: Path, run_dir: Path) -> dict[str, np.ndarray]:
    b = bx.Booz_xform()
    b.read_boozmn(str(boozmn_path))

    bmnc = np.asarray(b.bmnc_b)
    xn = np.asarray(b.xn_b)
    s_qs = np.asarray(b.s_b)

    numerator = np.sum(bmnc[xn != 0, :] ** 2, axis=0)
    denominator = np.sum(bmnc**2, axis=0)
    qs_error = np.sqrt(numerator / denominator)

    s_iota, iota = _read_iota_from_wout(_find_wout(run_dir))

    return {"s": s_iota, "iota": iota, "qs_s": s_qs, "qs_error": qs_error}


def _style_profile_axis(ax, cfg: PlotConfig):
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.tick_params(axis="both", labelsize=cfg.ticklabelfontsize)


def _parse_namelist_array(text: str, name: str) -> np.ndarray:
    """
    Parse a VMEC namelist array assignment, including wrapped continuation lines.
    """
    lines = text.splitlines()
    start_re = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(.*)$", re.IGNORECASE)
    next_assign_re = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=")

    start_idx = None
    first_rhs = ""
    for i, line in enumerate(lines):
        m = start_re.match(line)
        if m:
            start_idx = i
            first_rhs = m.group(1)
            break
    if start_idx is None:
        raise ValueError(f"Could not find assignment for {name!r}")

    rhs_chunks = [first_rhs]
    for j in range(start_idx + 1, len(lines)):
        raw = lines[j]
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("&", "/")):
            break
        if next_assign_re.match(raw):
            break
        if "!" in line:
            line = line.split("!", 1)[0].strip()
        rhs_chunks.append(line)

    rhs = " ".join(rhs_chunks)
    if "!" in rhs:
        rhs = rhs.split("!", 1)[0]
    # VMEC inputs may use comma- or whitespace-separated values.
    parts = re.split(r"[,\s]+", rhs.strip())
    return np.asarray([float(p) for p in parts if p], dtype=float)


def _read_input_profile_knots(hbt_input_path: Path, curtor_scale: float) -> dict[str, np.ndarray]:
    """
    Read exact spline knots passed to VMEC in the profiles input:
      - pressure: am_aux_s / am_aux_f
      - enclosed current: ac_aux_s / ac_aux_f
    """
    raw = hbt_input_path.read_text()
    s_pressure = _parse_namelist_array(raw, "am_aux_s")
    p_pa = _parse_namelist_array(raw, "am_aux_f")
    s_current = _parse_namelist_array(raw, "ac_aux_s")
    I_enc_kA = _parse_namelist_array(raw, "ac_aux_f") / 1e3 * curtor_scale

    if s_pressure.size != p_pa.size:
        raise ValueError(
            f"am_aux_s ({s_pressure.size}) and am_aux_f ({p_pa.size}) must have same length."
        )
    if s_current.size != I_enc_kA.size:
        raise ValueError(
            f"ac_aux_s ({s_current.size}) and ac_aux_f ({I_enc_kA.size}) must have same length."
        )

    return {
        "s_pressure": s_pressure,
        "s_current": s_current,
        "p_pa": p_pa,
        "I_enc_kA": I_enc_kA,
    }


def plot_vacuum_vs_finite_beta_profiles(
    run_dirs: dict[str, Path],
    hbt_input: Path,
    plots_dir: Path,
    curtor_scale: float,
    cfg: PlotConfig,
) -> None:
    """Compare vacuum vs finite-beta Boozer QS error and rotational transform."""
    colors = {"Vacuum": "k", r"Finite-$\beta$": "tab:red"}
    linestyles = {"Vacuum": "-", r"Finite-$\beta$": "--"}

    datasets: dict[str, dict[str, np.ndarray]] = {}
    for label, run_dir in run_dirs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        boozmn_path = _get_or_create_boozmn(run_dir, MBOZ, NBOZ)
        datasets[label] = _read_profiles_from_boozmn(boozmn_path, run_dir)
        print(f"[{label}] Loaded Boozer data from {boozmn_path}")

    hbt = _read_input_profile_knots(hbt_input, curtor_scale)
    print(f"[HBT profiles] Loaded {hbt_input}")

    fig_qs, ax_qs = plt.subplots(figsize=(8, 3), constrained_layout=True)
    for label, data in datasets.items():
        ax_qs.plot(
            data["qs_s"],
            data["qs_error"],
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=2.2,
            label=label,
        )

    ax_qs.set_xlabel(
        r"$s$",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
    )
    ax_qs.set_ylabel(
        r"$\left(\sum_{m,n\neq 0} B_{mn}^2 \,/\, \sum_{m,n} B_{mn}^2\right)^{1/2}$",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
    )
    ax_qs.legend(fontsize=cfg.legendfontsize, framealpha=0.85)
    _style_profile_axis(ax_qs, cfg)

    qs_plot_path = plots_dir / "qs_error_comparison.png"
    fig_qs.savefig(qs_plot_path, dpi=cfg.dpi, bbox_inches="tight")
    fig_qs.savefig(qs_plot_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_qs)
    print(f"Saved -> {qs_plot_path}")

    fig_hw, (ax_hbt, ax_iota_hw) = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.6},
    )

    ax_hbt_twin = ax_hbt.twinx()
    ax_hbt.plot(
        hbt["s_pressure"],
        hbt["p_pa"],
        color="tab:orange",
        linewidth=3.5,
        linestyle="-",
        label=r"Pressure $p$ [Pa]",
    )
    ax_hbt_twin.plot(
        hbt["s_current"],
        hbt["I_enc_kA"],
        color="tab:blue",
        linewidth=3.5,
        linestyle="--",
        label=r"$I_\mathrm{enc}$ [kA]",
    )
    ax_hbt.set_xlabel(
        r"$s$",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
    )
    ax_hbt.set_ylabel(
        r"$p$ [Pa]",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
        color="tab:orange",
    )
    ax_hbt.tick_params(
        axis="y",
        labelsize=cfg.ticklabelfontsize,
        labelcolor="tab:orange",
    )
    ax_hbt_twin.set_ylabel(
        r"$I_\mathrm{enc}$ [kA]",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
        color="tab:blue",
    )
    ax_hbt_twin.tick_params(
        axis="y",
        labelsize=cfg.ticklabelfontsize,
        labelcolor="tab:blue",
    )
    ax_hbt.tick_params(axis="x", labelsize=cfg.ticklabelfontsize)
    _style_profile_axis(ax_hbt, cfg)
    ax_hbt.set_ylim(bottom=0.0)
    ax_hbt_twin.set_ylim(bottom=0.0)

    for label, data in datasets.items():
        ax_iota_hw.plot(
            data["s"][2:],
            data["iota"][2:],
            color=colors[label],
            linestyle=linestyles[label],
            linewidth=3.5,
            label=label,
        )
    ax_iota_hw.set_xlabel(
        r"$s$",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
    )
    ax_iota_hw.set_ylabel(
        r"$\iota$",
        fontsize=cfg.axisfontsize,
        fontweight="bold",
    )
    ax_iota_hw.legend(fontsize=cfg.legendfontsize, framealpha=0.85)
    _style_profile_axis(ax_iota_hw, cfg)

    combo_path = plots_dir / "hbt_profiles_and_rotational_transform.png"
    fig_hw.savefig(combo_path, dpi=cfg.dpi, bbox_inches="tight")
    fig_hw.savefig(combo_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_hw)
    print(f"Saved -> {combo_path}")


def build_full_bs(base_coils, surf_plasma):
    curves = apply_symmetries_to_curves(
        base_curves=[c.curve for c in base_coils],
        nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym,
    )
    currents = apply_symmetries_to_currents(
        base_currents=[c.current for c in base_coils],
        nfp=surf_plasma.nfp, stellsym=surf_plasma.stellsym,
    )
    return BiotSavart([Coil(cu, cr) for cu, cr in zip(curves, currents)])


def main():
    for path in (VMEC_INPUT, PROFILES_INPUT):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

    bs = load(str(BS_OPT_JSON))

    # Load the optimised boundary from the single-stage run and convert from
    # SurfaceXYZTensorFourier → SurfaceRZFourier (the representation VMEC uses).
    # This replaces the coarser mpol=6/ntor=6 shape stored in input.nersc_vacuum.
    _surf_opt_raw = load(str(SINGLE_STAGE_DIR / "surf_opt.json"))
    surf_opt_rz = _surf_opt_raw.to_RZFourier()
    print(f"Loaded boundary from surf_opt.json: "
          f"mpol={surf_opt_rz.mpol}, ntor={surf_opt_rz.ntor}, nfp={surf_opt_rz.nfp}")

    # ------------------------------------------------------------------
    # Vacuum VMEC run (zero pressure, zero current) – QS baseline
    # ------------------------------------------------------------------
    local_input_vac = (VMEC_WORK_DIR_VAC / VMEC_INPUT.name).resolve()
    shutil.copy(VMEC_INPUT, local_input_vac)
    os.chdir(VMEC_WORK_DIR_VAC)
    print(f"Running VMEC (vacuum) from input file {local_input_vac}...")
    vmec_vac = Vmec(str(local_input_vac), verbose=verbose, range_surface="half period")
    # Replace the template boundary with the actual optimised shape and update
    # the internal mpol/ntor so VMEC resolves all modes in the new boundary.
    vmec_vac.boundary = surf_opt_rz
    vmec_vac.indata.mpol = surf_opt_rz.mpol
    vmec_vac.indata.ntor = surf_opt_rz.ntor
    vmec_vac.indata.nzeta = TRGT_NPHI
    vmec_vac.indata.ntheta = TRGT_NTHETA
    phiedge = ToroidalFlux(vmec_vac.boundary, bs).J()
    vmec_vac.indata.phiedge = phiedge
    vmec_vac.run()
    print(f"  wout: {vmec_vac.output_file}")
    print(f"  volavgB={vmec_vac.wout.volavgB:.4e} T  "
          f"ctor={float(vmec_vac.wout.ctor):.4e} A  "
          f"betatot={vmec_vac.wout.betatotal:.4e}")
    qs_vac = QuasisymmetryRatioResidual(
        vmec_vac, QS_SURFACES,
        helicity_m=QS_HELICITY_M, helicity_n=QS_HELICITY_N,
    )
    print(f"  QS residual (vacuum):     {qs_vac.total():.4e}  "
          f"(helicity m={QS_HELICITY_M}, n={QS_HELICITY_N})")

    # ------------------------------------------------------------------
    # Finite-beta VMEC run
    # Same solver settings and boundary as the vacuum run (VMEC_INPUT +
    # surf_opt_rz), but with pressure and current profiles grafted from
    # PROFILES_INPUT via vmec.indata before running.
    # ------------------------------------------------------------------
    local_input_fb = (VMEC_WORK_DIR / VMEC_INPUT.name).resolve()
    shutil.copy(VMEC_INPUT, local_input_fb)
    os.chdir(VMEC_WORK_DIR)
    print(f"Running VMEC (finite beta) from input file {local_input_fb}...")
    # Parse PROFILES_INPUT for its profile arrays only — this Vmec instance is never run.
    vmec_hbt = Vmec(str(PROFILES_INPUT), verbose=verbose)
    pmass_type = vmec_hbt.indata.pmass_type
    am_aux_s = vmec_hbt.indata.am_aux_s.copy()   # pressure spline knot values s
    am_aux_f = vmec_hbt.indata.am_aux_f.copy()   # pressure spline knot values p(s)
    pcurr_type = vmec_hbt.indata.pcurr_type
    ac_aux_s = vmec_hbt.indata.ac_aux_s.copy()   # current spline knot values s
    ac_aux_f = vmec_hbt.indata.ac_aux_f.copy()   # current spline knot values j(s)
    curtor = vmec_hbt.indata.curtor               # total toroidal current [A]

    vmec = Vmec(str(local_input_fb), verbose=verbose, range_surface="half period")
    vmec.boundary = surf_opt_rz
    vmec.indata.mpol = surf_opt_rz.mpol
    vmec.indata.ntor = surf_opt_rz.ntor
    vmec.indata.nzeta = TRGT_NPHI
    vmec.indata.ntheta = TRGT_NTHETA
    vmec.indata.phiedge = phiedge

    # Pressure profile: CUBIC_SPLINE
    vmec.indata.pmass_type = pmass_type
    vmec.indata.am_aux_s = am_aux_s
    vmec.indata.am_aux_f = am_aux_f

    # Current profile: cubic_spline_I  (scaled by CURTOR_SCALE)
    vmec.indata.pcurr_type = pcurr_type
    vmec.indata.ac_aux_s = ac_aux_s
    vmec.indata.ac_aux_f = ac_aux_f * CURTOR_SCALE
    vmec.indata.curtor = curtor * CURTOR_SCALE
    print(f"  CURTOR_SCALE={CURTOR_SCALE}  →  curtor={vmec.indata.curtor:.4e} A")

    vmec.run()
    print(f"  wout:    {vmec.output_file}")
    print(f"  nfp={vmec.wout.nfp}, mpol={vmec.wout.mpol}, ntor={vmec.wout.ntor}")
    print(f"  volavgB={vmec.wout.volavgB:.4e} T")
    print(f"  ctor   ={float(vmec.wout.ctor):.4e} A")
    print(f"  betatot={vmec.wout.betatotal:.4e}")
    qs_fb = QuasisymmetryRatioResidual(
        vmec, QS_SURFACES,
        helicity_m=QS_HELICITY_M, helicity_n=QS_HELICITY_N,
    )
    print(f"  QS residual (finite beta): {qs_fb.total():.4e}  "
          f"(helicity m={QS_HELICITY_M}, n={QS_HELICITY_N})")
    print(f"\n  QS change due to plasma current: "
          f"{qs_vac.total():.4e} (vacuum)  →  {qs_fb.total():.4e} (finite beta)")

    print("Running VirtualCasing...")
    vc = VirtualCasing.from_vmec(
        vmec,
        src_nphi=SRC_NPHI,
        src_ntheta=SRC_NTHETA,
        trgt_nphi=TRGT_NPHI,
        trgt_ntheta=TRGT_NTHETA,
        filename=str(HERE / "vcasing_nersc_finite_beta.nc"),
    )
    print(f"  vc.nfp={vc.nfp}, trgt_nphi={vc.trgt_nphi}, trgt_ntheta={vc.trgt_ntheta}")
    print(f"  <|B_external_normal|> = {np.mean(np.abs(vc.B_external_normal)):.4e} T")
    print(f"  max |B_external_normal| = {np.max(np.abs(vc.B_external_normal)):.4e} T")

    # Sanity check: B_total.n should be ~0 on a flux surface.
    if (vc.src_nphi == vc.trgt_nphi) and (vc.src_ntheta == vc.trgt_ntheta):
        B_tot_n = np.sum(vc.B_total * vc.unit_normal, axis=2)
        print(f"  <|B_total.n|>        = {np.mean(np.abs(B_tot_n)):.4e} T "
              f"(flux surface check)")

    surf_plasma = SurfaceRZFourier.from_wout(
        vmec.output_file, nphi=TRGT_NPHI, ntheta=TRGT_NTHETA,
        range="half period",
    )
    print(f"  surf_plasma: nfp={surf_plasma.nfp}, stellsym={surf_plasma.stellsym}, "
          f"grid={surf_plasma.gamma().shape[:2]}")
    assert surf_plasma.gamma().shape[:2] == vc.B_external_normal.shape

    # ------------------------------------------------------------------
    # Load nersc_run coils
    # ------------------------------------------------------------------
    with open(RESULTS_JSON, 'r') as f:
        results = json.load(f)
    if "ntf" not in results and "graph" in results:
        results = results["graph"]

    # Reconstruct the elliptical winding surface (vacuum vessel) from the
    # VV parameters stored in results.json – needed for the current grid plot.
    VV = SurfaceRZFourier(
        mpol=1, ntor=0,
        nfp=int(results["surf_nfp"]), stellsym=True,
        quadpoints_phi=np.linspace(0, 1, 64, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 64, endpoint=False),
    )
    VV.set_rc(0, 0, results["VV_R0"])
    VV.set_rc(1, 0, results["VV_a"])
    VV.set_zs(1, 0, results["VV_b"])

    coils_full = bs.coils
    ntf_total = int(results["ntf"]) * 2 * int(results["surf_nfp"])
    tf_coils_full = coils_full[:ntf_total]
    dipole_coils_full = coils_full[ntf_total:]
    nsym = 2 * int(results["surf_nfp"])
    ntf_base = len(tf_coils_full) // nsym
    ndip_base = len(dipole_coils_full) // nsym
    tf_coils = tf_coils_full[:ntf_base]
    dipole_coils = dipole_coils_full[:ndip_base]
    print(f"  Loaded: {len(tf_coils_full)} TF + {len(dipole_coils_full)} dipole "
          f"(full); using {ntf_base}+{ndip_base} as half-period base")
    assert surf_plasma.nfp == int(results["surf_nfp"])

    for c in tf_coils:
        c.curve.fix_all()
        c.current.fix_all()
    for c in dipole_coils:
        c.curve.fix_all()

    dip_init = np.array([c.current.get_value() for c in dipole_coils])
    tf_init = np.array([c.current.get_value() for c in tf_coils])
    print(f"  dipole init currents [A]: min={dip_init.min():.3e}, "
          f"max={dip_init.max():.3e}, median={np.median(dip_init):.3e}")
    print(f"  TF init currents [A]:     min={tf_init.min():.3e}, "
          f"max={tf_init.max():.3e}")

    # ------------------------------------------------------------------
    # Initial diagnostics
    # ------------------------------------------------------------------
    bs_initial = build_full_bs(tf_coils + dipole_coils, surf_plasma)
    print("\n--- Initial (pre-optimization) ---")
    init_mean, init_max = compute_residual_stats(
        bs_initial, surf_plasma, vc, label="initial")

    plot_relBfinal_norm_modB(bs_initial, surf_plasma, str(PLOTS_DIR),
                             "initial", plot_config, vc=vc)
    plot_relBfinal_norm_modB(bs_initial, surf_plasma, str(PLOTS_DIR),
                             "initial_noVC", plot_config)
    plot_coil_currents_on_theta_phi_grid(
        dipole_coils, VV, str(PLOTS_DIR), "initial", plot_config)

    bs_initial.set_points(surf_plasma.gamma().reshape((-1, 3)))
    Bbs_init = bs_initial.B().reshape(surf_plasma.gamma().shape)
    BdotN_init = np.sum(Bbs_init * surf_plasma.unitnormal(), axis=2) - vc.B_external_normal
    surf_plasma.to_vtk(str(COILS_DIR / "surf_initial"),
                       extra_data={"B_N": BdotN_init[:, :, None]})
    print(f"  Saved surf_initial.vts with B_N to {COILS_DIR}")

    # ------------------------------------------------------------------
    # Stage-2 dipole optimization against VC target
    # ------------------------------------------------------------------
    print("\n--- Running dipole-only stage-2 optimization ---")
    res, bs_opt = optimize_windowpane_currents(
        base_wp_coils=dipole_coils,
        base_tf_coils=tf_coils,
        surf_plasma=surf_plasma,
        definition="quadratic flux",
        precomputed=True,
        maxiter=300,
        current_threshold=5e5,
        current_weight=0,
        num_fixed=len(tf_coils),
        verbose=verbose,
        target_normal=vc.B_external_normal,
    )
    print(f"Optimizer: {res.message}")
    print(f"Iterations: {res.nit},  final J = {res.fun:.6e}")

    dip_fin = np.array([c.current.get_value() for c in dipole_coils])
    print(f"  dipole final currents [A]: min={dip_fin.min():.3e}, "
          f"max={dip_fin.max():.3e}, median={np.median(dip_fin):.3e}")

    print("\n--- Final (post-optimization) ---")
    fin_mean, fin_max = compute_residual_stats(
        bs_opt, surf_plasma, vc, label="final")

    plot_relBfinal_norm_modB(bs_opt, surf_plasma, str(PLOTS_DIR), "final",
                             plot_config, vc=vc)
    plot_relBfinal_norm_modB(bs_opt, surf_plasma, str(PLOTS_DIR), "final_noVC",
                             plot_config)
    plot_coil_currents_on_theta_phi_grid(
        dipole_coils, VV, str(PLOTS_DIR), "final", plot_config)

    bs_opt.set_points(surf_plasma.gamma().reshape((-1, 3)))
    Bbs_opt = bs_opt.B().reshape(surf_plasma.gamma().shape)
    BdotN_opt = np.sum(Bbs_opt * surf_plasma.unitnormal(), axis=2) - vc.B_external_normal
    surf_plasma.to_vtk(str(COILS_DIR / "surf_final"),
                       extra_data={"B_N": BdotN_opt[:, :, None]})
    print(f"  Saved surf_final.vts with B_N to {COILS_DIR}")

    print(f"\nMean residual drop: {init_mean:.4e} -> {fin_mean:.4e} "
          f"(factor {init_mean/fin_mean:.2f}x)")
    print(f"Max  residual drop: {init_max:.4e} -> {fin_max:.4e} "
          f"(factor {init_max/fin_max:.2f}x)")

    print("\n--- QS error summary ---")
    print(f"  QS residual (vacuum):      {qs_vac.total():.4e}  "
          f"(helicity m={QS_HELICITY_M}, n={QS_HELICITY_N})")
    print(f"  QS residual (finite beta): {qs_fb.total():.4e}  "
          f"(helicity m={QS_HELICITY_M}, n={QS_HELICITY_N})")
    print(f"  QS change due to plasma current: "
          f"{qs_vac.total():.4e}  →  {qs_fb.total():.4e}  "
          f"(factor {qs_fb.total()/qs_vac.total():.2f}x)")

    bs_opt.save(str(COILS_DIR / "biot_savart_opt.json"))
    print(f"Saved optimized BiotSavart to {COILS_DIR / 'biot_savart_opt.json'}")

    print("\n--- Vacuum vs finite-beta profile comparison plots ---")
    plot_vacuum_vs_finite_beta_profiles(
        run_dirs={
            "Vacuum": VMEC_WORK_DIR_VAC,
            r"Finite-$\beta$": VMEC_WORK_DIR,
        },
        hbt_input=PROFILES_INPUT,
        plots_dir=PLOTS_DIR,
        curtor_scale=CURTOR_SCALE,
        cfg=profile_plot_config,
    )

if __name__ == "__main__":
    main()
