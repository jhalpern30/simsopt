#!/usr/bin/env python3
"""
Scan outboard-midplane ripple δ vs. number of TF coils for three coil layouts.

Inputs (hardcoded in script):
  - VV and TF geometry
  - Circular tokamak plasma surface (R0, a) and scan grid (ntf = 5–10,
    npoloidal = 10 window-pane coils per half-period)

Steps:
  1. Build plasma and vacuum-vessel surfaces; save cross-section check plot
  2. For each ntf: run three cases —
     TF only; TF + dense dipole array; TF + sparse array (no OMP row)
  3. Optimise windowpane currents (dense/sparse) to minimise local B.n on plasma
  4. Compute δ = (B_max − B_min)/(B_max + B_min) at the outboard midplane
  5. Write per-case field, current, and coil VTK diagnostics
  6. Aggregate δ vs. N_TF into a summary plot and CSV

Outputs (under ``ripple_scan/``):
  - ``cross_section.png``
  - ``TF_only/``, ``TF_dense/``, ``TF_sparse/`` – per-ntf relBn/modB plots,
    coil-current maps (WP cases), and ``TF_coils_*.vtu`` / ``WP_coils_*.vtu``
  - ``delta_results.csv``, ``delta_vs_ntf.png``
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── make sure helper_functions.py is importable ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.field import coils_via_symmetries, BiotSavart
from helper_functions import (
    PlotConfig,
    coil_currents_on_theta_phi_grid,
    generate_tf_array,
    generate_windowpane_array,
    optimize_windowpane_currents,
    plot_relBfinal_norm_modB,
)

# ═══════════════════════════════════════════════════════════════════════════
# ── Parameters ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# Vacuum-vessel geometry (from results.json)
VV_R0 = 1.042435734478781
VV_a  = 0.24958796105352915
VV_b  = 0.2823287702903717

# TF coil geometry (from results.json – TF_a/TF_b are coil bore half-axes)
TF_a  = 0.3995879610535291
TF_b  = 0.43232877029037164

# Plasma surface (simple circular tokamak)
R0_plasma = 1.0
a_plasma  = 0.14          # minor radius [m]  (user wrote "0.14 cm" → 0.14 m)

# Field & scan
mu0          = 4e-7 * np.pi   # T·m/A  (= 4π×10⁻⁷)
B_T          = 0.5            # on-axis toroidal field [T]
ntf_values   = list(range(5, 11))  # ntf per half-period → total TF = 2×ntf
nfp          = 1
stellsym     = True

# Dipole array
npoloidal         = 10     # fixed poloidal count per half-period
wp_fil_spacing    = 0.05   # filament-to-filament gap [m]
half_per_spacing  = 0.05   # gap at half-period boundary [m]
wp_n              = 4      # superellipse exponent (2 = ellipse)

# Optimisation
current_threshold = 1e6    # [A] – hard threshold for current penalty
current_weight    = 1e-12  # weight on current penalty
maxiter           = 300    # L-BFGS-B iterations
definition        = "local"

# Surface quadpoints
nphi_plasma   = 64
ntheta_plasma = 32
nphi_VV       = 256    # fine φ-grid so the interpolator always covers coil positions

numquadpoints = 32     # quadrature points per coil curve

# ═══════════════════════════════════════════════════════════════════════════
# ── Output directories ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

base_dir = os.path.join(_HERE, "ripple_scan")
dirs = {
    "TF_only":   os.path.join(base_dir, "TF_only"),
    "TF_dense":  os.path.join(base_dir, "TF_dense"),
    "TF_sparse": os.path.join(base_dir, "TF_sparse"),
}
for d in [base_dir, *dirs.values()]:
    os.makedirs(d, exist_ok=True)

plot_config = PlotConfig()

# ═══════════════════════════════════════════════════════════════════════════
# ── Surface factories ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def make_plasma_surf(nfp=1, nphi=64, ntheta=32):
    """Circular tokamak plasma surface for squared-flux optimisation."""
    qphi   = np.linspace(0, 1 / (2 * nfp), nphi,   endpoint=False)
    qtheta = np.linspace(0, 1,              ntheta, endpoint=False)
    s = SurfaceRZFourier(
        nfp=nfp, stellsym=True, mpol=1, ntor=0,
        quadpoints_phi=qphi, quadpoints_theta=qtheta,
    )
    s.set_rc(0, 0, R0_plasma)
    s.set_rc(1, 0, a_plasma)
    s.set_zs(1, 0, a_plasma)
    return s


def make_VV_surf(nfp=1, nphi=256, ntheta=64):
    """Elliptical vacuum-vessel winding surface.

    Uses endpoint=True in φ so the RegularGridInterpolator inside
    generate_windowpane_array never receives out-of-bounds queries,
    even for large ntoroidal where the last coil centre sits close to
    φ = π (= 0.5 in fractional units).
    """
    # Include both 0 and 0.5 so the interpolation range is [0, 0.5].
    qphi   = np.linspace(0, 1 / (2 * nfp), nphi,   endpoint=True)
    qtheta = np.linspace(0, 1,              ntheta, endpoint=False)
    VV = SurfaceRZFourier(
        nfp=nfp, stellsym=True, mpol=1, ntor=0,
        quadpoints_phi=qphi, quadpoints_theta=qtheta,
    )
    VV.set_rc(0, 0, VV_R0)
    VV.set_rc(1, 0, VV_a)
    VV.set_zs(1, 0, VV_b)
    return VV

# ═══════════════════════════════════════════════════════════════════════════
# ── Cross-section verification plot ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def save_cross_section_plot():
    surf_check = make_plasma_surf()
    VV_check   = make_VV_surf()

    fig, ax = plt.subplots(figsize=(6, 6))
    phi_slices = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for phi_s in phi_slices:
        cs  = surf_check.cross_section(phi_s)
        rs  = np.sqrt(cs[:, 0] ** 2 + cs[:, 1] ** 2)
        rs  = np.append(rs, rs[0])
        zs  = np.append(cs[:, 2], cs[0, 2])
        cs2 = VV_check.cross_section(phi_s)
        rs2 = np.sqrt(cs2[:, 0] ** 2 + cs2[:, 1] ** 2)
        rs2 = np.append(rs2, rs2[0])
        zs2 = np.append(cs2[:, 2], cs2[0, 2])
        ax.plot(rs, zs, label=rf"$\phi={phi_s/np.pi:.2f}\pi$")
        ax.plot(rs2, zs2, "k", alpha=0.4)

    ax.set_xlabel("R [m]", fontsize=plot_config.axisfontsize, fontweight="bold")
    ax.set_ylabel("Z [m]", fontsize=plot_config.axisfontsize, fontweight="bold")
    ax.set_title(
        f"Plasma surface (coloured) and VV (black)\n"
        f"R0 = {R0_plasma:.3f} m,  a = {a_plasma:.3f} m,  "
        f"aspect = {R0_plasma/a_plasma:.2f}",
        fontsize=plot_config.titlefontsize, fontweight="bold",
    )
    ax.legend(fontsize=plot_config.legendfontsize)
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(base_dir, "cross_section.png")
    plt.savefig(path, dpi=plot_config.dpi)
    plt.close()
    print(f"  Cross-section plot saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ── Ripple delta on outboard midplane ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_omp(bs, R0, a, npoints=720):
    """
    Ripple parameter δ = (B_max − B_min) / (B_max + B_min) evaluated at the
    outboard midplane (R = R0 + a, Z = 0), scanning φ over the full torus.
    """
    phi_vals = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
    R_omp = R0 + a
    pts = np.column_stack([
        R_omp * np.cos(phi_vals),
        R_omp * np.sin(phi_vals),
        np.zeros(npoints),
    ])
    bs.set_points(pts)
    modB = np.linalg.norm(bs.B(), axis=1)
    return (modB.max() - modB.min()) / (modB.max() + modB.min())


# ═══════════════════════════════════════════════════════════════════════════
# ── Coil-current plot (nfp=1 aware) ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def plot_wp_currents(base_wp_coils, VV, out_dir, label, cfg):
    """Scatter plot of windowpane coil currents on the (φ, θ) winding surface."""
    data     = coil_currents_on_theta_phi_grid(base_wp_coils, VV)
    currents = data[:, 0] / 1e3                        # → kA
    phis     = data[:, 1]
    thetas   = np.mod(data[:, 2], 2 * np.pi)

    vmax  = max(np.max(np.abs(currents)), 1e-6)
    norm  = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap  = plt.cm.seismic
    colors = cmap(norm(currents))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        phis / (2 * np.pi), thetas / (2 * np.pi),
        edgecolors=colors, facecolors="none", s=200, linewidths=1.5,
    )
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel("WP Currents [kA]", fontsize=cfg.cbarfontsize, fontweight="bold")
    cbar.ax.tick_params(axis="y", labelsize=cfg.ticklabelfontsize)

    half_period = 1.0 / (2 * nfp)   # = 0.5 for nfp=1
    ax.set_xlim(-0.005, half_period + 0.005)
    ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel(r"$\phi/2\pi$", fontsize=cfg.axisfontsize, fontweight="bold")
    ax.set_ylabel(r"$\theta/2\pi$", fontsize=cfg.axisfontsize, fontweight="bold")
    ax.set_title(f"WP Coil Currents – {label}", fontsize=cfg.titlefontsize, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(out_dir, f"coil_currents_{label}.png")
    plt.savefig(out, dpi=cfg.dpi)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# ── VTK coil output for Paraview ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def save_coils_vtk(coil_list, vtk_path):
    """
    Write a list of coils to a VTK polydata (.vtp) file for Paraview.

    Each curve segment carries a scalar 'current_A' so coils can be
    coloured by current.  curves_to_vtk with close=True appends the first
    point at the end of each polyline (numquadpoints+1 entries per curve).
    """
    if not coil_list:
        return
    curves = [c.curve for c in coil_list]
    npts   = [len(c.curve.gamma()) + 1 for c in coil_list]
    currents_flat = np.concatenate([
        np.full(n, c.current.get_value())
        for c, n in zip(coil_list, npts)
    ])
    curves_to_vtk(curves, vtk_path, close=True,
                  extra_data={"current_A": currents_flat})


def save_tf_wp_vtk(all_coils, n_tf_full, out_dir, label):
    """
    Save TF and windowpane (shaping) coils as separate VTK files.

    In the ordered coil list returned by optimize_windowpane_currents,
    apply_symmetries_to_curves is called once per base TF coil (nfp=1,
    stellsym=True → 2 copies each), so the first n_tf_full = 2·ntf entries
    are TF coils and the remainder are windowpane coils.

    Output files:
        <out_dir>/TF_coils_<label>.vtp
        <out_dir>/WP_coils_<label>.vtp   (omitted when coil list is TF-only)
    """
    tf_coils = all_coils[:n_tf_full]
    wp_coils = all_coils[n_tf_full:]
    save_coils_vtk(tf_coils, os.path.join(out_dir, f"TF_coils_{label}"))
    save_coils_vtk(wp_coils, os.path.join(out_dir, f"WP_coils_{label}"))


# ═══════════════════════════════════════════════════════════════════════════
# ── Main scan ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TF Ripple Scan")
print(f"  VV:     R0={VV_R0:.3f}  a={VV_a:.3f}  b={VV_b:.3f}")
print(f"  Plasma: R0={R0_plasma:.3f}  a={a_plasma:.3f} m  "
      f"(aspect = {R0_plasma/a_plasma:.2f})")
print(f"  nfp = {nfp},  stellsym = {stellsym}")
print(f"  ntf scan: {ntf_values[0]} → {ntf_values[-1]}  "
      f"(total TF: {2*ntf_values[0]} → {2*ntf_values[-1]})")
print("=" * 60)

# Save the cross-section verification plot first
save_cross_section_plot()

delta_tf     = []
delta_dense  = []
delta_sparse = []

t_total = time.time()

for ntf in ntf_values:
    print(f"\n{'─'*60}")
    print(f"ntf = {ntf}  │  total TF coils = {2*nfp*ntf}  │  "
          f"ntoroidal WP = {2*ntf}")
    print(f"{'─'*60}")

    ntoroidal = 2 * ntf   # toroidal WP coils per half-period

    # TF current: toroidal-solenoid approximation
    #   B_T = μ₀ · N_total · I / (2π R0)  →  I = B_T · 2π R0 / (μ₀ · N_total)
    N_total_TF = 2 * nfp * ntf          # stellsym → 2·nfp copies
    TF_current = B_T * 2 * np.pi * R0_plasma / (mu0 * N_total_TF)
    print(f"  TF current per base coil: {TF_current:,.0f} A")

    # Shared surfaces
    surf_plasma = make_plasma_surf(nfp=nfp, nphi=nphi_plasma, ntheta=ntheta_plasma)
    VV          = make_VV_surf(nfp=nfp, nphi=nphi_VV)

    # ── Case 1: TF only ──────────────────────────────────────────────────
    print("  [1/3] TF only …", flush=True)
    t0 = time.time()

    _, base_tf_coils = generate_tf_array(
        VV, ntf, VV_R0, TF_a, TF_b, TF_current,
        fixed_geo_tfs=True, numquadpoints=numquadpoints, tf_coil_radius=0,
    )
    tf_full = coils_via_symmetries(
        [c.curve    for c in base_tf_coils],
        [c.current  for c in base_tf_coils],
        nfp, stellsym,
    )
    bs_tf = BiotSavart(tf_full)

    # Relative Bnormal on plasma surface
    plot_relBfinal_norm_modB(
        bs_tf, surf_plasma,
        dirs["TF_only"], f"ntf{ntf:02d}", plot_config,
    )
    d_tf = compute_delta_omp(bs_tf, R0_plasma, a_plasma)
    delta_tf.append(d_tf)
    save_coils_vtk(tf_full, os.path.join(dirs["TF_only"], f"TF_coils_ntf{ntf:02d}"))
    print(f"    δ = {d_tf:.5f}  ({100*d_tf:.3f} %)   [{time.time()-t0:.1f} s]")

    # ── Case 2: TF + dense WP array ──────────────────────────────────────
    print("  [2/3] TF + dense …", flush=True)
    t0 = time.time()

    _, base_tf_coils_dense = generate_tf_array(
        VV, ntf, VV_R0, TF_a, TF_b, TF_current,
        fixed_geo_tfs=True, numquadpoints=numquadpoints, tf_coil_radius=0,
    )
    base_wp_dense, Rpol, Rtor_min, Rtor_max, npol_act, ntor_act = \
        generate_windowpane_array(
            VV, None, wp_fil_spacing, half_per_spacing, wp_n,
            numquadpoints=numquadpoints, order=12, verbose=True,
            nwps_poloidal_target=npoloidal,
            nwps_toroidal_target=ntoroidal,
        )
    print(f"    WP base coils: {len(base_wp_dense)} "
          f"(npol={npol_act}, ntor={ntor_act})")

    _, bs_dense = optimize_windowpane_currents(
        base_wp_dense, base_tf_coils_dense, surf_plasma,
        definition=definition, precomputed=True,
        maxiter=maxiter,
        current_threshold=current_threshold,
        current_weight=current_weight,
        num_fixed=ntf,      # fix ALL TF currents
        verbose=True,
    )

    d_dense = compute_delta_omp(bs_dense, R0_plasma, a_plasma)
    delta_dense.append(d_dense)
    save_tf_wp_vtk(bs_dense.coils, 2 * nfp * ntf, dirs["TF_dense"], f"ntf{ntf:02d}")

    plot_wp_currents(base_wp_dense, VV, dirs["TF_dense"], f"ntf{ntf:02d}", plot_config)
    plot_relBfinal_norm_modB(
        bs_dense, surf_plasma,
        dirs["TF_dense"], f"ntf{ntf:02d}", plot_config,
    )
    print(f"    δ = {d_dense:.5f}  ({100*d_dense:.3f} %)   [{time.time()-t0:.1f} s]")

    # ── Case 3: TF + sparse WP array (no outboard-midplane row) ──────────
    print("  [3/3] TF + sparse …", flush=True)
    t0 = time.time()

    _, base_tf_coils_sparse = generate_tf_array(
        VV, ntf, VV_R0, TF_a, TF_b, TF_current,
        fixed_geo_tfs=True, numquadpoints=numquadpoints, tf_coil_radius=0,
    )

    # generate_windowpane_array stores coils as [ii=0,jj=0..ntor-1, ii=1, …]
    # ii=0 → θ ≈ 0 → outboard midplane row.  Skip it for the sparse case.
    base_wp_all, _, _, _, _, ntor_act2 = generate_windowpane_array(
        VV, None, wp_fil_spacing, half_per_spacing, wp_n,
        numquadpoints=numquadpoints, order=12, verbose=False,
        nwps_poloidal_target=npoloidal,
        nwps_toroidal_target=ntoroidal,
    )
    base_wp_sparse = base_wp_all[ntor_act2:]   # drop first row (θ ≈ 0)
    print(f"    WP base coils (sparse): {len(base_wp_sparse)} "
          f"(dropped {ntor_act2} outboard-midplane coils)")

    _, bs_sparse = optimize_windowpane_currents(
        base_wp_sparse, base_tf_coils_sparse, surf_plasma,
        definition=definition, precomputed=True,
        maxiter=maxiter,
        current_threshold=current_threshold,
        current_weight=current_weight,
        num_fixed=ntf,
        verbose=True,
    )

    d_sparse = compute_delta_omp(bs_sparse, R0_plasma, a_plasma)
    delta_sparse.append(d_sparse)
    save_tf_wp_vtk(bs_sparse.coils, 2 * nfp * ntf, dirs["TF_sparse"], f"ntf{ntf:02d}")

    plot_wp_currents(base_wp_sparse, VV, dirs["TF_sparse"], f"ntf{ntf:02d}", plot_config)
    plot_relBfinal_norm_modB(
        bs_sparse, surf_plasma,
        dirs["TF_sparse"], f"ntf{ntf:02d}", plot_config,
    )
    print(f"    δ = {d_sparse:.5f}  ({100*d_sparse:.3f} %)   [{time.time()-t0:.1f} s]")

print(f"\nTotal scan time: {(time.time()-t_total)/60:.1f} min")

# ═══════════════════════════════════════════════════════════════════════════
# ── Save tabular results ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

csv_path = os.path.join(base_dir, "delta_results.csv")
with open(csv_path, "w") as f:
    f.write("ntf,ntf_total,delta_TF_only,delta_TF_dense,delta_TF_sparse\n")
    for ntf, d0, d1, d2 in zip(ntf_values, delta_tf, delta_dense, delta_sparse):
        f.write(f"{ntf},{2*ntf},{d0:.6e},{d1:.6e},{d2:.6e}\n")
print(f"\nTabular results saved → {csv_path}")

# ═══════════════════════════════════════════════════════════════════════════
# ── Summary plot: δ vs. N_TF ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

ntf_arr    = np.array(ntf_values)
ntf_total  = 2 * ntf_arr          # total TF coils (displayed on x-axis)

delta_tf_arr     = 100 * np.array(delta_tf)
delta_dense_arr  = 100 * np.array(delta_dense)
delta_sparse_arr = 100 * np.array(delta_sparse)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.semilogy(ntf_total, delta_tf_arr,     "b-o",  lw=3, ms=10, label="TF only")
ax.semilogy(ntf_total, delta_dense_arr,  "r-s",  lw=3, ms=10, label="TF + dense array")
ax.semilogy(ntf_total, delta_sparse_arr, "g-^",  lw=3, ms=10, label="TF + sparse array")
ax.axhline(1.0, color="k", lw=3, ls="--", label=r"$\delta = 1\%$ threshold")

ax.set_xlabel(
    "Number of TF coils",
    fontsize=plot_config.axisfontsize, fontweight="bold",
)
ax.set_ylabel(
    r"Outboard $\mathbf{\delta}$ [%]",
    fontsize=plot_config.axisfontsize, fontweight="bold",
)

ax.set_xticks(ntf_total)
ax.set_xticklabels(
    [f"{n}" for n in ntf_total],
    fontsize=plot_config.ticklabelfontsize,
)
ax.tick_params(axis="y", labelsize=plot_config.ticklabelfontsize)
ax.legend(fontsize=plot_config.legendfontsize, loc="best")
ax.grid(True, which="both", alpha=0.35)

plt.tight_layout()
summary_path = os.path.join(base_dir, "delta_vs_ntf.png")
plt.savefig(summary_path, dpi=plot_config.dpi)
plt.close()
print(f"Summary plot saved → {summary_path}")

# ── Print final table ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'ntf':>5} {'N_TF':>6} {'δ TF-only [%]':>16} "
      f"{'δ dense [%]':>14} {'δ sparse [%]':>14}")
print("-" * 60)
for ntf, d0, d1, d2 in zip(ntf_values, delta_tf_arr, delta_dense_arr, delta_sparse_arr):
    print(f"{ntf:>5} {2*ntf:>6} {d0:>16.3f} {d1:>14.3f} {d2:>14.3f}")
print("=" * 60)
