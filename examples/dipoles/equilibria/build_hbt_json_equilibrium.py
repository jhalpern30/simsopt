#!/usr/bin/env python3
"""
Build HBT VMEC inputs from JSON pressure and q profiles via a two-step workflow.

Inputs:
  - ``input.hbt_template`` – VMEC template (boundary, solver settings)
  - ``HBT_p(Phi)_profile.json``, ``HBT_q(Phi)_profile.json`` – p(Φ) and q(Φ)

Steps:
  1. Interpolate JSON profiles onto a uniform s grid; set ι(s) = 1/q(s)
  2. Write ``input.hbt_json_pq`` (NCURR=0: pressure + ι) and run fixed-boundary VMEC
  3. Extract enclosed toroidal current I_enc(s) from wout ``buco``
  4. Write ``input.hbt_json_pI`` (NCURR=1: pressure + I_enc, ``curtor`` at s=1)
  5. Plot input knots vs VMEC pressure, current, and safety factor

Outputs (in this directory):
  - ``input.hbt_json_pq``, ``input.hbt_json_pI``
  - VMEC ``wout*.nc`` from the pressure/q run
  - ``hbt_json_profiles.png``, ``hbt_json_profiles_q.png``
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import netcdf_file

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent))
from helper_functions import PlotConfig  # noqa: E402

from simsopt.mhd.vmec import Vmec, array_to_namelist

TEMPLATE = HERE / "input.hbt_template"
P_JSON = HERE / "HBT_p(Phi)_profile.json"
Q_JSON = HERE / "HBT_q(Phi)_profile.json"
INPUT_PQ = HERE / "input.hbt_json_pq"
INPUT_PI = HERE / "input.hbt_json_pI"
PLOT_OUT = HERE / "hbt_json_profiles.png"

MU0 = 4.0 * np.pi * 1e-7
N_KNOTS = 99

plot_config = PlotConfig(
    dpi=150,
    titlefontsize=14,
    axisfontsize=13,
    legendfontsize=11,
    ticklabelfontsize=12,
)


def _load_json_profile(path: Path, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        data = json.load(f)
    phi = np.asarray(data["Phi"], dtype=float)
    y = np.asarray(data[y_key], dtype=float)
    if phi.size != y.size:
        raise ValueError(f"{path}: Phi and {y_key} length mismatch.")
    return phi, y


def _template_header_footer() -> tuple[str, str, float]:
    """Return (prefix, suffix, phiedge) from input.hbt_template, excluding profile block."""
    raw = TEMPLATE.read_text()
    start = raw.index("!----- Current/Iota Parameters -----")
    end = raw.index("!----- Axis Parameters -----")
    prefix = raw[:start]
    suffix = raw[end:]
    m = re.search(r"PHIEDGE\s*=\s*([0-9.eE+\-]+)", raw)
    if not m:
        raise ValueError("PHIEDGE not found in template.")
    return prefix, suffix, float(m.group(1))


def _write_profile_block_pq(s: np.ndarray, p_pa: np.ndarray, iota: np.ndarray) -> str:
    nml = "!----- Current/Iota Parameters -----\n"
    nml += "NCURR = 0\n"
    nml += 'PIOTA_TYPE = "cubic_spline"\n'
    nml += "AI_AUX_S = " + array_to_namelist(s, True)
    nml += "AI_AUX_F = " + array_to_namelist(iota)
    nml += "PMASS_TYPE = 'CUBIC_SPLINE'\n"
    nml += "AM_AUX_S = " + array_to_namelist(s, True)
    nml += "AM_AUX_F = " + array_to_namelist(p_pa)
    return nml


def _write_profile_block_pi(s: np.ndarray, p_pa: np.ndarray, I_enc_A: np.ndarray) -> str:
    curtor = float(I_enc_A[-1])
    nml = "!----- Current/Iota Parameters -----\n"
    nml += "NCURR = 1\n"
    nml += f"curtor = {curtor}\n"
    nml += "PCURR_TYPE = 'cubic_spline_I'\n"
    nml += "AC_AUX_S = " + array_to_namelist(s, True)
    nml += "AC_AUX_F = " + array_to_namelist(I_enc_A)
    nml += "PMASS_TYPE = 'CUBIC_SPLINE'\n"
    nml += "AM_AUX_S = " + array_to_namelist(s, True)
    nml += "AM_AUX_F = " + array_to_namelist(p_pa)
    return nml


def _profiles_on_uniform_grid(
    phi: np.ndarray, p: np.ndarray, q: np.ndarray, phiedge: float, n_knots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_raw = phi / phiedge
    if np.any(s_raw < -1e-12) or s_raw[-1] > 1.0 + 1e-10:
        raise ValueError(
            f"Normalized flux s must lie in [0, 1]; got [{s_raw.min():.6g}, {s_raw.max():.6g}]."
        )
    s_knots = np.linspace(0.0, 1.0, n_knots)
    p_interp = interp1d(s_raw, p, kind="cubic", fill_value="extrapolate")
    q_interp = interp1d(s_raw, q, kind="cubic", fill_value="extrapolate")
    iota_interp = interp1d(s_raw, 1.0 / q, kind="cubic", fill_value="extrapolate")
    return s_knots, p_interp(s_knots), iota_interp(s_knots)


def _enclosed_current_from_wout(wout_path: Path) -> dict[str, np.ndarray]:
    """Enclosed toroidal current from VMEC wout (consistent with plot_profiles.py)."""
    f = netcdf_file(str(wout_path), mmap=False)
    ns = int(f.variables["ns"][()])
    buco = np.asarray(f.variables["buco"][()], dtype=float)
    presf = np.asarray(f.variables["presf"][()], dtype=float)
    ctor = float(f.variables["ctor"][()])
    f.close()

    s_full = np.linspace(0.0, 1.0, ns)
    I_enc_A = -(2.0 * np.pi / MU0) * buco
    return {"s_full": s_full, "I_enc_A": I_enc_A, "presf": presf, "ctor": ctor}


def _plot_profiles(
    s_in: np.ndarray,
    p_in_pa: np.ndarray,
    I_enc_knots_kA: np.ndarray,
    s_vmec: np.ndarray,
    p_vmec_pa: np.ndarray,
    I_enc_vmec_kA: np.ndarray,
    s_q: np.ndarray | None,
    q_target: np.ndarray | None,
    iota_vmec: np.ndarray | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(s_in, p_in_pa / 1e3, "o", ms=3, label="input knots")
    ax.plot(s_vmec, p_vmec_pa / 1e3, "-", lw=2, alpha=0.8, label="VMEC presf")
    ax.set_ylabel(r"$p$ [kPa]")
    ax.set_title("Pressure vs normalized toroidal flux")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=plot_config.legendfontsize)

    ax = axes[1]
    ax.plot(s_in, I_enc_knots_kA, "o", ms=3, label="input knots")
    ax.plot(s_vmec, I_enc_vmec_kA, "-", lw=2, color="tab:red", alpha=0.8,
            label=r"$I_{\mathrm{enc}}$ from VMEC")
    ax.set_xlabel(r"Normalized toroidal flux $s$")
    ax.set_ylabel(r"$I_{\mathrm{enc}}$ [kA]")
    ax.set_title("Enclosed toroidal current")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=plot_config.legendfontsize)
    ax.set_xlim(0.0, 1.0)

    fig.savefig(out_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)

    if s_q is not None and q_target is not None and iota_vmec is not None:
        fig_q, ax_q = plt.subplots(figsize=(7.5, 3.5), constrained_layout=True)
        ax_q.plot(s_q, q_target, "o", ms=4, label=r"$q$ input (JSON)")
        ax_q.plot(s_vmec, 1.0 / iota_vmec, "-", lw=2, label=r"$1/\iota$ from VMEC")
        ax_q.set_xlabel(r"Normalized toroidal flux $s$")
        ax_q.set_ylabel(r"$q$")
        ax_q.set_title("Safety factor check")
        ax_q.grid(True, linestyle="--", alpha=0.45)
        ax_q.legend(fontsize=plot_config.legendfontsize)
        ax_q.set_xlim(0.0, 1.0)
        q_path = out_path.with_name(out_path.stem + "_q.png")
        fig_q.savefig(q_path, dpi=plot_config.dpi, bbox_inches="tight")
        plt.close(fig_q)
        print(f"Saved plot -> {q_path}")


def main() -> None:
    for path in (TEMPLATE, P_JSON, Q_JSON):
        if not path.exists():
            raise FileNotFoundError(path)

    phi_p, p_pa = _load_json_profile(P_JSON, "p_q")
    phi_q, q = _load_json_profile(Q_JSON, "q")
    if not np.allclose(phi_p, phi_q):
        raise ValueError("Phi grids in pressure and q JSON files do not match.")

    phiedge = float(phi_p[-1])
    prefix, suffix, _ = _template_header_footer()
    prefix = re.sub(
        r"PHIEDGE\s*=\s*[0-9.eE+\-]+",
        f"PHIEDGE = {phiedge}",
        prefix,
        count=1,
    )

    s_knots, p_knots, iota_knots = _profiles_on_uniform_grid(
        phi_p, p_pa, q, phiedge, N_KNOTS
    )

    profile_pq = _write_profile_block_pq(s_knots, p_knots, iota_knots)
    INPUT_PQ.write_text(prefix + profile_pq + suffix)
    print(f"Wrote {INPUT_PQ}  (NCURR=0, p(s) + ι(s)=1/q(s), PHIEDGE={phiedge})")

    os.chdir(HERE)
    print("Running fixed-boundary VMEC …")
    vmec = Vmec(str(INPUT_PQ.name))
    vmec.run()
    wout_path = Path(vmec.output_file)
    print(f"VMEC finished; wout -> {wout_path}")

    prof = _enclosed_current_from_wout(wout_path)
    I_on_knots = interp1d(
        prof["s_full"], prof["I_enc_A"], kind="cubic", fill_value="extrapolate"
    )(s_knots)
    I_on_knots[0] = 0.0

    profile_pi = _write_profile_block_pi(s_knots, p_knots, I_on_knots)
    INPUT_PI.write_text(prefix + profile_pi + suffix)
    print(f"Wrote {INPUT_PI}  (NCURR=1, curtor={I_on_knots[-1]:.6g} A)")

    iota_vmec = np.asarray(vmec.wout.iotaf, dtype=float) * float(vmec.wout.signgs)
    phi_edge = float(vmec.wout.phi[-1])
    iota_vmec *= np.sign(phi_edge)

    s_q = phi_q / phiedge
    _plot_profiles(
        s_knots,
        p_knots,
        I_on_knots / 1e3,
        prof["s_full"],
        prof["presf"],
        prof["I_enc_A"] / 1e3,
        s_q,
        q,
        iota_vmec,
        PLOT_OUT,
    )
    print(f"Saved plot -> {PLOT_OUT}")
    print(f"VMEC ctor = {prof['ctor']:.4g} A; spline I(s=1) = {I_on_knots[-1]:.4g} A")


if __name__ == "__main__":
    main()
