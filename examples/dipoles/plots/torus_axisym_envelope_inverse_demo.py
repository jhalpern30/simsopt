#!/usr/bin/env python3
"""Inverse envelope demo: fixed axisymmetric envelope with amplitude-driven volume reduction.

This script does the inverse of the direct-envelope workflow:
1) Set a fixed axisymmetric confining envelope in the (R-R0, Z) plane.
2) Increase perturbation amplitude for:
   - Axis torsion (m=1 center displacement).
   - Rotating ellipticity.
3) For each amplitude, compute the largest allowable scale of the perturbed surface that
   still fits inside the fixed envelope.
4) Report and visualize the resulting decrease in torus volume.

Outputs:
- Static PNGs for each amplitude and each mechanism.
- One GIF per mechanism showing amplitude ramp and shrinking required scale.

Edit CONFIG below to tune amplitudes, envelope shape, and output options.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


"""
amplitudes: List of perturbation amplitudes used for static plots.
n_toroidal_plot: Number of toroidal slices to draw as solid curves in each figure.
n_toroidal_envelope: Number of toroidal samples used to build surface envelopes.
n_poloidal: Number of poloidal angle samples per curve.
gif_frames: Number of frames in each amplitude-ramp GIF.
gif_fps: Frames per second for GIF export.
gif_repeat: Number of GIF repeats (0 means loop forever).
major_radius: Major radius R0 used to compute torus volume V=2*pi*R0*A.
fixed_envelope_radius_r: Fixed envelope semiaxis along R-R0.
fixed_envelope_radius_z: Fixed envelope semiaxis along Z.
torsion_wavenumber: Toroidal mode number for m=1 torsion center displacement.
ellipse_rotation_wavenumber: Toroidal mode number for ellipse principal-axis rotation.
output_dir: Directory where PNG and GIF outputs are written.
"""
CONFIG = {
    "amplitudes": [0.1, 0.2, 0.3, 0.4, 0.5],
    "n_toroidal_plot": 5,
    "n_toroidal_envelope": 256,
    "n_poloidal": 256,
    "gif_frames": 20,
    "gif_fps": 4,
    "gif_repeat": 5,
    "major_radius": 3.0,
    "fixed_envelope_radius_r": 1,
    "fixed_envelope_radius_z": 1,
    "torsion_wavenumber": 1,
    "ellipse_rotation_wavenumber": 1,
    "output_dir": Path(__file__).resolve().parent,
}

STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
}

POPPING_COLORS = [
    "#ff0054",
    "#00b4d8",
    "#ffbe0b",
    "#8338ec",
    "#06d6a0",
    "#fb5607",
]


def apply_presentation_style() -> None:
    plt.rcParams.update(STYLE)


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def toroidal_samples(count: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)


def fixed_axisymmetric_envelope(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed axisymmetric envelope as an ellipse and its radial profile r(psi)."""
    a_r = CONFIG["fixed_envelope_radius_r"]
    b_z = CONFIG["fixed_envelope_radius_z"]

    x_env = a_r * np.cos(theta)
    y_env = b_z * np.sin(theta)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    r_env = 1.0 / np.sqrt((cos_t / a_r) ** 2 + (sin_t / b_z) ** 2)
    return x_env, y_env, r_env


def build_torsion_sections(
    amplitude: float,
    phi_samples: np.ndarray,
    theta: np.ndarray,
    scale: float,
    torsion_wavenumber: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build m=1 center-displaced sections with a global scale factor."""
    base_x = np.cos(theta)
    base_y = np.sin(theta)

    x_sections = np.empty((phi_samples.size, theta.size))
    y_sections = np.empty((phi_samples.size, theta.size))

    for i, phi in enumerate(phi_samples):
        phase = torsion_wavenumber * phi
        shift_x = amplitude * np.cos(phase)
        shift_y = amplitude * np.sin(phase)
        x_sections[i, :] = scale * (base_x + shift_x)
        y_sections[i, :] = scale * (base_y + shift_y)

    return x_sections, y_sections


def build_rotating_ellipse_sections(
    amplitude: float,
    phi_samples: np.ndarray,
    theta: np.ndarray,
    scale: float,
    rotation_wavenumber: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rotating ellipse sections with a global scale factor."""
    a = (1.0 + amplitude)
    b = (1.0 - amplitude)

    local_x = a * np.cos(theta)
    local_y = b * np.sin(theta)

    x_sections = np.empty((phi_samples.size, theta.size))
    y_sections = np.empty((phi_samples.size, theta.size))

    for i, phi in enumerate(phi_samples):
        alpha = rotation_wavenumber * phi
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        x_sections[i, :] = scale * (local_x * cos_a - local_y * sin_a)
        y_sections[i, :] = scale * (local_x * sin_a + local_y * cos_a)

    return x_sections, y_sections


def radial_envelope_from_sections(
    x_sections: np.ndarray,
    y_sections: np.ndarray,
    psi_grid: np.ndarray,
) -> np.ndarray:
    """Compute r_env(psi)=max_phi r(phi,psi) from sampled section curves."""
    r_env = np.zeros_like(psi_grid)

    for i in range(x_sections.shape[0]):
        x_curve = x_sections[i, :]
        y_curve = y_sections[i, :]
        psi_curve = np.mod(np.arctan2(y_curve, x_curve), 2.0 * np.pi)
        r_curve = np.sqrt(x_curve**2 + y_curve**2)

        order = np.argsort(psi_curve)
        psi_sorted = psi_curve[order]
        r_sorted = r_curve[order]

        psi_unique, unique_idx = np.unique(psi_sorted, return_index=True)
        r_unique = r_sorted[unique_idx]

        psi_ext = np.concatenate(
            ([psi_unique[-1] - 2.0 * np.pi], psi_unique, [psi_unique[0] + 2.0 * np.pi])
        )
        r_ext = np.concatenate(([r_unique[-1]], r_unique, [r_unique[0]]))

        r_on_grid = np.interp(psi_grid, psi_ext, r_ext)
        r_env = np.maximum(r_env, r_on_grid)

    return r_env


def required_scale_for_fixed_envelope(
    builder,
    amplitude: float,
    theta: np.ndarray,
    phi_dense: np.ndarray,
    mode_wavenumber: int,
    fixed_r_env: np.ndarray,
) -> float:
    """Return largest global scale so the perturbed shape stays inside fixed envelope."""
    x_unit, y_unit = builder(amplitude, phi_dense, theta, 1.0, mode_wavenumber)
    r_unit_env = radial_envelope_from_sections(x_unit, y_unit, np.mod(theta, 2.0 * np.pi))

    safe = np.maximum(r_unit_env, 1e-12)
    return float(np.min(fixed_r_env / safe))


def torus_volume_from_sections(x_sections: np.ndarray, y_sections: np.ndarray, major_radius: float) -> float:
    """Compute equivalent torus volume using mean section area over toroidal slices."""
    areas = np.array([polygon_area(x_sections[i, :], y_sections[i, :]) for i in range(x_sections.shape[0])])
    return float(2.0 * np.pi * major_radius * np.mean(areas))


def render_static_plot(
    x_sections: np.ndarray,
    y_sections: np.ndarray,
    x_fixed_env: np.ndarray,
    y_fixed_env: np.ndarray,
    phi_plot: np.ndarray,
    mechanism_label: str,
    amplitude: float,
    volume_ratio: float,
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = [POPPING_COLORS[i % len(POPPING_COLORS)] for i in range(len(phi_plot))]

    for i in range(len(phi_plot)):
        ax.plot(x_sections[i, :], y_sections[i, :], color=colors[i], linewidth=2.6)

    ax.plot(
        x_fixed_env,
        y_fixed_env,
        color="black",
        linewidth=2.4,
        linestyle="--",
        label="Fixed axisymmetric envelope",
    )

    ax.set_title(
        f"{mechanism_label}: Nonaxisymmetric Volume Ratio = {volume_ratio:.3f}\n"
    )
    ax.set_xlabel(r"$R - R_0$ [arb]")
    ax.set_ylabel(r"$Z$ [arb]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.45, alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    fig.savefig(str(output_path))
    plt.close(fig)


def make_inverse_gif(
    builder,
    mechanism_label: str,
    out_path: Path,
    theta: np.ndarray,
    phi_plot: np.ndarray,
    phi_dense: np.ndarray,
    mode_wavenumber: int,
    max_amplitude: float,
    x_fixed_env: np.ndarray,
    y_fixed_env: np.ndarray,
    fixed_r_env: np.ndarray,
    baseline_volume: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = [POPPING_COLORS[i % len(POPPING_COLORS)] for i in range(len(phi_plot))]

    section_lines = [ax.plot([], [], color=colors[i], linewidth=2.6)[0] for i in range(len(phi_plot))]
    fixed_line = ax.plot(x_fixed_env, y_fixed_env, color="black", linewidth=2.4, linestyle="--")[0]

    ax.set_xlabel(r"$R - R_0$ [arb]")
    ax.set_ylabel(r"$Z$ [arb]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.45, alpha=0.6)
    ax.legend([fixed_line], ["Fixed axisymmetric envelope"], loc="upper right", frameon=True)

    # text_box = ax.text(
    #     0.02,
    #     0.98,
    #     "",
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88},
    # )

    amps = np.linspace(0.0, max_amplitude, CONFIG["gif_frames"])

    def update(frame: int):
        amp = float(amps[frame])
        scale = required_scale_for_fixed_envelope(
            builder, amp, theta, phi_dense, mode_wavenumber, fixed_r_env
        )
        x_plot, y_plot = builder(amp, phi_plot, theta, scale, mode_wavenumber)
        x_dense, y_dense = builder(amp, phi_dense, theta, scale, mode_wavenumber)

        volume = torus_volume_from_sections(x_dense, y_dense, CONFIG["major_radius"])
        vol_ratio = volume / baseline_volume

        for i in range(len(phi_plot)):
            section_lines[i].set_data(x_plot[i, :], y_plot[i, :])

        ax.set_title(
            f"{mechanism_label}: Nonaxisymmetric Volume Ratio = {vol_ratio:.3f}"
        )
        # text_box.set_text(
        #     f"Amplitude = {amp:.3f}\n"
        # )

        return section_lines + [fixed_line] #[fixed_line, text_box]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=CONFIG["gif_frames"],
        interval=1000.0 / CONFIG["gif_fps"],
        blit=True,
    )

    try:
        ani.save(
            str(out_path),
            writer="pillow",
            fps=CONFIG["gif_fps"],
            metadata={"loop": CONFIG["gif_repeat"]},
        )
    except Exception as err:
        plt.close(fig)
        raise RuntimeError("GIF export failed. Install Pillow in your Python environment and rerun.") from err

    plt.close(fig)


def main() -> None:
    apply_presentation_style()
    out_dir = CONFIG["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    amps = CONFIG["amplitudes"]
    theta = np.linspace(0.0, 2.0 * np.pi, CONFIG["n_poloidal"], endpoint=False)
    phi_plot = toroidal_samples(CONFIG["n_toroidal_plot"])
    phi_dense = toroidal_samples(CONFIG["n_toroidal_envelope"])

    x_fixed_env, y_fixed_env, fixed_r_env = fixed_axisymmetric_envelope(theta)

    max_span = 1.2 * max(
        np.max(np.abs(x_fixed_env)),
        np.max(np.abs(y_fixed_env)),
    )
    xlim = (-max_span, max_span)
    ylim = (-max_span, max_span)

    generated_paths: list[Path] = []

    # Baseline volumes at amplitude=0 after fitting inside fixed envelope.
    s0_torsion = required_scale_for_fixed_envelope(
        build_torsion_sections,
        0.0,
        theta,
        phi_dense,
        CONFIG["torsion_wavenumber"],
        fixed_r_env,
    )
    x0_t, y0_t = build_torsion_sections(0.0, phi_dense, theta, s0_torsion, CONFIG["torsion_wavenumber"])
    v0_torsion = torus_volume_from_sections(x0_t, y0_t, CONFIG["major_radius"])

    s0_ellipse = required_scale_for_fixed_envelope(
        build_rotating_ellipse_sections,
        0.0,
        theta,
        phi_dense,
        CONFIG["ellipse_rotation_wavenumber"],
        fixed_r_env,
    )
    x0_e, y0_e = build_rotating_ellipse_sections(
        0.0,
        phi_dense,
        theta,
        s0_ellipse,
        CONFIG["ellipse_rotation_wavenumber"],
    )
    v0_ellipse = torus_volume_from_sections(x0_e, y0_e, CONFIG["major_radius"])

    for amp in amps:
        scale = required_scale_for_fixed_envelope(
            build_torsion_sections,
            amp,
            theta,
            phi_dense,
            CONFIG["torsion_wavenumber"],
            fixed_r_env,
        )
        x_plot, y_plot = build_torsion_sections(
            amp, phi_plot, theta, scale, CONFIG["torsion_wavenumber"]
        )
        x_dense, y_dense = build_torsion_sections(
            amp, phi_dense, theta, scale, CONFIG["torsion_wavenumber"]
        )
        volume = torus_volume_from_sections(x_dense, y_dense, CONFIG["major_radius"])
        ratio = volume / v0_torsion

        out_file = out_dir / f"inverse_torsion_amp_{amp:.2f}".replace(".", "p")
        out_file = out_file.with_suffix(".png")
        render_static_plot(
            x_plot,
            y_plot,
            x_fixed_env,
            y_fixed_env,
            phi_plot,
            "Axis Torsion",
            amp,
            ratio,
            out_file,
            xlim,
            ylim,
        )
        generated_paths.append(out_file)

    for amp in amps:
        scale = required_scale_for_fixed_envelope(
            build_rotating_ellipse_sections,
            amp,
            theta,
            phi_dense,
            CONFIG["ellipse_rotation_wavenumber"],
            fixed_r_env,
        )
        x_plot, y_plot = build_rotating_ellipse_sections(
            amp, phi_plot, theta, scale, CONFIG["ellipse_rotation_wavenumber"]
        )
        x_dense, y_dense = build_rotating_ellipse_sections(
            amp, phi_dense, theta, scale, CONFIG["ellipse_rotation_wavenumber"]
        )
        volume = torus_volume_from_sections(x_dense, y_dense, CONFIG["major_radius"])
        ratio = volume / v0_ellipse

        out_file = out_dir / f"inverse_rotating_ellipticity_amp_{amp:.2f}".replace(".", "p")
        out_file = out_file.with_suffix(".png")
        render_static_plot(
            x_plot,
            y_plot,
            x_fixed_env,
            y_fixed_env,
            phi_plot,
            "Rotating Ellipticity",
            amp,
            ratio,
            out_file,
            xlim,
            ylim,
        )
        generated_paths.append(out_file)

    torsion_gif = out_dir / "inverse_torsion_volume_drop.gif"
    make_inverse_gif(
        build_torsion_sections,
        "Axis Torsion",
        torsion_gif,
        theta,
        phi_plot,
        phi_dense,
        CONFIG["torsion_wavenumber"],
        max(amps),
        x_fixed_env,
        y_fixed_env,
        fixed_r_env,
        v0_torsion,
        xlim,
        ylim,
    )
    generated_paths.append(torsion_gif)

    ellipse_gif = out_dir / "inverse_rotating_ellipticity_volume_drop.gif"
    make_inverse_gif(
        build_rotating_ellipse_sections,
        "Rotating Ellipticity",
        ellipse_gif,
        theta,
        phi_plot,
        phi_dense,
        CONFIG["ellipse_rotation_wavenumber"],
        max(amps),
        x_fixed_env,
        y_fixed_env,
        fixed_r_env,
        v0_ellipse,
        xlim,
        ylim,
    )
    generated_paths.append(ellipse_gif)

    print("Generated files:")
    for pth in generated_paths:
        print(f" - {pth}")


if __name__ == "__main__":
    main()
