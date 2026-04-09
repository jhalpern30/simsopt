#!/usr/bin/env python3
"""Generate presentation-quality plots and GIFs for axisymmetric confinement envelopes.

This standalone script demonstrates how the minimum confining axisymmetric cross-section
(and corresponding torus volume) grows when increasing:
1) Axis torsion amplitude.
2) Rotating-ellipse ellipticity amplitude.

Outputs are written to examples/dipoles/plots and include:
- 3 PNGs for torsion amplitudes.
- 3 PNGs for rotating-ellipse amplitudes.
- 1 GIF ramping torsion amplitude.
- 1 GIF ramping ellipticity amplitude.

Edit CONFIG near the top of this file to change amplitudes, resolution, and styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


"""
amplitudes: List of perturbation amplitudes used for static plots.
n_toroidal_plot: Number of toroidal slices to draw as solid curves in each figure.
n_toroidal_envelope: Number of toroidal samples used to build the confining envelope
    r_env(theta)=max_phi r(phi, theta). Larger values improve envelope accuracy.
n_poloidal: Number of poloidal angle samples per cross-section curve.
gif_frames: Number of frames in each amplitude-ramp GIF.
gif_fps: Frames per second for GIF export.
gif_repeat: Number of GIF repeats (0 means loop forever).
major_radius: Major radius R0 used to report equivalent axisymmetric volume V=2*pi*R0*A_env.
minor_radius: Baseline minor radius for the unperturbed circular cross-section.
torsion_wavenumber: Toroidal mode number for the axis-torsion m=1 center shift,
    where horizontal and vertical components rotate with phi.
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
    "gif_repeat": 3,
    "major_radius": 3.0,
    "minor_radius": 1.0,
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
    """Apply a consistent visual style intended for presentation plots."""
    plt.rcParams.update(STYLE)


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area enclosed by a planar closed curve represented by samples."""
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def toroidal_samples(count: int) -> np.ndarray:
    """Return evenly spaced toroidal angles in [0, 2pi)."""
    return np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)


def build_torsion_sections(
    amplitude: float,
    phi_samples: np.ndarray,
    theta: np.ndarray,
    minor_radius: float,
    torsion_wavenumber: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cross-sections for an axis-torsion perturbation model.

    Returns arrays x_sections and y_sections with shape (n_phi, n_theta)
    in a fixed local (R-R0, Z) plotting frame.

    The perturbation is a rigid m=1 center displacement around the magnetic axis:
    the section center rotates in the local plane with toroidal angle phi.
    """
    base_x = minor_radius * np.cos(theta)
    base_y = minor_radius * np.sin(theta)

    x_sections = np.empty((phi_samples.size, theta.size))
    y_sections = np.empty((phi_samples.size, theta.size))

    for i, phi in enumerate(phi_samples):
        phase = torsion_wavenumber * phi
        shift_x = amplitude * minor_radius * np.cos(phase)
        shift_y = amplitude * minor_radius * np.sin(phase)
        x_sections[i, :] = base_x + shift_x
        y_sections[i, :] = base_y + shift_y

    return x_sections, y_sections


def build_rotating_ellipse_sections(
    amplitude: float,
    phi_samples: np.ndarray,
    theta: np.ndarray,
    minor_radius: float,
    rotation_wavenumber: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cross-sections for a rotating ellipse perturbation model.

    Ellipticity amplitude controls semiaxes: a = r(1+amp), b = r(1-amp).
    The principal axes rotate with toroidal angle.
    """
    a = minor_radius * (1.0 + amplitude)
    b = minor_radius * (1.0 - amplitude)

    local_x = a * np.cos(theta)
    local_y = b * np.sin(theta)

    x_sections = np.empty((phi_samples.size, theta.size))
    y_sections = np.empty((phi_samples.size, theta.size))

    for i, phi in enumerate(phi_samples):
        alpha = rotation_wavenumber * phi
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        x_sections[i, :] = local_x * cos_a - local_y * sin_a
        y_sections[i, :] = local_x * sin_a + local_y * cos_a

    return x_sections, y_sections


def confining_axisymmetric_envelope(
    x_sections: np.ndarray,
    y_sections: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute minimum confining axisymmetric profile via pointwise envelope.

    Construction used: r_env(theta) = max_phi r(phi, theta).
    """
    psi_grid = np.mod(theta, 2.0 * np.pi)
    r_env = np.zeros_like(psi_grid)

    for i in range(x_sections.shape[0]):
        x_curve = x_sections[i, :]
        y_curve = y_sections[i, :]
        psi_curve = np.mod(np.arctan2(y_curve, x_curve), 2.0 * np.pi)
        r_curve = np.sqrt(x_curve**2 + y_curve**2)

        order = np.argsort(psi_curve)
        psi_sorted = psi_curve[order]
        r_sorted = r_curve[order]

        # Drop duplicate angles to keep interpolation well-defined.
        psi_unique, unique_idx = np.unique(psi_sorted, return_index=True)
        r_unique = r_sorted[unique_idx]

        # Enforce periodicity by extending one point on each side.
        psi_ext = np.concatenate(([psi_unique[-1] - 2.0 * np.pi], psi_unique, [psi_unique[0] + 2.0 * np.pi]))
        r_ext = np.concatenate(([r_unique[-1]], r_unique, [r_unique[0]]))

        r_on_grid = np.interp(psi_grid, psi_ext, r_ext)
        r_env = np.maximum(r_env, r_on_grid)

    x_env = r_env * np.cos(psi_grid)
    y_env = r_env * np.sin(psi_grid)
    return x_env, y_env


def filename_from_amplitude(prefix: str, amplitude: float) -> str:
    """Build deterministic file names with decimal-safe amplitude tokens."""
    amp_token = f"{amplitude:.2f}".replace(".", "p")
    return f"{prefix}_amp_{amp_token}.png"


def render_static_plot(
    x_sections: np.ndarray,
    y_sections: np.ndarray,
    x_env: np.ndarray,
    y_env: np.ndarray,
    phi_plot: np.ndarray,
    mechanism_label: str,
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Render and save one static figure for a mechanism/amplitude pair."""
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = [POPPING_COLORS[i % len(POPPING_COLORS)] for i in range(len(phi_plot))]

    for i, phi in enumerate(phi_plot):
        ax.plot(
            x_sections[i, :],
            y_sections[i, :],
            color=colors[i],
            linewidth=2.6,
        )

    ax.plot(
        x_env,
        y_env,
        color="black",
        linewidth=2.4,
        linestyle="--",
        label="Minimum confining axisymmetric envelope",
    )

    area_env = polygon_area(x_env, y_env)

    ax.set_title(
        f"{mechanism_label}: "
        f"Envelope area = {area_env:.3f}"
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


def compute_axis_limits(
    builder,
    max_amplitude: float,
    theta: np.ndarray,
    minor_radius: float,
    phi_dense: np.ndarray,
    mode_wavenumber: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute fixed plotting limits from the max-amplitude geometry."""
    x_sec, y_sec = builder(
        max_amplitude,
        phi_dense,
        theta,
        minor_radius,
        mode_wavenumber,
    )
    x_env, y_env = confining_axisymmetric_envelope(x_sec, y_sec, theta)

    all_x = np.concatenate([x_sec.reshape(-1), x_env])
    all_y = np.concatenate([y_sec.reshape(-1), y_env])
    span = 1.15 * max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
    return (-span, span), (-span, span)


def make_gif(
    builder,
    mechanism_label: str,
    out_path: Path,
    theta: np.ndarray,
    phi_plot: np.ndarray,
    phi_dense: np.ndarray,
    minor_radius: float,
    mode_wavenumber: int,
    max_amplitude: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Create a smooth amplitude-ramp GIF for one perturbation mechanism."""
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = [POPPING_COLORS[i % len(POPPING_COLORS)] for i in range(len(phi_plot))]

    section_lines = [
        ax.plot([], [], color=colors[i], linewidth=2.6)[0]
        for i in range(len(phi_plot))
    ]
    envelope_line = ax.plot([], [], color="black", linewidth=2.4, linestyle="--")[0]

    ax.set_xlabel(r"$R - R_0$ [arb]")
    ax.set_ylabel(r"$Z$ [arb]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.45, alpha=0.6)

    ax.legend([envelope_line], ["Minimum confining axisymmetric envelope"], loc="upper right", frameon=True)

    amplitude_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88},
    )

    amplitudes = np.linspace(0.0, max_amplitude, CONFIG["gif_frames"])

    def update(frame: int):
        amp = amplitudes[frame]
        x_plot, y_plot = builder(amp, phi_plot, theta, minor_radius, mode_wavenumber)
        x_dense, y_dense = builder(amp, phi_dense, theta, minor_radius, mode_wavenumber)
        x_env, y_env = confining_axisymmetric_envelope(x_dense, y_dense, theta)

        for i in range(len(phi_plot)):
            section_lines[i].set_data(x_plot[i, :], y_plot[i, :])
        envelope_line.set_data(x_env, y_env)

        area_env = polygon_area(x_env, y_env)
        ax.set_title(
            f"{mechanism_label}: "
            f"Envelope area = {area_env:.3f}"
        )
        amplitude_text.set_text(f"Amplitude = {amp:.3f}")

        return section_lines + [envelope_line, amplitude_text]

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
        raise RuntimeError(
            "GIF export failed. Install Pillow in your Python environment and rerun."
        ) from err

    plt.close(fig)


def main() -> None:
    apply_presentation_style()
    out_dir = CONFIG["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    amplitudes = CONFIG["amplitudes"]
    theta = np.linspace(0.0, 2.0 * np.pi, CONFIG["n_poloidal"], endpoint=False)
    phi_plot = toroidal_samples(CONFIG["n_toroidal_plot"])
    phi_dense = toroidal_samples(CONFIG["n_toroidal_envelope"])
    max_amp = max(amplitudes)

    torsion_limits = compute_axis_limits(
        build_torsion_sections,
        max_amp,
        theta,
        CONFIG["minor_radius"],
        phi_dense,
        CONFIG["torsion_wavenumber"],
    )
    ellipse_limits = compute_axis_limits(
        build_rotating_ellipse_sections,
        max_amp,
        theta,
        CONFIG["minor_radius"],
        phi_dense,
        CONFIG["ellipse_rotation_wavenumber"],
    )

    generated_paths: list[Path] = []

    for amp in amplitudes:
        x_plot, y_plot = build_torsion_sections(
            amp,
            phi_plot,
            theta,
            CONFIG["minor_radius"],
            CONFIG["torsion_wavenumber"],
        )
        x_dense, y_dense = build_torsion_sections(
            amp,
            phi_dense,
            theta,
            CONFIG["minor_radius"],
            CONFIG["torsion_wavenumber"],
        )
        x_env, y_env = confining_axisymmetric_envelope(x_dense, y_dense, theta)
        out_file = out_dir / filename_from_amplitude("torsion", amp)
        render_static_plot(
            x_plot,
            y_plot,
            x_env,
            y_env,
            phi_plot,
            "Axis torsion",
            out_file,
            torsion_limits[0],
            torsion_limits[1],
        )
        generated_paths.append(out_file)

    for amp in amplitudes:
        x_plot, y_plot = build_rotating_ellipse_sections(
            amp,
            phi_plot,
            theta,
            CONFIG["minor_radius"],
            CONFIG["ellipse_rotation_wavenumber"],
        )
        x_dense, y_dense = build_rotating_ellipse_sections(
            amp,
            phi_dense,
            theta,
            CONFIG["minor_radius"],
            CONFIG["ellipse_rotation_wavenumber"],
        )
        x_env, y_env = confining_axisymmetric_envelope(x_dense, y_dense, theta)
        out_file = out_dir / filename_from_amplitude("rotating_ellipse", amp)
        render_static_plot(
            x_plot,
            y_plot,
            x_env,
            y_env,
            phi_plot,
            "Rotating Ellipticity",
            out_file,
            ellipse_limits[0],
            ellipse_limits[1],
        )
        generated_paths.append(out_file)

    torsion_gif = out_dir / "torsion_amplitude_ramp.gif"
    make_gif(
        build_torsion_sections,
        "Axis torsion",
        torsion_gif,
        theta,
        phi_plot,
        phi_dense,
        CONFIG["minor_radius"],
        CONFIG["torsion_wavenumber"],
        max_amp,
        torsion_limits[0],
        torsion_limits[1],
    )
    generated_paths.append(torsion_gif)

    ellipse_gif = out_dir / "rotating_ellipse_amplitude_ramp.gif"
    make_gif(
        build_rotating_ellipse_sections,
        "Rotating Ellipticity",
        ellipse_gif,
        theta,
        phi_plot,
        phi_dense,
        CONFIG["minor_radius"],
        CONFIG["ellipse_rotation_wavenumber"],
        max_amp,
        ellipse_limits[0],
        ellipse_limits[1],
    )
    generated_paths.append(ellipse_gif)

    print("Generated files:")
    for pth in generated_paths:
        print(f" - {pth}")


if __name__ == "__main__":
    main()
