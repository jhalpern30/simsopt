"""
File: stage_2_scan.py
Author: Jake Halpern
Last Edit Date: 03/10/2026
Description: This script sends off an optimization scan over various geometric parameters for
             windowpane coil currents on an axisymmetric surface for a specified plasma equilibrium
             It is a modified version of run_optimize_scan.py that uses Latin Hypercube Sampling (LHS)
             to scan over 4D space: [dipole_radius, VV_a, VV_b, VV_R0]
"""

import os
import shutil
import numpy as np
from datetime import datetime
from optimize import *
from simsopt.geo import SurfaceRZFourier
from scipy.stats import qmc

# Set script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define output parent folder
output_subfolder = "stage_2_LHS_3D_scan_no_sparsity"  # Change this as needed
parent_run_dir = os.path.join(script_dir, f"../outputs/{output_subfolder}")
os.makedirs(parent_run_dir, exist_ok=True)

### Base Simulation Parameters ###
# Dipole parameters
fil_distance = 0.05  # distance between dipole filaments for finite coil winding pack [m]
half_per_distance = 0.05  # distance between dipole panels between half field periods of the device [m]

# Plasma Surface
surf_s = 1  # value of s to cut the surface at (1 if already HBT sized)
surf_dof_scale = 1  # used to scale the dofs of the surface (1 if already HBT sized)
eq_name = "wout_nfp22ginsburg_000_000281"  # name of the wout file from vmec
eq_dir = os.path.join(script_dir, "equilibria")  # equilibria should be in this folder

# Extract the minimum axisymmetric VV size
eq_name_full = os.path.join(eq_dir, eq_name + ".nc")
surf = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range="full torus", nphi=128, ntheta=64)
surf.set_dofs(surf_dof_scale * surf.get_dofs())
R = np.sqrt(surf.gamma()[:, :, 0]**2 + surf.gamma()[:, :, 1]**2)
Rmin = np.min(R); Rmax = np.max(R)
Zmin = np.min(surf.gamma()[:, :, 2]); Zmax = np.max(surf.gamma()[:, :, 2])
VV_Ravg = (Rmin + Rmax) / 2
VV_a_min = (Rmax - Rmin) / 2
VV_b_min = (Zmax - Zmin) / 2
print(f"VV_Ravg = {VV_Ravg:.2f}, VV_a_min = {VV_a_min:.2f}, VV_b_min = {VV_b_min:.2f}")

# TF coils parameters
ntf = 4
num_fixed = ntf # all currents are fixed in TF coils
field_on_axis = 0.5  # on-axis magnetic field (Tesla)
fixed_geo_TFs = True

# Directory naming determinations
current_date = datetime.now().strftime("%Y%m%d")
existing_runs = [d for d in os.listdir(parent_run_dir) if d.split("_")[0].isdigit()]
if existing_runs:
    next_run_number = max(int(d.split("_")[0]) for d in existing_runs) + 1
else:
    next_run_number = 1

# Define the scan parameters
# Number of LHS samples
n_samples = 100  # choose based on how many ~20s runs you can afford

# Bounds for 3D LHS (min, max) for each parameter
VV_a_dist_min, VV_a_dist_max = 0.08, 0.16
VV_b_dist_min, VV_b_dist_max = 0.08, 0.16
VV_R0_dist_min, VV_R0_dist_max = -0.03, 0.03

# 3D Latin Hypercube Sampling: [VV_a_dist, VV_b_dist, VV_R0_dist]
n_dim = 3
sampler = qmc.LatinHypercube(d=n_dim)
unit_samples = sampler.random(n=n_samples)  # shape (n_samples, 3)
# Map from [0, 1]^4 to physical parameter ranges
VV_a_dists   = qmc.scale(unit_samples[:, 0][:, None], VV_a_dist_min, VV_a_dist_max)
VV_b_dists   = qmc.scale(unit_samples[:, 1][:, None], VV_b_dist_min, VV_b_dist_max)
VV_R0_dists  = qmc.scale(unit_samples[:, 2][:, None], VV_R0_dist_min, VV_R0_dist_max)

# Loop over 3D LHS samples
for i in range(n_samples):
    # Unpack the 3D LHS sample and apply to the scan parameters
    VV_a = VV_a_min + VV_a_dists[i][0]
    VV_b = VV_b_min + VV_b_dists[i][0]
    VV_R0 = VV_Ravg + VV_R0_dists[i][0]

    # Choose dipole size that keeps ntoroidal = 8 - npoloidal will be determined later on to keep approximately square coils on inboard side
    nt = 8 # only really achieve good field error when ntoroidal is a multiple of ntf
    Rtor_inboard = (np.pi/surf.nfp*(VV_R0 - VV_a) - half_per_distance - (nt-1) * fil_distance) / (2 * nt)

    # For this scan, we keep the distance between TF coils and VV constant
    # Pros: physical because this distance is likely to be constant
    # Cons: this changes things like ripple and BdotN, which we don't want to change
    TF_R0 = VV_R0 # Same as HBT
    TF_a = VV_a + 0.15  # HBT VV_a = 0.25, TF_a = 0.4 - keep 0.15m spacing constant
    TF_b = VV_b + 0.15 # same spacing in both directions
    print(f"\nWP rad = {Rtor_inboard}, VV_a = {VV_a:.3f}, VV_b = {VV_b:.3f}, VV_R0 = {VV_R0:.3f}, TF_R0 = {TF_R0:.2f}, TF_a = {TF_a:.2f}, TF_b = {TF_b:.2f}")
    # Create a unique directory name
    run_dir_name = f"{next_run_number:02}_{current_date}_diprad_{Rtor_inboard:.3f}_VV_a_{VV_a:.3f}_VV_b_{VV_b:.3f}_VV_R0_{VV_R0:.3f}"
    run_dir = os.path.join(parent_run_dir, run_dir_name)
    # Create directory and copy files into it
    os.makedirs(run_dir, exist_ok=True)
    # Run the optimization
    optimize(
        fil_distance=fil_distance,
        half_per_distance=half_per_distance,
        dipole_radius=Rtor_inboard, # dipole parameters
        VV_a=VV_a,
        VV_b=VV_b,
        VV_R0=VV_R0,  # vessel parameters
        surf_s=surf_s,
        surf_dof_scale=surf_dof_scale,
        eq_dir=eq_dir,
        eq_name=eq_name,  # equilibrium parameters
        ntf=ntf,
        num_fixed=num_fixed,
        field_on_axis=field_on_axis,
        TF_R0=TF_R0,
        TF_a=TF_a,
        TF_b=TF_b,
        fixed_geo_TFs=fixed_geo_TFs,
        CURRENT_THRESHOLD=1e6, # don't worry about in this scan
        CURRENT_WEIGHT=1e-12,
        output_dir=run_dir,
        verbose=False,
    )

    next_run_number += 1
