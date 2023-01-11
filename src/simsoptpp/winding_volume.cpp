#include "winding_volume.h"

// compute which cells are next to which cells 
Array connections(Array& coil_points, int Nadjacent, int dx, int dy, int dz)
{
    int Ndipole = coil_points.shape(0);
    
    // Last index indicates +- the x/y/z directions
    Array connectivity_inds = xt::zeros<int>({Ndipole, 6, 6});
    
    // Compute distances between dipole j and all other dipoles
    // By default computes distance between dipole j and itself
#pragma omp parallel for schedule(static)
    for (int j = 0; j < Ndipole; ++j) {
	vector<double> dist_ij(Ndipole, 1e10);
        for (int i = 0; i < Ndipole; ++i) {
	    if (i != j) {
	        dist_ij[i] = sqrt((coil_points(i, 0) - coil_points(j, 0)) * (coil_points(i, 0) - coil_points(j, 0)) + (coil_points(i, 1) - coil_points(j, 1)) * (coil_points(i, 1) - coil_points(j, 1)) + (coil_points(i, 2) - coil_points(j, 2)) * (coil_points(i, 2) - coil_points(j, 2)));
	    }
        }
        for (int k = 0; k < 6; ++k) {
	    auto result = std::min_element(dist_ij.begin(), dist_ij.end());
            int dist_ind = std::distance(dist_ij.begin(), result);
	    // Get dx, dy, dz from this ind
	    double dxx = coil_points(j, 0) - coil_points(dist_ind, 0);
	    double dyy = coil_points(j, 1) - coil_points(dist_ind, 1);
	    double dzz = coil_points(j, 2) - coil_points(dist_ind, 2);
            // check if this cell is not directly adjacent
	    if (dxx > dx) connectivity_inds(j, k, 0) = -1; // -1 to indicate no adjacent cell
	    else if (dxx < -dx) connectivity_inds(j, k, 1) = -1;
	    else if (dyy > dy) connectivity_inds(j, k, 2) = -1;
	    else if (dyy < -dy) connectivity_inds(j, k, 3) = -1;
	    else if (dzz > dz) connectivity_inds(j, k, 4) = -1;
	    else if (dzz < -dz) connectivity_inds(j, k, 5) = -1;
	    // okay so the cell is adjacent... which direction is it?
	    else if ((abs(dxx) > abs(dyy)) && (abs(dxx) > abs(dzz)) && (dxx > 0.0)) connectivity_inds(j, k, 0) = dist_ind;
	    else if ((abs(dxx) > abs(dyy)) && (abs(dxx) > abs(dzz)) && (dxx < 0.0)) connectivity_inds(j, k, 1) = dist_ind;
	    else if ((abs(dyy) > abs(dxx)) && (abs(dyy) > abs(dzz)) && (dyy > 0.0)) connectivity_inds(j, k, 2) = dist_ind;
	    else if ((abs(dyy) > abs(dxx)) && (abs(dyy) > abs(dzz)) && (dyy < 0.0)) connectivity_inds(j, k, 3) = dist_ind;
	    else if ((abs(dzz) > abs(dxx)) && (abs(dzz) > abs(dyy)) && (dzz > 0.0)) connectivity_inds(j, k, 4) = dist_ind;
	    else if ((abs(dzz) > abs(dxx)) && (abs(dzz) > abs(dyy)) && (dzz < 0.0)) connectivity_inds(j, k, 5) = dist_ind;
            dist_ij[dist_ind] = 1e10; // eliminate the min to get the next min
	}
    }
    return connectivity_inds;
}


// Calculate the geometrics factor from the polynomial basis functions 
Array winding_volume_geo_factors(Array& points, Array& coil_points, Array& integration_points, Array& plasma_normal, Array& Phi) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(coil_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_coil_points = coil_points.shape(0);

    // integration points should be shape (num_coil_points, num_integration_points, 3)
    int num_integration_points = integration_points.shape(1);
    int num_basis_functions = Phi.shape(0);
    Array geo_factor = xt::zeros<double>({num_points, num_coil_points, num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_points; i++) {
	double nx = plasma_normal(i, 0);
	double ny = plasma_normal(i, 1);
	double nz = plasma_normal(i, 2);
	double rx = points(i, 0);
	double ry = points(i, 1);
	double rz = points(i, 2);
        for (int jj = 0; jj < num_coil_points; jj++) {
            for (int j = 0; j < num_integration_points; j++) {
	        double rprimex = integration_points(j, 0);
	        double rprimey = integration_points(j, 1);
	        double rprimez = integration_points(j, 2);
	        double rvecx = rx - rprimex;
	        double rvecy = ry - rprimey;
	        double rvecz = rz - rprimez;
	        double rvec_mag = sqrt(rvecx * rvecx + rvecy * rvecy + rvecz * rvecz);
	        double rvec_inv3 = 1.0 / (rvec_mag * (rvec_mag * rvec_mag));
	        double n_cross_rvec_x = ny * rvecz - nz * rvecy;
	        double n_cross_rvec_y = nz * rvecx - nx * rvecz;
	        double n_cross_rvec_z = nx * rvecy - ny * rvecx;
                for (int k = 0; k < num_basis_functions; k++) {
                    geo_factor(i, jj, k) += (n_cross_rvec_x * Phi(k, i, j, 0) + n_cross_rvec_y * Phi(k, i, j, 1) + n_cross_rvec_z * Phi(k, i, j, 2)) * rvec_inv3;
                }
	    }
	}
    }
    return geo_factor;
} 


// Calculate the geometrics factor from the polynomial basis functions 
std::tuple<Array, Array> winding_volume_flux_jumps(Array& coil_points, Array& Phi, double dx, double dy, double dz) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(coil_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("coil_points needs to be in row-major storage order");
    if(Phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("Phi needs to be in row-major storage order");

    int num_coil_points = coil_points.shape(0);
    // here integration points should be shape (num_coil_points, Nx, Ny, Nz, 3)
    int Nx = Phi.shape(2);
    int Ny = Phi.shape(3);
    int Nz = Phi.shape(4);
    Array Connect = connections(coil_points, 6, dx, dy, dz);

    // here Phi should be shape (n_basis_functions, num_coil_points, Nx, Ny, Nz, 3)
    int num_basis_functions = Phi.shape(0);
    Array flux_factor = xt::zeros<double>({6, num_coil_points, num_basis_functions});
    Array flux_constraint_matrix = xt::zeros<double>({6 * num_coil_points, num_coil_points * num_basis_functions});
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_coil_points; i++) {
	for (int kk = 0; kk < 6; kk++) {
            int nx = 0;
            int ny = 0;
            int nz = 0;
	    if (kk == 0) nx = 1;
	    if (kk == 1) nx = -1;
	    if (kk == 2) ny = 1;
	    if (kk == 3) ny = -1;
	    if (kk == 4) nz = 1;
	    if (kk == 5) nz = -1;
	    if (kk < 2) {
		int x_ind = 0;
		if (kk == 0) x_ind = Nx - 1;
		for (int j = 0; j < Ny; j++) {
		    for (int p = 0; p < Nz; p++) {
			for (int k = 0; k < num_basis_functions; k++) {
			    flux_factor(kk, i, k) += (nx * Phi(k, i, x_ind, j, p, 0) + ny * Phi(k, i, x_ind, j, p, 1) + nz * Phi(k, i, x_ind, j, p, 2)) * dy * dz;
			}
		    }
		}
	    }
	    else if (kk < 4) {
		int y_ind = 0;
		if (kk == 2) y_ind = Ny - 1;
		for (int j = 0; j < Nx; j++) {
		    for (int p = 0; p < Nz; p++) {
			for (int k = 0; k < num_basis_functions; k++) {
			    flux_factor(kk, i, k) += (nx * Phi(k, i, j, y_ind, p, 0) + ny * Phi(k, i, j, y_ind, p, 1) + nz * Phi(k, i, j, y_ind, p, 2)) * dx * dz;
			}
		    }
		}
	    }
	    else {
		int z_ind = 0;
		if (kk == 4) z_ind = Nz - 1;
		for (int j = 0; j < Nx; j++) {
		    for (int p = 0; p < Ny; p++) {
			for (int k = 0; k < num_basis_functions; k++) {
			    flux_factor(kk, i, k) += (nx * Phi(k, i, j, p, z_ind, 0) + ny * Phi(k, i, j, p, z_ind, 1) + nz * Phi(k, i, j, p, z_ind, 2) * dx * dy);
			}
		    }
		}
            }
	}
	for (int k = 0; k < num_basis_functions; k++) {
            for (int kk = 0; kk < 6; kk++) {  // loop over the 6 possible directions 
		double Fik = flux_factor(kk, i, k);
		flux_constraint_matrix(i * 6 + kk, i * num_basis_functions + k) = Fik;
		for (int j = 0; j < 6; j++) {  // loop over the 6 neighbors
		    int q = Connect(i, j, kk);
		    if (q > 0) {
                        double Fqk = flux_factor(kk, q, k);
			flux_constraint_matrix(i * 6 + kk, q * num_basis_functions + k) = Fqk;
		    }
		}
	    }
	}
    }
    return std::make_tuple(flux_constraint_matrix, Connect);
}   


// Takes a uniform CARTESIAN grid of dipoles, and loops through
// and creates a final set of points which lie between the
// inner and outer toroidal surfaces defined by extending the plasma
// boundary by its normal vectors * some minimum distance. 
Array make_winding_volume_grid(Array& normal_inner, Array& normal_outer, Array& dipole_grid_xyz, Array& xyz_inner, Array& xyz_outer)
{
    // For each toroidal cross-section:
    // For each dipole location:
    //     1. Find nearest point from dipole to the inner surface
    //     2. Find nearest point from dipole to the outer surface
    //     3. Select nearest point that is closest to the dipole
    //     4. Get normal vector of this inner/outer surface point
    //     5. Draw ray from dipole location in the direction of this normal vector
    //     6. If closest point between inner surface and the ray is the 
    //           start of the ray, conclude point is outside the inner surface. 
    //     7. If closest point between outer surface and the ray is the
    //           start of the ray, conclude point is outside the outer surface. 
    //     8. If Step 4 was True but Step 5 was False, add the point to the final grid.
   	
    int num_inner = xyz_inner.shape(0);
    int ngrid = dipole_grid_xyz.shape(0);
    int num_ray = 500;
    Array final_grid = xt::zeros<double>({ngrid, 3});

    // Loop through every dipole (cant use openmp because sharing access to the vector final_grid)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrid; i++) {
        double X = dipole_grid_xyz(i, 0);
        double Y = dipole_grid_xyz(i, 1);
        double Z = dipole_grid_xyz(i, 2);
           
	// find nearest point on inner/outer toroidal surface
	double min_dist_inner = 1e5;
	double min_dist_outer = 1e5;
	int inner_loc = 0;
	int outer_loc = 0;
        for (int k = 0; k < num_inner; k++) {
	    double x_inner = xyz_inner(k, 0);
	    double y_inner = xyz_inner(k, 1);
	    double z_inner = xyz_inner(k, 2);
	    double x_outer = xyz_outer(k, 0);
	    double y_outer = xyz_outer(k, 1);
	    double z_outer = xyz_outer(k, 2);
	    double dist_inner = sqrt((x_inner - X) * (x_inner - X) + (y_inner - Y) * (y_inner - Y) + (z_inner - Z) * (z_inner - Z)); 
	    double dist_outer = sqrt((x_outer - X) * (x_outer - X) + (y_outer - Y) * (y_outer - Y) + (z_outer - Z) * (z_outer - Z)); 
            if (dist_inner < min_dist_inner) {
		min_dist_inner = dist_inner;
	        inner_loc = k;
	    }
            if (dist_outer < min_dist_outer) {
		min_dist_outer = dist_outer;
	        outer_loc = k;
	    }
	}   
	double nx = 0.0;
	double ny = 0.0;
	double nz = 0.0;
	if (min_dist_inner < min_dist_outer) {
            nx = normal_inner(inner_loc, 0);
            ny = normal_inner(inner_loc, 1);
	    nz = normal_inner(inner_loc, 2);
	}
	else {
            nx = normal_outer(outer_loc, 0);
            ny = normal_outer(outer_loc, 1);
	    nz = normal_outer(outer_loc, 2);
	}
	// normalize the normal vectors
	double norm_vec = sqrt(nx * nx + ny * ny + nz * ny);
        double ray_x = nx / norm_vec;
        double ray_y = ny / norm_vec;
        double ray_z = nz / norm_vec;
           
	// Compute all the rays and find the location of minimum ray-surface distance
	double dist_inner_ray = 0.0;
	double dist_outer_ray = 0.0;
	double min_dist_inner_ray = 1e5;
	double min_dist_outer_ray = 1e5;
        int nearest_loc_inner = 0;
        int nearest_loc_outer = 0;
	double ray_equation_x = 0.0;
	double ray_equation_y = 0.0;
        double ray_equation_z = 0.0;
        for (int k = 0; k < num_ray; k++) {
	    ray_equation_x = X + ray_x * (4.0 / ((double) num_ray)) * k;
	    ray_equation_y = Y + ray_y * (4.0 / ((double) num_ray)) * k;
	    ray_equation_z = Z + ray_z * (4.0 / ((double) num_ray)) * k;
	    dist_inner_ray = sqrt((xyz_inner(inner_loc, 0) - ray_equation_x) * (xyz_inner(inner_loc, 0) - ray_equation_x) + (xyz_inner(inner_loc, 1) - ray_equation_y) * (xyz_inner(inner_loc, 1) - ray_equation_y) + (xyz_inner(inner_loc, 2) - ray_equation_z) * (xyz_inner(inner_loc, 2) - ray_equation_z));
	    dist_outer_ray = sqrt((xyz_outer(outer_loc, 0) - ray_equation_x) * (xyz_outer(outer_loc, 0) - ray_equation_x) + (xyz_outer(outer_loc, 1) - ray_equation_y) * (xyz_outer(outer_loc, 1) - ray_equation_y) + (xyz_outer(outer_loc, 2) - ray_equation_z) * (xyz_outer(outer_loc, 2) - ray_equation_z));
            if (dist_inner_ray < min_dist_inner_ray) {
		min_dist_inner_ray = dist_inner_ray;
		nearest_loc_inner = k;
            }
            if (dist_outer_ray < min_dist_outer_ray) {
		min_dist_outer_ray = dist_outer_ray;
		nearest_loc_outer = k;
	    }
            
	    // nearest distance from the inner surface to the ray should be just the original point
	    if (nearest_loc_inner > 0)
                continue;
            
	    // nearest distance from the outer surface to the ray should NOT be the original point
            if (nearest_loc_outer > 0) {
                //vector<double> new_point(3);
		//new_point[0] = X;
		//new_point[1] = Y;
		//new_point[2] = Z;
		//final_grid.push_back(new_point);
		final_grid(i, 0) = X;
		final_grid(i, 1) = Y;
		final_grid(i, 2) = Z;
	    }
	}
    }
    return final_grid; 
}
