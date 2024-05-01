#include "psc.h"
#include <cstdio>

// Calculate the inductance matrix needed for the PSC forward problem
Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
    
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    Array L = xt::zeros<double>({num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* phi_ptr = &(phi(0));
    
    // Loop through the the PSC array
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        auto cai = cos(alphas_ptr[i]);
        auto sai = sin(alphas_ptr[i]);
        auto cdi = cos(deltas_ptr[i]);
        auto sdi = sin(deltas_ptr[i]);
        auto xi = points_ptr[3 * i];
        auto yi = points_ptr[3 * i + 1];
        auto zi = points_ptr[3 * i + 2];
    	// Loop through all j > i coils
        for (int j = (i + 1); j < num_coils; ++j) {
            auto xj = (points_ptr[3 * j] - xi) / R;
            auto yj = (points_ptr[3 * j + 1] - yi) / R;
            auto zj = (points_ptr[3 * j + 2] - zi) / R;
            auto caj = cos(alphas_ptr[j]);
            auto saj = sin(alphas_ptr[j]);
            auto cdj = cos(deltas_ptr[j]);
            auto sdj = sin(deltas_ptr[j]);
            auto integrand = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto ck = cos(phi_ptr[k]);
                auto sk = sin(phi_ptr[k]);
                for (int kk = 0; kk < num_phi; ++kk) {
                    auto ckk = cos(phi_ptr[kk]);
                    auto skk = sin(phi_ptr[kk]);
//                     auto dl2_x = (-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj);
//                     auto dl2_y = (ck * cai) * (ckk * caj);
//                     auto dl2_z = (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj);
                    auto x2 = (xj + ckk * cdj + skk * saj * sdj - ck * cdi - sk * sai * sdi);
                    auto y2 = (yj + skk * caj - sk * cai);
                    auto z2 = (zj + skk * saj * cdj - ckk * sdj - sk * sai * cdi + ck * sdi);
                    integrand += ((-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj) + (ck * cai) * (ckk * caj) + (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj)) / sqrt(x2 * x2 + y2 * y2 + z2 * z2);
                }
            }
            L(i, j) = integrand;
        }
    }
    return L;
}

// Calculate the inductance matrix needed for the PSC forward problem
Array L_deriv(Array& points, Array& alphas, Array& deltas, Array& phi, double R)
{
    // points shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    Array L_deriv = xt::zeros<double>({2 * num_coils, num_coils, num_coils});
    
    // initialize pointers to the beginning of alphas, deltas, points
    double* points_ptr = &(points(0, 0));
    double* alphas_ptr = &(alphas(0));
    double* deltas_ptr = &(deltas(0));
    double* phi_ptr = &(phi(0));
    
    // Loop through the the PSC array
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_coils; i++) {
        auto cai = cos(alphas_ptr[i]);
        auto sai = sin(alphas_ptr[i]);
        auto cdi = cos(deltas_ptr[i]);
        auto sdi = sin(deltas_ptr[i]);
        auto xi = points_ptr[3 * i];
        auto yi = points_ptr[3 * i + 1];
        auto zi = points_ptr[3 * i + 2];
    	// Loop through all j > i coils
        for (int j = (i + 1); j < num_coils; ++j) {
            auto xj = (points_ptr[3 * j] - xi) / R;
            auto yj = (points_ptr[3 * j + 1] - yi) / R;
            auto zj = (points_ptr[3 * j + 2] - zi) / R;
            auto caj = cos(alphas_ptr[j]);
            auto saj = sin(alphas_ptr[j]);
            auto cdj = cos(deltas_ptr[j]);
            auto sdj = sin(deltas_ptr[j]);
            auto integrand1 = 0.0;
            auto integrand2 = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto ck = cos(phi_ptr[k]);
                auto sk = sin(phi_ptr[k]);
                for (int kk = 0; kk < num_phi; ++kk) {
                    auto ckk = cos(phi_ptr[kk]);
                    auto skk = sin(phi_ptr[kk]);
                    auto dl2_x = (-sk * cdi + ck * sai * sdi) * (-skk * cdj + ckk * saj * sdj);
                    auto dl2_y = (ck * cai) * (ckk * caj);
                    auto dl2_z = (sk * sdi + ck * sai * cdi) * (skk * sdj + ckk * saj * cdj);
                    // dkappa of the numerator
                    auto dl2_x_dalpha = (ck * cai * sdi) * (-skk * cdj + ckk * saj * sdj);
                    auto dl2_y_dalpha = -(ck * sai) * (ckk * caj);
                    auto dl2_z_dalpha = (ck * cai * cdi) * (skk * sdj + ckk * saj * cdj);
                    auto dl2_x_ddelta = (sk * sdi + ck * sai * cdi) * (-skk * cdj + ckk * saj * sdj);
                    auto dl2_y_ddelta = 0.0;
                    auto dl2_z_ddelta = (sk * cdi - ck * sai * sdi) * (skk * sdj + ckk * saj * cdj);
                    auto xxi = ck * cdi + sk * sai * sdi;  // fixed minus sign on the last term here
                    auto xxj = xj + ckk * cdj + skk * saj * sdj;
                    auto yyi = sk * cai;
                    auto yyj = yj + skk * caj;
                    auto zzi = sk * sai * cdi - ck * sdi;
                    auto zzj = zj + skk * saj * cdj - ckk * sdj;
                    auto xxi_dalpha = sk * cai * sdi; //0.0;
                    auto yyi_dalpha = -sk * sai;
                    auto zzi_dalpha = sk * cai * cdi;
                    auto xxi_ddelta = -ck * sdi + sk * sai * cdi;
                    auto yyi_ddelta = 0.0;
                    auto zzi_ddelta = -sk * sai * sdi - ck * cdi;
                    auto deriv_alpha = xxi_dalpha * (xxi - xxj) + yyi_dalpha * (yyi - yyj) + zzi_dalpha * (zzi - zzj);
                    auto deriv_delta = xxi_ddelta * (xxi - xxj) + yyi_ddelta * (yyi - yyj) + zzi_ddelta * (zzi - zzj);
                    auto x2 = (xxj - xxi) * (xxj - xxi);
                    auto y2 = (yyj - yyi) * (yyj - yyi);
                    auto z2 = (zzj - zzi) * (zzj - zzi);
                    auto denom = sqrt(x2 + y2 + z2);
                    auto denom3 = denom * denom * denom;
                    // First term in the derivative
                    integrand1 += (dl2_x_dalpha + dl2_y_dalpha + dl2_z_dalpha) / denom;
                    integrand2 += (dl2_x_ddelta + dl2_y_ddelta + dl2_z_ddelta) / denom;
                    // Second term in the derivative
                    integrand1 -= (dl2_x + dl2_y + dl2_z) * deriv_alpha / denom3;
                    integrand2 -= (dl2_x + dl2_y + dl2_z) * deriv_delta / denom3;
                }
            }
            L_deriv(i, i, j) = integrand1;
            L_deriv(i + num_coils, i, j) = integrand2;
        }
    }
    return L_deriv;
}

Array flux_xyz(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& normal)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas normal needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(rho.layout() != xt::layout_type::row_major)
          throw std::runtime_error("rho needs to be in row-major storage order");
    if(phi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("phi needs to be in row-major storage order");
    if(normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // normal shape should be (num_coils, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    int num_rho = rho.shape(0);  // shape should be (num_rho)
    Array XYZs = xt::zeros<double>({num_coils, num_rho, num_phi, 3});
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto cdj = cos(deltas(j));
        auto caj = cos(alphas(j));
        auto sdj = sin(deltas(j));
        auto saj = sin(alphas(j));
        auto nx = normal(j, 0);
        auto ny = normal(j, 1);
        auto nz = normal(j, 2);
        auto xj = points(j, 0);
        auto yj = points(j, 1);
        auto zj = points(j, 2);
        for (int k = 0; k < num_rho; ++k) {
            auto xx = rho(k);
            for (int kk = 0; kk < num_phi; ++kk) {
                auto yy = phi(kk);
                auto x0 = xx * cos(yy);
                auto y0 = xx * sin(yy);
                auto x = x0 * cdj + y0 * saj * sdj + xj;
                auto y = y0 * caj + yj;
                auto z = -x0 * sdj + y0 * saj * cdj + zj;
                XYZs(j, k, kk, 0) = x;
                XYZs(j, k, kk, 1) = y;
                XYZs(j, k, kk, 2) = z;
            }
        }
    }
    return XYZs;
}

Array flux_integration(Array& B, Array& rho, Array& normal)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(B.layout() != xt::layout_type::row_major)
          throw std::runtime_error("B needs to be in row-major storage order");
    if(rho.layout() != xt::layout_type::row_major)
          throw std::runtime_error("rho needs to be in row-major storage order");
    if(normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal needs to be in row-major storage order");
          
    // normal shape should be (num_coils, 3)
    int num_coils = B.shape(0);  // shape should be (num_coils, N, Nphi, 3)
    int N = rho.shape(0);  // shape should be (N)
    int Nphi = B.shape(2);
    Array Psi = xt::zeros<double>({num_coils}); 
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto nx = normal(j, 0);
        auto ny = normal(j, 1);
        auto nz = normal(j, 2);
        double integral = 0.0;
        for(int k = 0; k < N; k++) {
            auto xx = rho(k);
            for(int kk = 0; kk < Nphi; kk++) {
                auto Bn = B(j, k, kk, 0) * nx + B(j, k, kk, 1) * ny + B(j, k, kk, 2) * nz; 
                integral += Bn * xx;
            }
        }
        Psi(j) = integral;
    }
    return Psi;
}

Array A_matrix(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(plasma_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_points needs to be in row-major storage order");
    if(plasma_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array A = xt::zeros<double>({num_plasma_points, num_coils});
    double R2 = R * R;
    double fac = 2.0e-7;
    using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < num_coils; j++) {
        auto cdj = cos(deltas(j));
        auto caj = cos(alphas(j));
        auto sdj = sin(deltas(j));
        auto saj = sin(alphas(j));
        auto xj = points(j, 0);
        auto yj = points(j, 1);
        auto zj = points(j, 2);
        for (int i = 0; i < num_plasma_points; ++i) {
            auto xp = plasma_points(i, 0);
            auto yp = plasma_points(i, 1);
            auto zp = plasma_points(i, 2);
            auto nx = plasma_normal(i, 0);
            auto ny = plasma_normal(i, 1);
            auto nz = plasma_normal(i, 2);
            auto x0 = (xp - xj);
            auto y0 = (yp - yj);
            auto z0 = (zp - zj);
            auto nxx = cdj;
            auto nxy = sdj * saj;
            auto nxz = sdj * caj;
            auto nyx = 0.0;
            auto nyy = caj;
            auto nyz = -saj;
            auto nzx = -sdj;
            auto nzy = cdj * saj;
            auto nzz = cdj * caj;
            auto x = x0 * nxx + y0 * nyx + z0 * nzx;
            auto y = x0 * nxy + y0 * nyy + z0 * nzy;
            auto z = x0 * nxz + y0 * nyz + z0 * nzz;
            auto rho2 = x * x + y * y;
            auto r2 = rho2 + z * z;
            auto rho = sqrt(rho2);
            auto R2_r2 = R2 + r2;
            auto gamma2 = R2_r2 - 2.0 * R * rho;
            auto beta2 = R2_r2 + 2.0 * R * rho;
            auto beta = sqrt(beta2);
            auto k2 = 1.0 - gamma2 / beta2;
            auto beta_gamma2 = beta * gamma2;
            auto k = sqrt(k2);
            auto ellipe = ellint_2(k);
            auto ellipk = ellint_1(k);
            auto Eplus = R2_r2 * ellipe - gamma2 * ellipk;
            auto Eminus = (R2 - r2) * ellipe + gamma2 * ellipk;
            auto Bx = x * z * Eplus / (rho2 * beta_gamma2);
            auto By = y * z * Eplus / (rho2 * beta_gamma2);
            auto Bz = Eminus / beta_gamma2;
            // Need to rotate the vector
            auto Bx_rot = Bx * nxx + By * nxy + Bz * nxz;
            auto By_rot = Bx * nyx + By * nyy + Bz * nyz;
            auto Bz_rot = Bx * nzx + By * nzy + Bz * nzz;
            A(i, j) = Bx_rot * nx + By_rot * ny + Bz_rot * nz;
        }
    }
    return A * fac;
}

Array B_PSC(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& psc_currents, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(plasma_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_points needs to be in row-major storage order");
    if(psc_currents.layout() != xt::layout_type::row_major)
          throw std::runtime_error("psc_currents needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array B_PSC = xt::zeros<double>({num_plasma_points, 3});
    double R2 = R * R;
    double fac = 2.0e-7;
    using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points(i, 0);
        auto yp = plasma_points(i, 1);
        auto zp = plasma_points(i, 2);
        for(int j = 0; j < num_coils; j++) {
            auto current = psc_currents(j);
            auto cdj = cos(deltas(j));
            auto caj = cos(alphas(j));
            auto sdj = sin(deltas(j));
            auto saj = sin(alphas(j));
            auto x0 = (xp - points(j, 0));
            auto y0 = (yp - points(j, 1));
            auto z0 = (zp - points(j, 2));
            auto nxx = cdj;
            auto nxy = sdj * saj;
            auto nxz = sdj * caj;
            auto nyx = 0.0;
            auto nyy = caj;
            auto nyz = -saj;
            auto nzx = -sdj;
            auto nzy = cdj * saj;
            auto nzz = cdj * caj;
            // multiply by R^T
            auto x = x0 * nxx + y0 * nyx + z0 * nzx;
            auto y = x0 * nxy + y0 * nyy + z0 * nzy;
            auto z = x0 * nxz + y0 * nyz + z0 * nzz;
            auto rho2 = x * x + y * y;
            auto r2 = rho2 + z * z;
            auto rho = sqrt(rho2);
            auto R2_r2 = R2 + r2;
            auto gamma2 = R2_r2 - 2.0 * R * rho;
            auto beta2 = R2_r2 + 2.0 * R * rho;
            auto beta = sqrt(beta2);
            auto k2 = 1.0 - gamma2 / beta2;
            auto beta_gamma2 = beta * gamma2;
            auto k = sqrt(k2);
            auto ellipe = ellint_2(k);
            auto ellipk = ellint_1(k);
            auto Eplus = R2_r2 * ellipe - gamma2 * ellipk;
            auto Eminus = (R2 - r2) * ellipe + gamma2 * ellipk;
            auto Bx = current * x * z * Eplus / (rho2 * beta_gamma2);
            auto By = current * y * z * Eplus / (rho2 * beta_gamma2);
            auto Bz = current * Eminus / beta_gamma2;
            // Need to rotate the vector
            B_PSC(i, 0) += Bx * nxx + By * nxy + Bz * nxz;
            B_PSC(i, 1) += Bx * nyx + By * nyy + Bz * nyz;
            B_PSC(i, 2) += Bx * nzx + By * nzy + Bz * nzz;
        }
    }
    return B_PSC * fac;
}


Array A_matrix_direct(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(plasma_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_points needs to be in row-major storage order");
   
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array A = xt::zeros<double>({num_plasma_points, num_coils});
    double fac = 1.0e-7;
    using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points(i, 0);
        auto yp = plasma_points(i, 1);
        auto zp = plasma_points(i, 2);
        auto nx = plasma_normal(i, 0);
        auto ny = plasma_normal(i, 1);
        auto nz = plasma_normal(i, 2);
        for(int j = 0; j < num_coils; j++) {
            auto cdj = cos(deltas(j));
            auto caj = cos(alphas(j));
            auto sdj = sin(deltas(j));
            auto saj = sin(alphas(j));
            auto xk = points(j, 0);
            auto yk = points(j, 1);
            auto zk = points(j, 2);
            auto x0 = xp - xk;
            auto y0 = yp - yk;
            auto z0 = zp - zk;
            auto Rxx = cdj;
            auto Rxy = sdj * saj;
            auto Rxz = sdj * caj;
            auto Ryx = 0.0;
            auto Ryy = caj;
            auto Ryz = -saj;
            auto Rzx = -sdj;
            auto Rzy = cdj * saj;
            auto Rzz = cdj * caj;
            auto Bx = 0.0;
            auto By = 0.0;
            auto Bz = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto dlx = -R * sin(phi(k));
                auto dly = R * cos(phi(k));
                auto dlz = 0.0;
                // multiply by R^T
                auto RTxdiff = (Rxx * x0 + Ryx * y0 + Rzx * z0) - R * cos(phi(k));
                auto RTydiff = (Rxy * x0 + Ryy * y0 + Rzy * z0) - R * sin(phi(k));
                auto RTzdiff = (Rxz * x0 + Ryz * y0 + Rzz * z0);
//                 auto RTxdiff = (cdj * xp - sdj * zp - xk);
//                 auto RTydiff = (sdj * saj * xp + caj * yp + cdj * saj * zp - yk);
//                 auto RTzdiff = (sdj * caj * xp - saj * yp + cdj * caj * zp - zk);
                auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                auto denom3 = denom * denom * denom;
                Bx += dl_cross_RTdiff_x / denom3;
                By += dl_cross_RTdiff_y / denom3;
                Bz += dl_cross_RTdiff_z / denom3;
            }
            // rotate by R
            A(i, j) = (Rxx * Bx + Rxy * By + Rxz * Bz) * nx + (Ryx * Bx + Ryy * By + Ryz * Bz) * ny + (Rzx * Bx + Rzy * By + Rzz * Bz) * nz;
        }
    }
    return A * fac;
}



Array dA_dkappa(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& phi, double R)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(alphas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("alphas needs to be in row-major storage order");
    if(deltas.layout() != xt::layout_type::row_major)
          throw std::runtime_error("deltas needs to be in row-major storage order");
    if(plasma_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_points needs to be in row-major storage order");
    if(plasma_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("plasma_normal needs to be in row-major storage order");
          
    // points shape should be (num_coils, 3)
    // plasma_normal shape should be (num_plasma_points, 3)
    // plasma_points should be (num_plasma_points, 3)
    int num_coils = alphas.shape(0);  // shape should be (num_coils)
    int num_plasma_points = plasma_points.shape(0);
    int num_phi = phi.shape(0);  // shape should be (num_phi)
    
    // this variable is the A matrix in the least-squares term so A * I = Bn
    Array dA = xt::zeros<double>({num_coils * 2, num_plasma_points});
    double fac = 1.0e-7;
    using namespace boost::math;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_plasma_points; ++i) {
        auto xp = plasma_points(i, 0);
        auto yp = plasma_points(i, 1);
        auto zp = plasma_points(i, 2);
        auto nx = plasma_normal(i, 0);
        auto ny = plasma_normal(i, 1);
        auto nz = plasma_normal(i, 2);
        for(int j = 0; j < num_coils; j++) {
            auto cdj = cos(deltas(j));
            auto caj = cos(alphas(j));
            auto sdj = sin(deltas(j));
            auto saj = sin(alphas(j));
            auto xk = points(j, 0);
            auto yk = points(j, 1);
            auto zk = points(j, 2);
            auto x0 = xp - xk;
            auto y0 = yp - yk;
            auto z0 = zp - zk;
            auto Rxx = cdj;
            auto Rxy = sdj * saj;
            auto Rxz = sdj * caj;
            auto Ryx = 0.0;
            auto Ryy = caj;
            auto Ryz = -saj;
            auto Rzx = -sdj;
            auto Rzy = cdj * saj;
            auto Rzz = cdj * caj;
            auto dRxx_dalpha = 0.0;
            auto dRxy_dalpha = sdj * caj;
            auto dRxz_dalpha = -sdj * saj;
            auto dRyx_dalpha = 0.0;
            auto dRyy_dalpha = -saj;
            auto dRyz_dalpha = -caj;
            auto dRzx_dalpha = 0.0;
            auto dRzy_dalpha = cdj * caj;
            auto dRzz_dalpha = -cdj * saj;
            auto dRxx_ddelta = -sdj;
            auto dRxy_ddelta = cdj * saj;
            auto dRxz_ddelta = cdj * caj;
            auto dRyx_ddelta = 0.0;
            auto dRyy_ddelta = 0.0;
            auto dRyz_ddelta = 0.0;
            auto dRzx_ddelta = -cdj;
            auto dRzy_ddelta = -sdj * saj;
            auto dRzz_ddelta = -sdj * caj;
            auto Bx1 = 0.0;
            auto By1 = 0.0;
            auto Bz1 = 0.0;
            auto Bx2 = 0.0;
            auto By2 = 0.0;
            auto Bz2 = 0.0;
            auto Bx3 = 0.0;
            auto By3 = 0.0;
            auto Bz3 = 0.0;
            auto Bx4 = 0.0;
            auto By4 = 0.0;
            auto Bz4 = 0.0;
            auto Bx5 = 0.0;
            auto By5 = 0.0;
            auto Bz5 = 0.0;
            for (int k = 0; k < num_phi; ++k) {
                auto dlx = -R * sin(phi(k));
                auto dly = R * cos(phi(k));
                auto dlz = 0.0;
                // multiply by R^T and then subtract off coil coordinate
                auto RTxdiff = (Rxx * x0 + Ryx * y0 + Rzx * z0) - R * cos(phi(k));
                auto RTydiff = (Rxy * x0 + Ryy * y0 + Rzy * z0) - R * sin(phi(k));
                auto RTzdiff = (Rxz * x0 + Ryz * y0 + Rzz * z0);
                auto dl_cross_RTdiff_x = dly * RTzdiff - dlz * RTydiff;
                auto dl_cross_RTdiff_y = dlz * RTxdiff - dlx * RTzdiff;
                auto dl_cross_RTdiff_z = dlx * RTydiff - dly * RTxdiff;
                auto denom = sqrt(RTxdiff * RTxdiff + RTydiff * RTydiff + RTzdiff * RTzdiff);
                auto denom3 = denom * denom * denom;
                auto denom5 = denom3 * denom * denom;
                // First derivative contribution of three
                Bx1 += dl_cross_RTdiff_x / denom3;
                By1 += dl_cross_RTdiff_y / denom3;
                Bz1 += dl_cross_RTdiff_z / denom3;
                // second derivative contribution (should be dRT/dalpha)
                auto dR_dalphaT_x = dRxx_dalpha * x0 + dRyx_dalpha * y0 + dRzx_dalpha * z0;
                auto dR_dalphaT_y = dRxy_dalpha * x0 + dRyy_dalpha * y0 + dRzy_dalpha * z0;
                auto dR_dalphaT_z = dRxz_dalpha * x0 + dRyz_dalpha * y0 + dRzz_dalpha * z0;
                auto dR_ddeltaT_x = dRxx_ddelta * x0 + dRyx_ddelta * y0 + dRzx_ddelta * z0;
                auto dR_ddeltaT_y = dRxy_ddelta * x0 + dRyy_ddelta * y0 + dRzy_ddelta * z0;
                auto dR_ddeltaT_z = dRxz_ddelta * x0 + dRyz_ddelta * y0 + dRzz_ddelta * z0;
                auto dl_cross_dR_dalphaT_x = dly * dR_dalphaT_z - dlz * dR_dalphaT_y;
                auto dl_cross_dR_dalphaT_y = dlz * dR_dalphaT_x - dlx * dR_dalphaT_z;
                auto dl_cross_dR_dalphaT_z = dlx * dR_dalphaT_y - dly * dR_dalphaT_x;
                auto dl_cross_dR_ddeltaT_x = dly * dR_ddeltaT_z - dlz * dR_ddeltaT_y;
                auto dl_cross_dR_ddeltaT_y = dlz * dR_ddeltaT_x - dlx * dR_ddeltaT_z;
                auto dl_cross_dR_ddeltaT_z = dlx * dR_ddeltaT_y - dly * dR_ddeltaT_x;
                Bx2 += dl_cross_dR_dalphaT_x / denom3;
                By2 += dl_cross_dR_dalphaT_y / denom3;
                Bz2 += dl_cross_dR_dalphaT_z / denom3;
                Bx4 += dl_cross_dR_ddeltaT_x / denom3;
                By4 += dl_cross_dR_ddeltaT_y / denom3;
                Bz4 += dl_cross_dR_ddeltaT_z / denom3;
                // third derivative contribution
                auto RTxdiff_dot_dR_dalpha = RTxdiff * dR_dalphaT_x + RTydiff * dR_dalphaT_y + RTzdiff * dR_dalphaT_z;
                auto RTxdiff_dot_dR_ddelta = RTxdiff * dR_ddeltaT_x + RTydiff * dR_ddeltaT_y + RTzdiff * dR_ddeltaT_z;
                Bx3 += - 3.0 * dl_cross_RTdiff_x * RTxdiff_dot_dR_dalpha / denom5;
                By3 += - 3.0 * dl_cross_RTdiff_y * RTxdiff_dot_dR_dalpha / denom5;
                Bz3 += - 3.0 * dl_cross_RTdiff_z * RTxdiff_dot_dR_dalpha / denom5;
                Bx5 += - 3.0 * dl_cross_RTdiff_x * RTxdiff_dot_dR_ddelta / denom5;
                By5 += - 3.0 * dl_cross_RTdiff_y * RTxdiff_dot_dR_ddelta / denom5;
                Bz5 += - 3.0 * dl_cross_RTdiff_z * RTxdiff_dot_dR_ddelta / denom5;
            }
            // rotate first contribution by dR/dalpha
            dA(j, i) += (dRxx_dalpha * Bx1 + dRxy_dalpha * By1 + dRxz_dalpha * Bz1) * nx;
            dA(j, i) += (dRyx_dalpha * Bx1 + dRyy_dalpha * By1 + dRyz_dalpha * Bz1) * ny;
            dA(j, i) += (dRzx_dalpha * Bx1 + dRzy_dalpha * By1 + dRzz_dalpha * Bz1) * nz;
            // rotate second and third contribution by R
            dA(j, i) += (Rxx * (Bx2 + Bx3) + Rxy * (By2 + By3) + Rxz * (Bz2 + Bz3)) * nx;
            dA(j, i) += (Ryx * (Bx2 + Bx3) + Ryy * (By2 + By3) + Ryz * (Bz2 + Bz3)) * ny;
            dA(j, i) += (Rzx * (Bx2 + Bx3) + Rzy * (By2 + By3) + Rzz * (Bz2 + Bz3)) * nz;
            // repeat for delta derivative
            dA(j + num_coils, i) += (dRxx_ddelta * Bx1 + dRxy_ddelta * By1 + dRxz_ddelta * Bz1) * nx;
            dA(j + num_coils, i) += (dRyx_ddelta * Bx1 + dRyy_ddelta * By1 + dRyz_ddelta * Bz1) * ny;
            dA(j + num_coils, i) += (dRzx_ddelta * Bx1 + dRzy_ddelta * By1 + dRzz_ddelta * Bz1) * nz;
            dA(j + num_coils, i) += (Rxx * (Bx4 + Bx5) + Rxy * (By4 + By5) + Rxz * (Bz4 + Bz5)) * nx;
            dA(j + num_coils, i) += (Ryx * (Bx4 + Bx5) + Ryy * (By4 + By5) + Ryz * (Bz4 + Bz5)) * ny;
            dA(j + num_coils, i) += (Rzx * (Bx4 + Bx5) + Rzy * (By4 + By5) + Rzz * (Bz4 + Bz5)) * nz;
        }
    }
    return dA * fac;
}
