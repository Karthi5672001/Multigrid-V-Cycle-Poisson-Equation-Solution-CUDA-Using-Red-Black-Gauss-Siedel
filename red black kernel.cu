#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void FinetoCoarseCooperative(float *f, float *u, float *r, int Nx, int Ny, int Nz, const float h2, int nu1) {
    // Handle to the global grid for synchronization
    cg::grid_group grid = cg::this_grid();

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    bool is_boundary = (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx = z * sliceSize + y * Nx + x;
    int parity = (x + y + z) & 1;

    // Loop inside the kernel safely
    for (int iter = 0; iter < nu1; ++iter) {
        // Red Update
        if (!is_boundary && parity == 0) {
            float neighbors = u[idx - 1]          + u[idx + 1] + 
                              u[idx - Nx]         + u[idx + Nx] + 
                              u[idx - sliceSize]  + u[idx + sliceSize];
            u[idx] = inv6 * (neighbors - h2 * f[idx]);
        }
        
        // Synchronize the ENTIRE GPU grid across all blocks
        grid.sync();

        // Black Update
        if (!is_boundary && parity != 0) {
            float neighbors = u[idx - 1]          + u[idx + 1] + 
                              u[idx - Nx]         + u[idx + Nx] + 
                              u[idx - sliceSize]  + u[idx + sliceSize];
            u[idx] = inv6 * (neighbors - h2 * f[idx]);
        }

        // Synchronize again before the next iteration starts
        grid.sync();
    }

    // Final Residual Calculation
    if (!is_boundary) {
        float neighbors = u[idx - 1]          + u[idx + 1] + 
                          u[idx - Nx]         + u[idx + Nx] + 
                          u[idx - sliceSize]  + u[idx + sliceSize];
        float Laplacian = (neighbors - 6.0f * u[idx]) / h2;
        r[idx] = f[idx] - Laplacian;
    }
}
#include <vector>
#include <iostream>

struct CoarseGrid3D {
    int Nx, Ny, Nz;         // Coarse grid dimensions
    size_t total_elements;  // Total size of the flat array
    float H2;               // Coarse grid spacing squared (4 * h^2)
    float inv_diagonal;     // 1.0f / 6.0f (The inverse of the matrix diagonal)

    // Flat arrays representing the vectors the matrix operates on
    std::vector<float> u;   // Coarse solution / correction vector
    std::vector<float> f;   // Coarse right-hand side / restricted residual vector

    // Constructor to derive the coarse matrix from fine dimensions
    CoarseGrid3D(int fine_Nx, int fine_Ny, int fine_Nz, float fine_h2) {
        // Integer division assumes grid sizing matches multigrid requirements (e.g., Nx is odd)
        Nx = (fine_Nx / 2) + 1;
        Ny = (fine_Ny / 2) + 1;
        Nz = (fine_Nz / 2) + 1;
        
        total_elements = static_cast<size_t>(Nx) * Ny * Nz;
        H2 = 4.0f * fine_h2;   // H = 2h, so H^2 = 4h^2
        inv_diagonal = 1.0f / 6.0f;

        // Allocate memory for the vectors
        u.assign(total_elements, 0.0f);
        f.assign(total_elements, 0.0f);
    }
};
void ApplyCoarseMatrix(const CoarseGrid3D& grid, const std::vector<float>& x, std::vector<float>& y) {
    size_t sliceSize = static_cast<size_t>(grid.Nx) * grid.Ny;

    // Enforce boundary conditions (Matrix row is an identity 1 for boundaries)
    for (int z = 0; z < grid.Nz; ++z) {
        for (int y_idx = 0; y_idx < grid.Ny; ++y_idx) {
            for (int x_idx = 0; x_idx < grid.Nx; ++x_idx) {
                size_t idx = z * sliceSize + y_idx * grid.Nx + x_idx;

                // If on boundary, matrix row identity means y[idx] = x[idx]
                if (x_idx == 0 || x_idx == grid.Nx - 1 || 
                    y_idx == 0 || y_idx == grid.Ny - 1 || 
                    z == 0 || z == grid.Nz - 1) {
                    y[idx] = x[idx]; 
                    continue;
                }

                // 7-Point Coarse Stencil Multiplication
                float neighbors = x[idx - 1]         + x[idx + 1] + 
                                  x[idx - grid.Nx]   + x[idx + grid.Nx] + 
                                  x[idx - sliceSize] + x[idx + sliceSize];
                
                // Representing the row calculation: (-neighbors + 6 * center) / H2
                y[idx] = (6.0f * x[idx] - neighbors) / grid.H2;
            }
        }
    }
}
void RestrictFineToCoarse(int fine_Nx, int fine_Ny, const std::vector<float>& r_fine, CoarseGrid3D& coarse_grid) {
    size_t fine_slice = static_cast<size_t>(fine_Nx) * fine_Ny;
    size_t coarse_slice = static_cast<size_t>(coarse_grid.Nx) * coarse_grid.Ny;

    for (int z_c = 1; z_c < coarse_grid.Nz - 1; ++z_c) {
        int z_f = z_c * 2; // Map coarse coordinate to fine coordinate
        
        for (int y_c = 1; y_c < coarse_grid.Ny - 1; ++y_c) {
            int y_f = y_c * 2;
            
            for (int x_c = 1; x_c < coarse_grid.Nx - 1; ++x_c) {
                int x_f = x_c * 2;

                float sum = 0.0f;

                // Loop over a 3x3x3 stencil on the fine grid
                for (int dz = -1; dz <= 1; ++dz) {
                    float weight_z = (dz == 0) ? 0.5f : 0.25f;
                    
                    for (int dy = -1; dy <= 1; ++dy) {
                        float weight_y = (dy == 0) ? 0.5f : 0.25f;
                        
                        for (int dx = -1; dx <= 1; ++dx) {
                            float weight_x = (dx == 0) ? 0.5f : 0.25f;
                            
                            // Combine 1D weights into a 3D structural weight
                            float weight = weight_x * weight_y * weight_z;

                            size_t fine_idx = (z_f + dz) * fine_slice + (y_f + dy) * fine_Nx + (x_f + dx);
                            sum += weight * r_fine[fine_idx];
                        }
                    }
                }
                
                size_t coarse_idx = z_c * coarse_slice + y_c * coarse_grid.Nx + x_c;
                coarse_grid.f[coarse_idx] = sum; // Coarse matrix right hand side populated
            }
        }
    }
}
// inv6 should be defined globally: 
const float inv6 = 1.0f / 6.0f;
/*
__global__ void FinetoCoarse(
    const float *f, float *u, float *r, float *d_f_coarse,
    int Nx, int Ny, int Nz, 
    int coarse_Nx, int coarse_Ny,
    const float h2, int nu1)
{
    // Global 3D indices for the FINE grid
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check boundaries on the fine grid
    bool is_boundary = (x == 0 || x >= Nx - 1 || y == 0 || y >= Ny - 1 || z == 0 || z >= Nz - 1);
    bool out_of_bounds = (x >= Nx || y >= Ny || z >= Nz);
    if (out_of_bounds) return;

    size_t sliceSize = static_cast<size_t>(Nx) * Ny;
    size_t idx = z * sliceSize + y * Nx + x;
    int parity = (x + y + z) & 1;

    // --- STEP 1: FUSED RED-BLACK GAUSS-SEIDEL ITERATIONS ---
    for (int iter = 0; iter < nu1; ++iter) {
        
        // Red Phase
        if (!is_boundary && parity == 0) {
            float neighbors = u[idx - 1]          + u[idx + 1] + 
                              u[idx - Nx]         + u[idx + Nx] + 
                              u[idx - sliceSize]  + u[idx + sliceSize];
            u[idx] = inv6 * (neighbors - h2 * f[idx]);
        }
        
        // Sync within thread blocks to stabilize local updates
        __syncthreads(); 

        // Black Phase
        if (!is_boundary && parity != 0) {
            float neighbors = u[idx - 1]          + u[idx + 1] + 
                              u[idx - Nx]         + u[idx + Nx] + 
                              u[idx - sliceSize]  + u[idx + sliceSize];
            u[idx] = inv6 * (neighbors - h2 * f[idx]);
        }

        __syncthreads();
    }

    // --- STEP 2: RESIDUAL CALCULATION ---
    float res = 0.0f;
    if (!is_boundary) {
        float neighbors = u[idx - 1]          + u[idx + 1] + 
                          u[idx - Nx]         + u[idx + Nx] + 
                          u[idx - sliceSize]  + u[idx + sliceSize];
        
        float Laplacian = (neighbors - 6.0f * u[idx]) / h2;
        res = f[idx] - Laplacian;
        r[idx] = res; // Save fine residual
    } else {
        r[idx] = 0.0f; // Boundary residual is 0
    }

    // Ensure all threads in the block have written their residuals to global memory
    __threadfence(); 
    __syncthreads();

    // --- STEP 3: COARSENING (FULL-WEIGHTING RESTRICTION) ---
    // Only threads that map exactly onto valid interior coarse nodes execute this
    if (x > 0 && x < Nx - 1 && y > 0 && y < Ny - 1 && z > 0 && z < Nz - 1) {
        if ((x % 2 == 0) && (y % 2 == 0) && (z % 2 == 0)) {
            
            int x_c = x / 2;
            int y_c = y / 2;
            int z_c = z / 2;

            // Only proceed if it lands on an interior coarse node
            if (x_c > 0 && x_c < coarse_Nx - 1 && 
                y_c > 0 && y_c < coarse_Ny - 1 && 
                z_c > 0 && z_c < coarse_Nz - 1) {

                float coarse_residual_sum = 0.0f;

                // 27-point Full-Weighting accumulation from the fine residual array `r`
                for (int dz = -1; dz <= 1; ++dz) {
                    float wz = (dz == 0) ? 0.5f : 0.25f;
                    for (int dy = -1; dy <= 1; ++dy) {
                        float wy = (dy == 0) ? 0.5f : 0.25f;
                        for (int dx = -1; dx <= 1; ++dx) {
                            float wx = (dx == 0) ? 0.5f : 0.25f;
                            
                            float weight = wx * wy * wz;
                            size_t neighbor_fine_idx = (z + dz) * sliceSize + (y + dy) * Nx + (x + dx);
                            
                            coarse_residual_sum += weight * r[neighbor_fine_idx];
                        }
                    }
                }

                // Write out to the coarse matrix right-hand side array
                size_t coarse_slice = static_cast<size_t>(coarse_Nx) * coarse_num_y; // assuming coarse_Ny passed
                size_t coarse_idx = z_c * (static_cast<size_t>(coarse_Nx) * coarse_Ny) + y_c * coarse_Nx + x_c;
                d_f_coarse[coarse_idx] = coarse_residual_sum;
            }
        }
    }
}
*/

__global__ void FinetoCoarse(
    const float *f, float *u, float *r, float *d_f_coarse,
    int Nx, int Ny, int Nz, 
    int coarse_Nx, int coarse_Ny, int coarse_Nz,
    const float h2, int nu1)
{
    // Global 3D indices for the FINE grid
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check boundaries on the fine grid
    bool is_boundary = (x == 0 || x >= Nx - 1 || y == 0 || y >= Ny - 1 || z == 0 || z >= Nz - 1);
    bool out_of_bounds = (x >= Nx || y >= Ny || z >= Nz);
    if (out_of_bounds) return;

    size_t sliceSize = static_cast<size_t>(Nx) * Ny;
    size_t idx = z * sliceSize + y * Nx + x;
    int parity = (x + y + z) & 1;

    // --- STEP 1: FUSED RED-BLACK GAUSS-SEIDEL ITERATIONS ---
    for (int iter = 0; iter < nu1; ++iter) {
        
        // Red Phase
        if (!is_boundary && parity == 0) {
            float neighbors = u[idx - 1]          + u[idx + 1] + 
                              u[idx - Nx]         + u[idx + Nx] + 
                              u[idx - sliceSize]  + u[idx + sliceSize];
            u[idx] = inv6 * (neighbors - h2 * f[idx]);
        }
        
        // Sync within thread blocks to stabilize local updates
        __syncthreads(); 

        // Black Phase
        if (!is_boundary && parity != 0) {
            float neighbors = u[idx - 1]          + u[idx + 1] + 
                              u[idx - Nx]         + u[idx + Nx] + 
                              u[idx - sliceSize]  + u[idx + sliceSize];
            u[idx] = inv6 * (neighbors - h2 * f[idx]);
        }

        __syncthreads();
    }

    // --- STEP 2: RESIDUAL CALCULATION ---
    float res = 0.0f;
    if (!is_boundary) {
        float neighbors = u[idx - 1]          + u[idx + 1] + 
                          u[idx - Nx]         + u[idx + Nx] + 
                          u[idx - sliceSize]  + u[idx + sliceSize];
        
        float Laplacian = (neighbors - 6.0f * u[idx]) / h2;
        res = f[idx] - Laplacian;
        r[idx] = res; // Save fine residual
    } else {
        r[idx] = 0.0f; // Boundary residual is 0
    }

    // Ensure all threads in the block have written their residuals to global memory
    __threadfence(); 
    __syncthreads();

    // --- STEP 3: COARSENING (FULL-WEIGHTING RESTRICTION) ---
    // Only threads that map exactly onto valid interior coarse nodes execute this
    if (x > 0 && x < Nx - 1 && y > 0 && y < Ny - 1 && z > 0 && z < Nz - 1) {
        if ((x % 2 == 0) && (y % 2 == 0) && (z % 2 == 0)) {
            
            int x_c = x / 2;
            int y_c = y / 2;
            int z_c = z / 2;

            // Only proceed if it lands on an interior coarse node
            if (x_c > 0 && x_c < coarse_Nx - 1 && 
                y_c > 0 && y_c < coarse_Ny - 1 && 
                z_c > 0 && z_c < coarse_Nz - 1) {

                float coarse_residual_sum = 0.0f;

                // 27-point Full-Weighting accumulation from the fine residual array `r`
                for (int dz = -1; dz <= 1; ++dz) {
                    float wz = (dz == 0) ? 0.5f : 0.25f;
                    for (int dy = -1; dy <= 1; ++dy) {
                        float wy = (dy == 0) ? 0.5f : 0.25f;
                        for (int dx = -1; dx <= 1; ++dx) {
                            float wx = (dx == 0) ? 0.5f : 0.25f;
                            
                            float weight = wx * wy * wz;
                            size_t neighbor_fine_idx = (z + dz) * sliceSize + (y + dy) * Nx + (x + dx);
                            
                            coarse_residual_sum += weight * r[neighbor_fine_idx];
                        }
                    }
                }

                // Write out to the coarse matrix right-hand side array using explicit size_t typing
                size_t coarse_slice = static_cast<size_t>(coarse_Nx) * coarse_Ny;
                size_t coarse_idx = z_c * coarse_slice + y_c * coarse_Nx + x_c;
                d_f_coarse[coarse_idx] = coarse_residual_sum;
            }
        }
    }
}
