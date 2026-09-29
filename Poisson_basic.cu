/*
Compilation in Docker WSL in powershell:
cd your/working/directory
docker build -t cuda133-cpp-dev .
docker run --gpus all -it -v ${PWD}:/workspace cuda133-cpp-dev bash
nvcc --version
nvcc -O3 --use_fast_math --generate-code arch=compute_120,code=sm_120 --ptxas-options=-v Poisson_basic.cu -o Poisson_solver
nvcc -O3 --generate-code arch=compute_120,code=sm_120 --ptxas-options=-v Poisson_basic.cu -o Poisson_solver
./Poisson_solver
*/
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <nvtx3/nvtx3.hpp>
/*--------------------------------------------------------------
  CUDA error Macro
----------------------------------------------------------------*/
//For tracking CUDA errors cleanly
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
/*--------------------------------------------------------------
  Helper: linear index from (x,y,z)
---------------------------------------------------------------*/
__device__ __inline__ size_t getLinearIdx3D(int x, int y, int z, const size_t Nx, const size_t Ny) {
    // Keeps the 2 sequential hardware Multiply-Accumulate instructions using pure integer math
    return (size_t)z * Nx * Ny + (size_t)y * Nx + x;
}
/*--------------------------------------------------------------
  Initialization Kernel: Sets up Manufactured Problem on Level 0
----------------------------------------------------------------*/
__global__ void InitializeProblem(float* __restrict__ f, 
                                  float* __restrict__ u_true, 
                                  const size_t Nx, 
                                  const size_t Ny, 
                                  const size_t Nz, 
                                  const float h) 
{   // Use size_t for avoiding out of bounds errors
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Out of Bounds check
    if (x >= Nx || y >= Ny || z >= Nz) return;

    // Uses linear index to get current cell
    size_t i = getLinearIdx3D((int)x, (int)y, (int)z, Nx, Ny);

    // Calculate physical continuous coordinates in space [0.0, 1.0]
    float px = x * h;
    float py = y * h;
    float pz = z * h;

    // Explicitly define Pi for mathematical precision
    constexpr float PI = 3.141592653589793f;

    // Evaluate exact trigonometric components
    float sin_x = sinf(PI * px);
    float sin_y = sinf(PI * py);
    float sin_z = sinf(PI * pz);

    // Populate arrays
    u_true[i] = sin_x * sin_y * sin_z;
    f[i]      = 3.0f * PI * PI * u_true[i];
}
/*--------------------------------------------------------------
  Error Square Kernel: Prepares the RMS calculation
----------------------------------------------------------------*/
__global__ void ComputeSquaredError(const float* __restrict__ u, 
                                    const float* __restrict__ u_true, 
                                    float* __restrict__ errorSquared, 
                                    const size_t totalElements) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    float diff = u[idx] - u_true[idx];
    errorSquared[idx] = diff * diff;
}
/*--------------------------------------------------------------
  Red kernel
---------------------------------------------------------------*/
__global__ void red_black_gauss_siedel_red(const float * __restrict__ f, 
                                           float * __restrict__ u, 
                                           const size_t Nx, const size_t Ny, const size_t Nz, 
                                           const float h, const float h2){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    // Define one sixth constant for multiplication
    constexpr float inv6 = 1.0f / 6.0f;
    // Boundry check
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    // Skip odd (black) cells using parity check
    if (((x + y + z) & 1) != 0) return; 
    // Calls linear index for current step
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    // Right-hand side
    float rhs = f[i];
    // Set sum to 0 initially
    float sum = 0.0f;
    // Summation of seven point stencil
    sum = (u[i-1] + u[i+1] + u[i+Nx] + u[i-Nx] + u[i-sliceSize] + u[i+sliceSize]);
    // Division is expensive on GPU, so we replace it with multiplication by h2
    u[i] = inv6 * (sum + h2 * rhs);
}
/*--------------------------------------------------------------
  Black kernel
---------------------------------------------------------------*/
__global__ void red_black_gauss_siedel_black(const float * __restrict__ f, float * __restrict__ u, 
                                             const size_t Nx, const size_t Ny, const size_t Nz, 
                                             const float h, const float h2){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    // Define one sixth constant for multiplication
    constexpr float inv6 = 1.0f / 6.0f;
    // Boundry check
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    // Skip even (red) cells using parity check
    if (((x + y + z) & 1) == 0) return; 
    // Calls linear index for current step
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    // Right-hand side
    float rhs = f[i];
    // Set sum to 0 initially
    float sum = 0.0f;
    // Summation of seven point stencil
    sum = (u[i-1] + u[i+1] + u[i+Nx] + u[i-Nx] + u[i-sliceSize] + u[i+sliceSize]);
    // Division is expensive on GPU, so we replace it with multiplication by h2
    u[i] = inv6 * (sum + h2 * rhs);
}
/*--------------------------------------------------------------
  Residual function
---------------------------------------------------------------*/
__global__ void Residual_function(const float * __restrict__ f, float * __restrict__ r, const float * __restrict__ u, // Added const qualifier for caching safety
                                  const size_t Nx, const size_t Ny, const size_t Nz, 
                                  const float h, const float h2){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    // We ensure that we stay within the 0-padding
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    // Define constant for multiplication
    float invH2 = 1.0f / h2;
    // Calls linear index for current step
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    // Make array for Laplacian (-∇²u) 0-padded, to ensure that no checks are necessary
    float Laplacian = -(u[i-1] + u[i+1] + u[i-Nx] + u[i+Nx] + u[i-sliceSize] + u[i+sliceSize] - 6.0f*u[i]) * invH2;
    r[i] = f[i] - Laplacian;
}
/*--------------------------------------------------------------
  Coarsen kernel
---------------------------------------------------------------*/
__global__ void Coarsen(const float* __restrict__ fineGrid, float* __restrict__ coarseGrid, 
                        const size_t Nx, const size_t Ny, const size_t Nz, 
                        const size_t fNx, const size_t fNy, const size_t fNz) {
    // Map thread to Coarse Grid Coordinates
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    // Boundry check
    if (x >= Nx || y >= Ny || z >= Nz) return;
    // Find the matching center point on the Fine Grid
    // Cast to int to align with helper's parameter signatures
    int fx = (int)x * 2;
    int fy = (int)y * 2;
    int fz = (int)z * 2;
    float sum = 0.0f;
    // 27-point stencil loop
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                // Determine stencil weight based on distance from center
                int dist = abs(dx) + abs(dy) + abs(dz);
                float weight = 0.0f;
                if (dist == 0)      weight = 1.0f / 8.0f;   // Center
                else if (dist == 1) weight = 1.0f / 16.0f;  // Faces
                else if (dist == 2) weight = 1.0f / 32.0f;  // Edges
                else if (dist == 3) weight = 1.0f / 64.0f;  // Corners
                // Clamp boundaries (Simple Dirichlet/Neumann handling)
                int s_fx = max(0, min(fx + dx, (int)fNx - 1));
                int s_fy = max(0, min(fy + dy, (int)fNy - 1));
                int s_fz = max(0, min(fz + dz, (int)fNz - 1));
                // Pass clamped INT coordinates and FINE grid dimensions
                size_t fineIdx = getLinearIdx3D(s_fx, s_fy, s_fz, fNx, fNy);
                sum += fineGrid[fineIdx] * weight;
            }
        }
    }
    // Write coalesced result to Coarse Grid
    // Pass current coarse coordinates and COARSE grid dimensions
    size_t coarseIdx = getLinearIdx3D((int)x, (int)y, (int)z, Nx, Ny);
    coarseGrid[coarseIdx] = sum;
}
__global__ void ProlongAndCorrect(float* __restrict__ fineSolution, const float* __restrict__ coarseError, 
                                  const size_t fNx, const size_t fNy, const size_t fNz, 
                                  const size_t cNx, const size_t cNy, const size_t cNz){
    // Map thread to Fine Destination Grid Coordinates
    size_t fx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t fy = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z  = blockIdx.z * blockDim.z + threadIdx.z;
    // Boundary protection skips physical edge faces
    if (fx <= 0 || fx >= fNx - 1 || fy <= 0 || fy >= fNy - 1 || z <= 0 || z >= fNz - 1) return;
    // Identify the base bounding coarse cell index (integer division)
    int cx0 = (int)(fx / 2);
    int cy0 = (int)(fy / 2);
    int cz0 = (int)(z / 2);
    // Identify the upper bounding coarse cell index
    int cx1 = min(cx0 + 1, (int)cNx - 1);
    int cy1 = min(cy0 + 1, (int)cNy - 1);
    int cz1 = min(cz0 + 1, (int)cNz - 1);
    // Compute structural linear interpolation weights (0.0 or 0.5)
    float tx = (fx & 1) * 0.5f;
    float ty = (fy & 1) * 0.5f;
    float tz = (z  & 1) * 0.5f;
    float o_tx = 1.0f - tx;
    float o_ty = 1.0f - ty;
    float o_tz = 1.0f - tz;
    // Retrieve the 8 surrounding coarse cell corner error values
    float e000 = coarseError[getLinearIdx3D(cx0, cy0, cz0, cNx, cNy)];
    float e100 = coarseError[getLinearIdx3D(cx1, cy0, cz0, cNx, cNy)];
    float e010 = coarseError[getLinearIdx3D(cx0, cy1, cz0, cNx, cNy)];
    float e110 = coarseError[getLinearIdx3D(cx1, cy1, cz0, cNx, cNy)];
    float e001 = coarseError[getLinearIdx3D(cx0, cy0, cz1, cNx, cNy)];
    float e101 = coarseError[getLinearIdx3D(cx1, cy0, cz1, cNx, cNy)];
    float e011 = coarseError[getLinearIdx3D(cx0, cy1, cz1, cNx, cNy)];
    float e111 = coarseError[getLinearIdx3D(cx1, cy1, cz1, cNx, cNy)];
    // Perform standard 3D trilinear interpolation
    float interpolated_error = o_tz * (o_ty * (o_tx * e000 + tx * e100) + ty * (o_tx * e010 + tx * e110)) +
           tz * (o_ty * (o_tx * e001 + tx * e101) + ty * (o_tx * e011 + tx * e111));
    // Apply error correction directly to fine target vector
    size_t fineIdx = getLinearIdx3D((int)fx, (int)fy, (int)z, fNx, fNy);
    fineSolution[fineIdx] += interpolated_error;
}
void v_cycle_hierarchical(std::vector<float*>& d_u_hierarchy, // Solution vectors per level
                          std::vector<float*>& d_f_hierarchy, // RHS vectors per level
                          std::vector<float*>& d_r_hierarchy, // Residual vectors per level
                          const std::vector<size_t>& Nx_lvl, const std::vector<size_t>& Ny_lvl, const std::vector<size_t>& Nz_lvl, // Track dimensions per level
                          int current_lvl, int total_levels, 
                          float h, float h2, 
                          int nu1 = 2, int nu2 = 2){
    size_t Nx = Nx_lvl[current_lvl];
    size_t Ny = Ny_lvl[current_lvl];
    size_t Nz = Nz_lvl[current_lvl];
    // Block sizes for called kernels
    dim3 block(32, 8, 1);
    dim3 grid((Nx + block.x - 1) / block.x, (Ny + block.y - 1) / block.y, (Nz + block.z - 1) / block.z);
    
    // Fetch active level pointers
    float* d_u = d_u_hierarchy[current_lvl];
    float* d_f = d_f_hierarchy[current_lvl];
    float* d_r = d_r_hierarchy[current_lvl];
    
    // -------------------------------------------------------------------------
    // PRE-SMOOTHING (Execute nu1 relaxation steps on current fine level)
    // -------------------------------------------------------------------------
    for (int i = 0; i < nu1; ++i) {
        red_black_gauss_siedel_red<<<grid, block>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
        red_black_gauss_siedel_black<<<grid, block>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
    }
    
    // -------------------------------------------------------------------------
    // CORNERSTONE LEVEL CHECK (If at base level 33^3, solve directly or stop)
    // -------------------------------------------------------------------------
    if (current_lvl == total_levels - 1) {
        // Coarsest level (e.g., 33^3): Smooth aggressively or use a direct solver
        for (int i = 0; i < 10; ++i) {
            red_black_gauss_siedel_red<<<grid, block>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
            red_black_gauss_siedel_black<<<grid, block>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
        }
        return; // Start traveling back up the V-cycle
    }
    
    // -------------------------------------------------------------------------
    // COMPUTE RESIDUAL & COARSEN DOWNWARD
    // -------------------------------------------------------------------------
    // Calculate residual on current level: r = f - A*u
        Residual_function<<<grid, block>>>(d_f, d_r, d_u, Nx, Ny, Nz, h, h2);

    size_t cNx = Nx_lvl[current_lvl + 1];
    size_t cNy = Ny_lvl[current_lvl + 1];
    size_t cNz = Nz_lvl[current_lvl + 1];
    // Define coarse grid block size
    dim3 coarse_grid((cNx + block.x - 1) / block.x, 
                     (cNy + block.y - 1) / block.y, 
                     (cNz + block.z - 1) / block.z);
    float* d_f_coarse = d_f_hierarchy[current_lvl + 1];
    float* d_u_coarse = d_u_hierarchy[current_lvl + 1];

    // Zero out the next coarse solution guess before restriction
    size_t coarse_bytes = cNx * cNy * cNz * sizeof(float);
    cudaMemset(d_u_coarse, 0, coarse_bytes);

    // Coarsen/Restrict the current residual into the next level's RHS vector
    Coarsen<<<coarse_grid, block>>>(d_r, d_f_coarse, cNx, cNy, cNz, Nx, Ny, Nz);

    // Calculate adjusted spatial step parameters for the coarser grid spacing
    float h_coarse = h * 2.0f;
    float h2_coarse = h_coarse * h_coarse;

    // -------------------------------------------------------------------------
    // RECURSIVE RECONSTRUCTION (Call V-cycle for the next level down)
    // -------------------------------------------------------------------------
    v_cycle_hierarchical(d_u_hierarchy, d_f_hierarchy, d_r_hierarchy, 
                         Nx_lvl, Ny_lvl, Nz_lvl, 
                         current_lvl + 1, total_levels, 
                         h_coarse, h2_coarse, nu1, nu2);

    // -------------------------------------------------------------------------
    // PROLONGATION & ERROR CORRECTION (Bring data back up)
    // -------------------------------------------------------------------------
    // Target grid configuration matches the current active fine level dimensions.
    // This injects the interpolated coarse error directly back into d_u.
    ProlongAndCorrect<<<grid, block>>>(
        d_u, d_u_coarse, 
        Nx, Ny, Nz,     // Fine target dimensions
        cNx, cNy, cNz   // Coarse source dimensions
    );

    // -------------------------------------------------------------------------
    // POST-SMOOTHING (Execute nu2 relaxation steps on current fine level)
    // -------------------------------------------------------------------------
    for (int i = 0; i < nu2; ++i) {
        red_black_gauss_siedel_red<<<grid, block>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
        red_black_gauss_siedel_black<<<grid, block>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
    }
}
int main() {
    // =========================================================================
    // BASE VARIABLES
    // =========================================================================
    const size_t N = 1025; 
    const size_t Nx = N, Ny = Nx, Nz = Nx;
    constexpr float L = 1.0f; 
    const float h = L / (Nx - 1);
    const float h2 = h * h;

    // =========================================================================
    // MULTIGRID HIERARCHY TRACKING SETUP
    // =========================================================================
    const size_t target_coarse_limit = 33;
    std::vector<float*> d_u_hierarchy; // Solution / Error Correction Pyramid
    std::vector<float*> d_f_hierarchy; // RHS Source Pyramid
    std::vector<float*> d_r_hierarchy; // Residual Pyramid
    
    std::vector<size_t> Nx_lvl, Ny_lvl, Nz_lvl;

    // Initialize the finest level dimensions using the main variables
    Nx_lvl.push_back(Nx);
    Ny_lvl.push_back(Ny);
    Nz_lvl.push_back(Nz);

    std::cout << "--- Initializing Multigrid Allocation Hierarchy ---" << std::endl;
    size_t current_N = N;
    int lvl = 0;
    nvtxMarkA("Starting level building");
    // Dynamically build levels until we hit our base 33^3 grid
    while (true) {
        size_t total_elements = Nx_lvl[lvl] * Ny_lvl[lvl] * Nz_lvl[lvl];
        size_t total_bytes = total_elements * sizeof(float);

        float *d_u_ptr = nullptr, *d_f_ptr = nullptr, *d_r_ptr = nullptr;
        
        // Allocate corresponding pointers for this layer tier
        CUDA_CHECK(cudaMalloc(&d_u_ptr, total_bytes));
        CUDA_CHECK(cudaMalloc(&d_f_ptr, total_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_ptr, total_bytes));

        // Zero out arrays (ensures 0-padding conditions on outer boundaries)
        CUDA_CHECK(cudaMemset(d_u_ptr, 0, total_bytes));
        CUDA_CHECK(cudaMemset(d_f_ptr, 0, total_bytes));
        CUDA_CHECK(cudaMemset(d_r_ptr, 0, total_bytes));

        d_u_hierarchy.push_back(d_u_ptr);
        d_f_hierarchy.push_back(d_f_ptr);
        d_r_hierarchy.push_back(d_r_ptr);

        // Print footprint metrics
        double mb_allocated_per_array = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        std::cout << "Level " << lvl << " Dimensions: " << Nx_lvl[lvl] << "x" << Ny_lvl[lvl] << "x" << Nz_lvl[lvl] 
                  << " | Total Tier Footprint: " << (mb_allocated_per_array * 3.0) << " MB" << std::endl;

        if (current_N <= target_coarse_limit) { break; }

        // Calculate next level dimensions using the vertex centered formula
        current_N = ((current_N - 1) / 2) + 1;
        Nx_lvl.push_back(current_N);
        Ny_lvl.push_back(current_N);
        Nz_lvl.push_back(current_N);
        lvl++;
    }
    nvtxMarkA("Level building complete");

    const int num_levels = d_u_hierarchy.size();
    std::cout << "Total Hierarchy Levels Generated: " << num_levels << "\n" << std::endl;

    // =========================================================================
    // PROBLEM INITIALIZATION (Manufactured Solution Setup)
    // =========================================================================
    nvtxMarkA("PROBLEM INITIALIZATION START");
    const size_t finest_elements = Nx_lvl[0] * Ny_lvl[0] * Nz_lvl[0];
    const size_t finest_bytes    = finest_elements * sizeof(float);

    // Allocate host and device memory for error tracking validation
    float* d_u_true = nullptr;
    float* d_err_sq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_true, finest_bytes));
    CUDA_CHECK(cudaMalloc(&d_err_sq, finest_bytes));

    std::vector<float> h_err_sq(finest_elements);

    // Configure 3D execution configurations targeting Level 0
    dim3 init_block(32, 8, 4);
    dim3 init_grid(
        (Nx_lvl[0] + init_block.x - 1) / init_block.x,
        (Ny_lvl[0] + init_block.y - 1) / init_block.y,
        (Nz_lvl[0] + init_block.z - 1) / init_block.z
    );

    std::cout << "--- Populating Problem Initial Fields via Manufactured Solution ---" << std::endl;
    InitializeProblem<<<init_grid, init_block>>>(d_f_hierarchy[0], d_u_true, Nx_lvl[0], Ny_lvl[0], Nz_lvl[0], h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxMarkA("PROBLEM INITIALIZATION END");
    // =========================================================================
    // EXECUTION LOOP: Launch Multigrid V-Cycles Until Convergence
    // =========================================================================
    std::cout << "--- Executing Multigrid Solver ---\n" << std::endl;
    
    nvtxMarkA("Execution loop Start");
    int max_v_cycles = 5; 
    for (int cycle = 0; cycle < max_v_cycles; ++cycle) {
        
        // Trigger the recursive multigrid engine starting explicitly at Level 0
        v_cycle_hierarchical(
            d_u_hierarchy, d_f_hierarchy, d_r_hierarchy,
            Nx_lvl, Ny_lvl, Nz_lvl,
            0,            // Start level 0 (finest)
            num_levels,   // Total levels tracked
            h, h2,        // Top-level spatial parameters
            2, 2          // nu1 (pre-smooth), nu2 (post-smooth)
        );
        
        // Monitor for asynchronous execution errors or failures
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate L2 Error Norm for convergence verification
        size_t threadsPerBlock = 256;
        size_t blocksPerGrid = (finest_elements + threadsPerBlock - 1) / threadsPerBlock;
        ComputeSquaredError<<<blocksPerGrid, threadsPerBlock>>>(d_u_hierarchy[0], d_u_true, d_err_sq, finest_elements);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy back to host to calculate sum
        CUDA_CHECK(cudaMemcpy(h_err_sq.data(), d_err_sq, finest_bytes, cudaMemcpyDeviceToHost));
        
        double absolute_error_sum = 0.0;
        for (size_t n = 0; n < finest_elements; ++n) {
            absolute_error_sum += h_err_sq[n];
        }
        double l2_norm = std::sqrt(absolute_error_sum / finest_elements);
        
        std::cout << "V-Cycle " << cycle + 1 << " Complete | RMS Error Norm vs Analytical Solution: " << l2_norm << std::endl;
        nvtxMarkA("Execution loop Iteration");
    }

    std::cout << "\n--- Finalizing Verification Cleanup Phases ---" << std::endl;
    nvtxMarkA("Execution loop End");
    // =========================================================================
    // MEMORY CLEANUP LIFECYCLE
    // =========================================================================
    nvtxMarkA("MEMORY CLEANUP Start");
    CUDA_CHECK(cudaFree(d_u_true));
    CUDA_CHECK(cudaFree(d_err_sq));

    for (int l = 0; l < num_levels; l++) {
        CUDA_CHECK(cudaFree(d_u_hierarchy[l]));
        CUDA_CHECK(cudaFree(d_f_hierarchy[l]));
        CUDA_CHECK(cudaFree(d_r_hierarchy[l]));
    }
    nvtxMarkA("MEMORY CLEANUP End");

    std::cout << "Execution completed successfully!" << std::endl;
    return 0;
}