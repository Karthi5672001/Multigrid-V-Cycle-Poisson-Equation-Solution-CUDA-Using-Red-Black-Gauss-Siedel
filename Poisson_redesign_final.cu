#ifdef _MSC_VER
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Compilation in Docker WSL in powershell:
cd C:\Users\Karthi\CUDA
docker build -t cuda133-cpp-dev .
docker run --gpus all --cap-add=SYS_ADMIN -it --rm -v "${PWD}:/workspace" nvidia/cuda:13.3.0-devel-ubuntu22.04 bash
docker run --gpus all --cap-add=SYS_ADMIN -it --rm -v ${PWD}:/workspace nvidia/cuda:13.3.0-devel-ubuntu22.04 bash
nvcc --version
cd workspace
nvcc -O3 --use_fast_math --generate-code arch=compute_120,code=sm_120 --ptxas-options=-v Poisson_redesign_final.cu -o Poisson_redesign_final
nvcc -O3 --generate-code arch=compute_120,code=sm_120 --ptxas-options=-v Poisson_redesign_final.cu -o Poisson_redesign_final
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --force-overwrite true -o Poisson_redesign_final ./Poisson_redesign_final
ncu --metrics sm__sol_pct,dram__sol_pct ./Poisson_redesign_final
ncu --set full -o Poisson_Redesign_opt ./Poisson_redesign_final
--------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#include <cmath>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <sstream> 
#include <iomanip>
#include <chrono>
#include <nvtx3/nvtx3.hpp>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#define BLOCK_X 16
#define BLOCK_Y 8
#define BLOCK_Z 4
/*--------------------------------------------------------------------------------------------------------------------------------------------------------
    CUDA error Macro - handle runtime errors cleanly.
    do {...} while(0) - for safety, preventing macro expansion bugs from if-else blocks, and thus acts like a single statement.
    __LINE__ - automatically prints the exact line number in the source file where the CUDA call failed, saving time during debugging.
    cudaGetErrorString() - Converts the numeric error code into readable message like cudaErrorIllegalAddress.
    exit(EXIT_FAILURE) - stops the program instantly.
--------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/*--------------------------------------------------------------------------------------------------------------------------------------------------------
    Inline helper to compute absolute distance sum clearly.
    Computes the Manhattan distance (L₁ norm) sum along a single dimension while adding pre-calculated absolute differences.
    __forceinline__ - ensures the compiler embeds the logic directly into the calling thread's execution path, eliminating.
    Function call overhead in performance-critical GPU kernels.
--------------------------------------------------------------------------------------------------------------------------------------------------------*/
__device__ __forceinline__ int abs_dx_calc(int dx, int abs_dy, int abs_dz) {
    return abs(dx) + abs_dy + abs_dz;
}

/*--------------------------------------------------------------------------------------------------------------------------------------------------------
    Helper: linear index from (x,y,z).
    Gives the 1D index from 3D grid so that the program can work with a flat array that is beneficial to GPU computing.
    Explicit casting to size_t for the prevention of 32 bit integer overflow.
--------------------------------------------------------------------------------------------------------------------------------------------------------*/
__device__ __inline__ size_t getLinearIdx3D(unsigned int x, unsigned int y, unsigned int z, 
                                            const size_t Nx, const size_t Ny) {
    return (size_t)z * Nx * Ny + (size_t)y * Nx + x;
}

/*--------------------------------------------------------------------------------------------------------------------------------------------------------
    Custom squaring difference functor compatible with Thrust zip iterators.
--------------------------------------------------------------------------------------------------------------------------------------------------------*/
struct DifferenceSquared {
    __host__ __device__ __forceinline__
    double operator()(const thrust::tuple<float, float>& t) const {
        // Unpack the paired values safely from the tuple
        const double a = static_cast<double>(thrust::get<0>(t));
        const double b = static_cast<double>(thrust::get<1>(t));
        
        // Cast the difference to double before squaring to prevent loss of precision during massive (>1 billion elements) reduction accumulations.
        const double diff = a - b;
        return diff * diff;
    }
};

/*-------------------------------------------------------------------------------------------------------------------------------------------
    Initialization Kernel
    This converts discrete, integer-based GPU thread coordinates (x, y, z) into continuous physical spatial coordinates (px, py, pz),
    using the grid spacing factor h (\(\Delta x, \Delta y, \Delta z\)).
    Sets up Manufactured Problem on Level 0.
    ALSO zeroes the solution u (initial guess 0 + Dirichlet shell 0).
    in the same pass, so no separate cudaMemset is needed for d_u[0].
    __restrict__ - marking pointers as restricted, so compiler knows that f, u, u_true.
    do not overlap in memory allowing for optimising memory usage and maximise cache efficiency.
    Since consecutive threads handle sequential x values, the writes to u_true[i], f[i], and u[i] are perfectly coalesced across the warp. 
    The memory controller groups these writes into high-bandwidth transactions.
-------------------------------------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------------
__global__ void InitializeProblem(float* __restrict__ f,
                                  float* __restrict__ u, float* __restrict__ u_true,
                                  const size_t Nx, const size_t Ny, const size_t Nz,
                                  const float h) {
    // Test Solution 1: Triple Sine Solution (Dirichlet Boundary Conditions)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Prevents out-of-bounds memory accesses when the total number of threads 
    // In the grid does not perfectly match the domain dimensions
    if (x >= Nx || y >= Ny || z >= Nz) return;
                
    size_t i = getLinearIdx3D(x, y, z, Nx, Ny);
    // Declaring \(\pi \) as a constexpr forces compile-time evaluation.
    const float px = x * h;
    const float py = y * h;
    const float pz = z * h;

    constexpr float PI = 3.141592653589793f;
    
    float u_true_i = sinf(PI * px) * sinf(PI * py) * sinf(PI * pz);

    // Because calculating trigonometric functions (sinf) consumes many clock cycles on a GPU's Special Function Units (SFU), 
    // Calculating u_true_i exactly once and reusing it to populate both u_true and f saves significant instruction overhead.
    u_true[i] = u_true_i;
    f[i]      = 3.0f * PI * PI * u_true_i;
    // initial guess = 0, and sets the Dirichlet boundary to 0
    u[i]      = 0.0f;   
}
__global__ void InitializeProblem(float* __restrict__ f,
                                  float* __restrict__ u, float* __restrict__ u_true,
                                  const size_t Nx, const size_t Ny, const size_t Nz,
                                  const float h) {
    // Test Solution 3: Mixed High-Frequency Trigonometric Field
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Prevents out-of-bounds memory accesses when the total number of threads 
    // In the grid does not perfectly match the domain dimensions
    if (x >= Nx || y >= Ny || z >= Nz) return;
                
    size_t i = getLinearIdx3D(x, y, z, Nx, Ny);

    const float px = x * h;
    const float py = y * h;
    const float pz = z * h;

    // Define pi using standard single-precision float accuracy
    const float pi = 3.14159265f;

    // Mixed High-Frequency Trigonometric Field
    // Re-use common sub-expressions to optimize ALU pipeline throughput
    float sin_2pi_x = sinf(2.0f * pi * px);
    float cos_pi_y  = cosf(pi * py);
    float sin_3pi_z = sinf(3.0f * pi * pz);

    // Exact Solution: u(x,y,z) = sin(2*pi*x) * cos(pi*y) * sin(3*pi*z)
    float u_true_i = sin_2pi_x * cos_pi_y * sin_3pi_z;
    
    u_true[i] = u_true_i;
    
    // Forcing Function: f(x,y,z) = -14 * pi^2 * sin(2*pi*x) * cos(pi*y) * sin(3*pi*z)
    f[i] = 14.0f * (pi * pi) * u_true_i;

    // Evaluate u_exact directly at the boundaries to enforce Dirichlet boundary conditions
    bool on_boundary = (x == 0 || x == Nx - 1 || 
                        y == 0 || y == Ny - 1 || 
                        z == 0 || z == Nz - 1);
                        
    if (on_boundary) {
        // Handled cleanly via math: 
        // u_true_i naturally evaluates to 0.0f at x=0, x=1, z=0, z=1 
        // and evaluates to the correct analytical values at y=0 and y=1.
        u[i] = u_true_i;   // Sets the non-zero Dirichlet boundary condition
    } else {
        u[i] = 0.0f;       // initial guess = 0 for interior elements
    }
}
__global__ void InitializeProblem(float* __restrict__ f,
                                  float* __restrict__ u, float* __restrict__ u_true,
                                  const size_t Nx, const size_t Ny, const size_t Nz,
                                  const float h) {
    // Test Solution 4: Asymmetric Exponential Solution
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Prevents out-of-bounds memory accesses when the total number of threads 
    // In the grid does not perfectly match the domain dimensions
    if (x >= Nx || y >= Ny || z >= Nz) return;
                
    size_t i = getLinearIdx3D(x, y, z, Nx, Ny);

    const float px = x * h;
    const float py = y * h;
    const float pz = z * h;

    // Asymmetric Exponential Solution: u(x,y,z) = e^(x + 2y + 3z)
    // Compute the sum of exponents first to optimize ALU pipeline throughput
    float u_true_i = expf(px + 2.0f * py + 3.0f * pz);
    
    u_true[i] = u_true_i;
    
    // Forcing Function: f(x,y,z) = 14 * e^(x + 2y + 3z)
    // Factored directly by scaling the exact solution to minimize math overhead
    f[i]      = 14.0f * u_true_i;

    // Evaluate u_exact directly at the boundaries to enforce non-zero Dirichlet boundary conditions
    bool on_boundary = (x == 0 || x == Nx - 1 || 
                        y == 0 || y == Ny - 1 || 
                        z == 0 || z == Nz - 1);
                        
    if (on_boundary) {
        // Sets the non-zero asymmetric Dirichlet boundary conditions
        u[i] = u_true_i;   
    } else {
        // Initial guess = 0 for interior elements
        u[i] = 0.0f;       
    }
}
------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void InitializeProblem(float* __restrict__ f,
                                  float* __restrict__ u, float* __restrict__ u_true,
                                  const size_t Nx, const size_t Ny, const size_t Nz,
                                  const float h) {

    // Test Solution 2: Polynomial Solution (Non-Zero Boundary Conditions)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Prevents out-of-bounds memory accesses when the total number of threads 
    // In the grid does not perfectly match the domain dimensions
    if (x >= Nx || y >= Ny || z >= Nz) return;
                
    size_t i = getLinearIdx3D(x, y, z, Nx, Ny);

    const float px = x * h;
    const float py = y * h;
    const float pz = z * h;

    // Exact Polynomial Solution: u(x,y,z) = x^3 + y^3 + z^3
    // Direct multiplications are used instead of powf to maximize ALU pipeline throughput
    float u_true_i = (px * px * px) + (py * py * py) + (pz * pz * pz);
    
    u_true[i] = u_true_i;
    
    // Forcing Function: f(x,y,z) = 6x + 6y + 6z
    f[i]      = -6.0f * (px + py + pz);

    // Evaluate u_exact directly at the boundaries to enforce non-zero Dirichlet boundary conditions
    bool on_boundary = (x == 0 || x == Nx - 1 || 
                        y == 0 || y == Ny - 1 || 
                        z == 0 || z == Nz - 1);
                        
    if (on_boundary) {
        // Sets the non-zero Dirichlet boundary condition
        u[i] = u_true_i;   
    } else {
        // Initial guess = 0 for interior elements
        u[i] = 0.0f;
    }
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Alternative ZeroBoundary Kernel that does one 2D face at a time.
    By using the __all_sync hardware primitive, the GPU checks if all 32 threads in a warp are safely inside the core. If they are, the entire warp exits simultaneously at instruction 3
    Horner’s factorization (((size_t)z * Ny + y) * Nx + x): This strips away an entire 64-bit multiplication step, reducing register usage per thread so that the GPU can hold more active warps in flight to hide memory latency.
    Reduced register footprint due to removing the explicit creation of unnecessary intermediate boolean evaluations and casting dimensions smoothly, register pressure is kept low, meaning higher warp occupancy on streaming multiprocessors.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void ZeroBoundary3D(float* __restrict__ a,
                               const size_t Nx, const size_t Ny, const size_t Nz) {
    // ==================================================================================================================================================
    // 1. Calculate thread coordinates
    // ==================================================================================================================================================
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    // ==================================================================================================================================================
    // 2. Immediate global out-of-bounds guard
    // ==================================================================================================================================================
    if (x >= Nx || y >= Ny || z >= Nz) return;
    // ==================================================================================================================================================
    // 3. Early Exit for interior warps (The Performance Saver)
    // ==================================================================================================================================================
    // If the entire warp is safely inside the interior, drop out instantly.
    // This stops thousands of interior warps from wasting pipeline slots.
    const bool is_interior_point = (x > 0 && x < Nx - 1 && 
                                    y > 0 && y < Ny - 1 && 
                                    z > 0 && z < Nz - 1);
                                    
    // __all_sync returns true only if ALL 32 threads in the warp are interior points
    if (__all_sync(0xFFFFFFFF, is_interior_point)) return;
    // ==================================================================================================================================================
    // 4. Boundary verification for the remaining active border warps
    // ==================================================================================================================================================
    const bool onShell = (x == 0 || y == 0 || z == 0 ||
                          x == Nx - 1 || y == Ny - 1 || z == Nz - 1);
    if (!onShell) return;
    // ==================================================================================================================================================
    // 5. Native 32-bit arithmetic optimization inside the linear index calculation
    // ==================================================================================================================================================
    const size_t linear_idx = ((size_t)z * Ny + y) * Nx + x;
    a[linear_idx] = 0.0f;
}

/*------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Red-Black Gauss-Siedel Kernel (Fused)
    Allows a much easier parallel apporach for GPUs, as we can calculate based on adjacent elements of a point
    freely as none of either red or black depend on others of the same colour.
    __restrict__ - marking pointers as restricted, so compiler knows that f, u
    __shared__ float s_u[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2]; - 3D Padded Shared Memory Array. Creates a local cache for the thread block, 
    +2 padding on the stencil to look at immediate neighbors, so the threads sitting on the very outer edges of the thread block need access to data outside their block boundary.
    This extra padding allocates space for halo (ghost) cells so that neighbor evaluations don't trigger out-of-bounds errors in shared memory.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void multigrid_red_black_fused(const float * __restrict__ f, 
                                          float * __restrict__ u,
                                          const int Nx, const int Ny, const int Nz, 
                                          const float h2) {
    __shared__ float s_u[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int z = blockIdx.z * blockDim.z + tz;

    const int stx = tx + 1;
    const int sty = ty + 1;
    const int stz = tz + 1;

    const size_t sliceSize = (size_t)Nx * Ny;
    const size_t i = (size_t)z * sliceSize + (size_t)y * Nx + x;
    // Determines the colour of the current 3D spatial coordinate using a bitwise AND operation. 
    // No two adjacent cells are the same colour.
    const int is_black = (x + y + z) & 1; 

    if (x < Nx && y < Ny && z < Nz) {
        s_u[stz][sty][stx] = u[i];
    } else {
        s_u[stz][sty][stx] = 0.0f;
    }
    // Ghost Cell Loading
    // Threads located exactly on the border of a thread block (tx == 0, ty == 0, etc.), In addition to loading their own data, 
    // they reach out into global memory to fetch the data of neighboring blocks and store it in the padded shared memory shell.
    if (tx == 0 && (x - 1) >= 0)           s_u[stz][sty][stx - 1] = u[i - 1];
    if (tx == BLOCK_X - 1 && (x + 1) < Nx) s_u[stz][sty][stx + 1] = u[i + 1];
    if (ty == 0 && (y - 1) >= 0)           s_u[stz][sty - 1][stx] = u[i - Nx];
    if (ty == BLOCK_Y - 1 && (y + 1) < Ny) s_u[stz][sty + 1][stx] = u[i + Nx];
    if (tz == 0 && (z - 1) >= 0)           s_u[stz - 1][sty][stx] = u[i - sliceSize];
    if (tz == BLOCK_Z - 1 && (z + 1) < Nz) s_u[stz + 1][sty][stx] = u[i + sliceSize];
    // Synchronization barriers to coordinate the mathematical dependency of the solver.
    // Guarantees that all core data and all halo cells are completely written to shared memory before any math takes place.
    __syncthreads();

    const bool is_interior = (x > 0 && x < Nx - 1 && y > 0 && y < Ny - 1 && z > 0 && z < Nz - 1);
    // 7-Point Stencil & Fused Multiply-Add for red cells.
    // Evaluates a standard 3D Laplacian 7-point stencil (the center point plus its 6 orthogonal neighbors)
    if (is_interior && !is_black) {
        const float sum = s_u[stz][sty][stx - 1] + s_u[stz][sty][stx + 1] +
                          s_u[stz][sty - 1][stx] + s_u[stz][sty + 1][stx] +
                          s_u[stz - 1][sty][stx] + s_u[stz + 1][sty][stx];
        
        s_u[stz][sty][stx] = (sum + h2 * f[i]) * (1.0f / 6.0f);
        u[i] = s_u[stz][sty][stx];
    }
    // Synchronization barriers to coordinate the mathematical dependency of the solver between the Red update and the Black update. 
    // This forces the Black update phase to wait until the Red phase has finished writing its fresh results into s_u. 
    // Without this, threads calculating Black cells would read stale data.
    __syncthreads();
    // 7-Point Stencil & Fused Multiply-Add for black cells.
    // Evaluates a standard 3D Laplacian 7-point stencil (the center point plus its 6 orthogonal neighbors)
    if (is_interior && is_black) {
        const float sum = s_u[stz][sty][stx - 1] + s_u[stz][sty][stx + 1] +
                          s_u[stz][sty - 1][stx] + s_u[stz][sty + 1][stx] +
                          s_u[stz - 1][sty][stx] + s_u[stz + 1][sty][stx];
        
        u[i] = (sum + h2 * f[i]) * (1.0f / 6.0f);
    }
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Residual Kernel
    Threads on the boundary planes drop out immediately. 
    They perform zero global memory reads or writes, keeping memory traffic low.
    Replacing division by h2 with multiplication by invH2 saves significant clock cycles per thread,
    since floating-point division is one of the slowest basic operations on a GPU.
    Because a thread block processes adjacent elements, loading the 6 neighboring elements through __ldg() allows threads in the same warp to share data through 
    the L1 cache seamlessly rather than pulling duplicates from high-latency global memory
----------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
__global__ void Residual_function(const float * __restrict__ f, 
                                  float * __restrict__ r, 
                                  const float * __restrict__ u,
                                  const size_t Nx, const size_t Ny, const size_t Nz,  
                                  const float h2) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    // ==================================================================================================================================================
    // 1. Early-exit boundary guard
    // ==================================================================================================================================================
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    // ==================================================================================================================================================
    // 2. Pre-calculate layout parameters and strides
    // ==================================================================================================================================================
    const size_t sliceSize = Nx * Ny;
    const size_t i = z * sliceSize + y * Nx + x;
    const float invH2 = 1.0f / h2;
    // ==================================================================================================================================================
    // 3. Load center variables and neighboring stencil points via L1 cache (__ldg)
    // ==================================================================================================================================================
    const float u_c = __ldg(&u[i]);
    const float sum = __ldg(&u[i - 1])         + __ldg(&u[i + 1]) + 
                      __ldg(&u[i - Nx])        + __ldg(&u[i + Nx]) + 
                      __ldg(&u[i - sliceSize]) + __ldg(&u[i + sliceSize]);
    // ==================================================================================================================================================
    // 4. Match exact mathematical formulation using fast multiplication instead of division
    // ==================================================================================================================================================
    const float Au = (6.0f * u_c - sum) * invH2;
    // ==================================================================================================================================================
    // 5. Stream the final residual value back to global memory
    // ==================================================================================================================================================
    r[i] = __ldg(&f[i]) - Au;
}

/*----------------------------------------------------------------------------------------------------------------------------
    Coarsen Kernel
    We take a branch separated optimisation approach, since Over 90% of the active points inside 
    the multigrid domain reside entirely in the interior. 
    Branching into an explicit is_interior code path eliminates 81 instances of min() and max() 
    operations for almost every thread warp.
    The if-else branch logic used to look up weights has been entirely replaced with precomputed 
    numeric constants compiled straight into the pipeline.
    __ldg() L1 Cache Alignment: Because adjacent coarse points read heavily overlapping regions of 
    the fine grid array, routing access through __ldg() turns global memory lookups into instant L1 hits.
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void Coarsen(const float* __restrict__ fineGrid, 
                        float* __restrict__ coarseGrid,
                        const size_t Nx, const size_t Ny, const size_t Nz, 
                        const size_t fNx, const size_t fNy, const size_t fNz) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // out of bounds check
    if (x >= Nx || y >= Ny || z >= Nz) return;
    // ==================================================================================================================================================
    // 1. Convert to fine-grid anchors (using fast bitwise shift)
    // ==================================================================================================================================================
    const int fx = (int)(x << 1);
    const int fy = (int)(y << 1);
    const int fz = (int)(z << 1);
    // ==================================================================================================================================================
    // 2. Pre-calculate structural offsets to prevent 3D index calculations inside loops
    // ==================================================================================================================================================
    const size_t fine_slice = fNx * fNy;
    // ==================================================================================================================================================
    // 3. Separate boundary threads from internal threads to remove 'min/max' overhead entirely
    // ==================================================================================================================================================
    const bool is_interior = (fx > 0 && fx < (int)fNx - 1 && 
                              fy > 0 && fy < (int)fNy - 1 && 
                              fz > 0 && fz < (int)fNz - 1);

    float sum = 0.0f;

    if (is_interior) {
        // --- HIGH PERFORMANCE PATH: NO CLIPPING, UNROLLED READ-ONLY CACHE ---
        // Stencil Weights: Center (1/8), Face (1/16), Edge (1/32), Corner (1/64)
        constexpr float w0 = 1.0f / 8.0f;
        constexpr float w1 = 1.0f / 16.0f;
        constexpr float w2 = 1.0f / 32.0f;
        constexpr float w3 = 1.0f / 64.0f;

        // Base pointer matching the exact center cell of the fine grid block
        const size_t base_idx = fz * fine_slice + fy * fNx + fx;

        // Z = -1 plane
        sum += w3 * __ldg(&fineGrid[base_idx - fine_slice - fNx - 1]);
        sum += w2 * __ldg(&fineGrid[base_idx - fine_slice - fNx]);
        sum += w3 * __ldg(&fineGrid[base_idx - fine_slice - fNx + 1]);
        sum += w2 * __ldg(&fineGrid[base_idx - fine_slice - 1]);
        sum += w1 * __ldg(&fineGrid[base_idx - fine_slice]);
        sum += w2 * __ldg(&fineGrid[base_idx - fine_slice + 1]);
        sum += w3 * __ldg(&fineGrid[base_idx - fine_slice + fNx - 1]);
        sum += w2 * __ldg(&fineGrid[base_idx - fine_slice + fNx]);
        sum += w3 * __ldg(&fineGrid[base_idx - fine_slice + fNx + 1]);

        // Z = 0 plane
        sum += w2 * __ldg(&fineGrid[base_idx - fNx - 1]);
        sum += w1 * __ldg(&fineGrid[base_idx - fNx]);
        sum += w2 * __ldg(&fineGrid[base_idx - fNx + 1]);
        sum += w1 * __ldg(&fineGrid[base_idx - 1]);
        sum += w0 * __ldg(&fineGrid[base_idx]);
        sum += w1 * __ldg(&fineGrid[base_idx + 1]);
        sum += w2 * __ldg(&fineGrid[base_idx + fNx - 1]);
        sum += w1 * __ldg(&fineGrid[base_idx + fNx]);
        sum += w2 * __ldg(&fineGrid[base_idx + fNx + 1]);

        // Z = +1 plane
        sum += w3 * __ldg(&fineGrid[base_idx + fine_slice - fNx - 1]);
        sum += w2 * __ldg(&fineGrid[base_idx + fine_slice - fNx]);
        sum += w3 * __ldg(&fineGrid[base_idx + fine_slice - fNx + 1]);
        sum += w2 * __ldg(&fineGrid[base_idx + fine_slice - 1]);
        sum += w1 * __ldg(&fineGrid[base_idx + fine_slice]);
        sum += w2 * __ldg(&fineGrid[base_idx + fine_slice + 1]);
        sum += w3 * __ldg(&fineGrid[base_idx + fine_slice + fNx - 1]);
        sum += w2 * __ldg(&fineGrid[base_idx + fine_slice + fNx]);
        sum += w3 * __ldg(&fineGrid[base_idx + fine_slice + fNx + 1]);
    } 
    else {
        // --- SAFE FALLBACK PATH: BOUNDARY ONLY (MATHEMATICALLY TRUNCATED) ---
        // Dynamically drops out-of-bounds contributions and normalizes valid weights
        float total_weight = 0.0f;

        #pragma unroll
        for (int dz = -1; dz <= 1; dz++) {
            int target_fz = fz + dz;
            if (target_fz < 0 || target_fz >= (int)fNz) continue; // Skip physical exterior
            size_t z_offset = (size_t)target_fz * fine_slice;
            int abs_dz = abs(dz);

            #pragma unroll
            for (int dy = -1; dy <= 1; dy++) {
                int target_fy = fy + dy;
                if (target_fy < 0 || target_fy >= (int)fNy) continue; // Skip physical exterior
                size_t y_offset = (size_t)target_fy * fNx;
                int abs_dy = abs(dy);

                #pragma unroll
                for (int dx = -1; dx <= 1; dx++) {
                    int target_fx = fx + dx;
                    if (target_fx < 0 || target_fx >= (int)fNx) continue; // Skip physical exterior

                    int dist = abs_dx_calc(dx, abs_dy, abs_dz); 
                    
                    float weight = (dist == 0) ? (1.0f / 8.0f) :
                                   (dist == 1) ? (1.0f / 16.0f) :
                                   (dist == 2) ? (1.0f / 32.0f) : (1.0f / 64.0f);

                    sum += __ldg(&fineGrid[z_offset + y_offset + target_fx]) * weight;
                    total_weight += weight;
                }
            }
        }

        // Re-normalize the partial stencil area back to full volume scaling factor
        if (total_weight > 0.0f) {
            sum /= (total_weight * 8.0f); 
        }
    }
    // ==================================================================================================================================================
    // 4. Stream final result to Coarse Grid Location
    // ==================================================================================================================================================
    const size_t coarseIdx = z * (Nx * Ny) + y * Nx + x;
    coarseGrid[coarseIdx] = sum;
}

/*----------------------------------------------------------------------------------------------------------------------------
    ProlongAndCorrect Kernel
    Prolongs the coarse correction to the fine grid, and corrects (adds to) the fine solution. i.e, For every interior fine-grid point, 
    it trilinearly interpolates the coarse-grid error/correction field and adds that interpolated correction to the current fine-grid solution.
    Boundary nodes are not updated by this kernel.
    Using unsigned comparison, we do not need to check for negative values as fx, fy and fz are non non-negative. only check if it is 0 or larger than N-1
    Maps fine coordinates to coarse by computing surrounding coordinates using precalculated weights and trilinear interpolation
    This is a memory bound kernel; 8 coarse reads, 1 fine read and 1 fine write per interior fine point. Use of __ldg and __restrict__ helps mitigate this as 
    the read-only coarse data can be cached effectively, especially because neighboring fine threads often read overlapping coarse values.
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void ProlongAndCorrect(float* __restrict__ fineSolution, 
                                  const float* __restrict__ coarseError,
                                  const size_t fNx, 
                                  const size_t fNy, 
                                  const size_t fNz, 
                                  const size_t cNx, 
                                  const size_t cNy, 
                                  const size_t cNz){
    // =========================================================================   
    // 1. Compute thread coordinates
    // =========================================================================
    const unsigned int fx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int fy = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int fz = blockIdx.z * blockDim.z + threadIdx.z;
    // =========================================================================
    // 2. Unsigned comparison: eliminates up to half the branching instructions
    // =========================================================================
    if (fx == 0 || fx >= fNx - 1 || fy == 0 || fy >= fNy - 1 || fz == 0 || fz >= fNz - 1) return;
    // =========================================================================
    // 3. Compute coarse coordinates: Creates a stencil around each coarse grid 
    //    point, maps a fine-grid point to the coarse-grid points surrounding it
    // =========================================================================
    const int cx0 = (int)(fx >> 1);
    const int cy0 = (int)(fy >> 1);
    const int cz0 = (int)(fz >> 1);
    // min() clamps the upper coarse index so the kernel does not read out of bounds near the edge of the coarse grid
    const int cx1 = min(cx0 + 1, (int)cNx - 1);
    const int cy1 = min(cy0 + 1, (int)cNy - 1);
    const int cz1 = min(cz0 + 1, (int)cNz - 1);
    // =========================================================================
    // 4. Linear interpolation weights
    // =========================================================================
    // (fx & 1) checks whether fx is odd
    // If fx is even, tx = 0.0f and o_tx = 1.0f and the fine point lies exactly on a coarse-grid point
    // If fx is odd, tx = 0.5f and o_tx = 0.5f and the fine point lies halfway between two coarse-grid points
    // The same logic applies in the y and z directions.
    const float tx = (fx & 1) * 0.5f;
    const float ty = (fy & 1) * 0.5f;
    const float tz = (fz & 1) * 0.5f;
    
    const float o_tx = 1.0f - tx;
    const float o_ty = 1.0f - ty;
    const float o_tz = 1.0f - tz;
    // =========================================================================
    // 5. Pre-calculate structural stride constants for 3D indexing
    // =========================================================================
    const size_t coarse_slice = cNx * cNy;
    // =========================================================================
    // 6. Explicit 64-bit base offset calculations: size_t for these offsets is 
    //    important because the final array index can be larger than 32 bits.
    // =========================================================================
    // Precomputes some of the larger offsets, which are used to form the eight coarse-grid indices around the fine point.
    const size_t z0_offset = (size_t)cz0 * coarse_slice;
    const size_t z1_offset = (size_t)cz1 * coarse_slice;
    const size_t y0_stride = (size_t)cy0 * cNx;
    const size_t y1_stride = (size_t)cy1 * cNx;
    // =========================================================================
    // 7. Reading the eight surrounding coarse values
    //    Read coarse error via L1 Read-Only Cache Directive (__ldg) - __ldg is a 
    //    read-only load intrinsic, allowing the compiler/hardware to use the read-only 
    //    data path or constant/texture-style caching, which is able to work due to coarseError is a const (read-only)
    // =========================================================================
    // Eight corners of the coarse-grid cell surrounding the fine point.
    float e000 = __ldg(&coarseError[z0_offset + y0_stride + cx0]);
    float e100 = __ldg(&coarseError[z0_offset + y0_stride + cx1]);
    float e010 = __ldg(&coarseError[z0_offset + y1_stride + cx0]);
    float e110 = __ldg(&coarseError[z0_offset + y1_stride + cx1]);
    float e001 = __ldg(&coarseError[z1_offset + y0_stride + cx0]);
    float e101 = __ldg(&coarseError[z1_offset + y0_stride + cx1]);
    float e011 = __ldg(&coarseError[z1_offset + y1_stride + cx0]);
    float e111 = __ldg(&coarseError[z1_offset + y1_stride + cx1]);
    // =========================================================================
    // 8. Trilinear interpolation synthesis
    //    Interpolating first in x, then in y, then in z.
    //    o_tx * e000 + tx * e100 - is linear interpolation in x at coarse location (y0, z0)
    //    o_ty * (...) + ty * (...) - interpolates in y at coarse z0
    //    o_tz * (...) + tz * (...) - interpolates in z
    // =========================================================================
    float interpolated_error = o_tz * (o_ty * (o_tx * e000 + tx * e100) + ty * (o_tx * e010 + tx * e110)) +
                               tz   * (o_ty * (o_tx * e001 + tx * e101) + ty * (o_tx * e011 + tx * e111));
    // =========================================================================
    // 9. Update fine grid array (True Correction Phase)
    // =========================================================================
    //  Fine-grid array is also stored in row-major order
    const size_t fineIdx = (size_t)fz * (fNx * fNy) + (size_t)fy * fNx + fx;
    
    // Changing this to += forces the prolonged correction to damp the error state.
    fineSolution[fineIdx] += interpolated_error;
}

/*----------------------------------------------------------------------------------------------------------------------------
    V cycle function
    Integrated CUDA streams
    To overlap host-side command queuing and asynchronous device execution, must pass a cudaStream_t 
    parameter through the recursive function and change synchronous cudaMemset to cudaMemsetAsync
    Must also explicitly provide the stream to every kernel launch.
    The active stream identifier passes down through recursive layers, ensuring all work for a single V-cycle 
    executes sequentially relative to itself without interfering with independent host tasks.
----------------------------------------------------------------------------------------------------------------------------*/
void v_cycle_hierarchical(std::vector<float*>& d_u_hierarchy, std::vector<float*>& d_f_hierarchy, std::vector<float*>& d_r_hierarchy,
                          const std::vector<size_t>& Nx_lvl, const std::vector<size_t>& Ny_lvl, const std::vector<size_t>& Nz_lvl,
                          int current_lvl, int total_levels, 
                          float h, float h2, 
                          int nu1 = 2, int nu2 = 2, cudaStream_t stream = 0) {

    const size_t Nx = Nx_lvl[current_lvl];
    const size_t Ny = Ny_lvl[current_lvl];
    const size_t Nz = Nz_lvl[current_lvl];
    // =========================================================================
    // 1. BLACKWELL WORK GENERATION SHIFT
    // =========================================================================
    // At coarser levels, a 16x8x4 block contains too few blocks to saturate an RTX 5090.
    // We adjust the block dimensions if the grid shrinks below saturation thresholds.
    const unsigned int bx = (Nx < 16) ? (unsigned int)Nx : 16;
    const unsigned int by = (Ny < 8)  ? (unsigned int)Ny : 8;
    const unsigned int bz = (Nz < 4)  ? (unsigned int)Nz : 4;
    
    dim3 block_fused(bx, by, bz);
    dim3 grid_fused((Nx + block_fused.x - 1) / block_fused.x, 
                    (Ny + block_fused.y - 1) / block_fused.y, 
                    (Nz + block_fused.z - 1) / block_fused.z);

    dim3 block_legacy(32, 8, 1);
    dim3 grid_legacy((Nx + block_legacy.x - 1) / block_legacy.x, 
                     (Ny + block_legacy.y - 1) / block_legacy.y, 
                     (Nz + block_legacy.z - 1) / block_legacy.z);

    float* d_u = d_u_hierarchy[current_lvl];
    float* d_f = d_f_hierarchy[current_lvl];
    float* d_r = d_r_hierarchy[current_lvl];
    // =========================================================================
    // 2. PRE-SMOOTHING
    // =========================================================================
    for (int i = 0; i < nu1; ++i) {
        multigrid_red_black_fused<<<grid_fused, block_fused, 0, stream>>>(d_f, d_u, Nx, Ny, Nz, h2);
    }
    // =========================================================================
    // 3. CORNERSTONE LEVEL CHECK
    // =========================================================================
    if (current_lvl == total_levels - 1) {
        // Unroll loops cleanly into the stream pipeline
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            multigrid_red_black_fused<<<grid_fused, block_fused, 0, stream>>>(d_f, d_u, Nx, Ny, Nz, h2);
        }
        return;
    }
    // =========================================================================
    // 4. COMPUTE RESIDUAL & COARSEN DOWNWARD
    // =========================================================================
    Residual_function<<<grid_legacy, block_legacy, 0, stream>>>(d_f, d_r, d_u, Nx, Ny, Nz, h2);

    const size_t cNx = Nx_lvl[current_lvl + 1];
    const size_t cNy = Ny_lvl[current_lvl + 1];
    const size_t cNz = Nz_lvl[current_lvl + 1];

    dim3 coarse_grid((cNx + block_legacy.x - 1) / block_legacy.x,
                     (cNy + block_legacy.y - 1) / block_legacy.y,
                     (cNz + block_legacy.z - 1) / block_legacy.z);

    float* d_f_coarse = d_f_hierarchy[current_lvl + 1];
    float* d_u_coarse = d_u_hierarchy[current_lvl + 1];

    const size_t coarse_bytes = cNx * cNy * cNz * sizeof(float);
    
    // Asynchronous clear completely decoupled from host tracking
    cudaMemsetAsync(d_u_coarse, 0, coarse_bytes, stream);
    
    Coarsen<<<coarse_grid, block_legacy, 0, stream>>>(d_r, d_f_coarse, cNx, cNy, cNz, Nx, Ny, Nz);

    const float h_coarse = h * 2.0f;
    const float h2_coarse = h_coarse * h_coarse;
    // =========================================================================
    // 5. RECURSIVE RECONSTRUCTION (Tail-Call Execution)
    // =========================================================================
    v_cycle_hierarchical(d_u_hierarchy, d_f_hierarchy, d_r_hierarchy,
                         Nx_lvl, Ny_lvl, Nz_lvl,
                         current_lvl + 1, total_levels,
                         h_coarse, h2_coarse, nu1, nu2, stream);
    // =========================================================================
    // 6. PROLONGATION & ERROR CORRECTION
    // =========================================================================
    ProlongAndCorrect<<<grid_legacy, block_legacy, 0, stream>>>(d_u, d_u_coarse, Nx, Ny, Nz, cNx, cNy, cNz);
    // =========================================================================
    // 7. POST-SMOOTHING
    // =========================================================================
    for (int i = 0; i < nu2; ++i) {
        multigrid_red_black_fused<<<grid_fused, block_fused, 0, stream>>>(d_f, d_u, Nx, Ny, Nz, h2);
    }
}

/*----------------------------------------------------------------------------------------------------------------------------
    One-time warmup: first-launch overhead so it is not charged to an initialisation timing.
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void warmup() {}

int main() {
    // Absorb one-time warmup cost before any measurement.
    warmup<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // 1. BASE VARIABLES
    // =========================================================================
    const size_t N = 1025;
    const size_t Nx = N, Ny = Nx, Nz = N;
    constexpr float L = 1.0f;
    const float h  = L / (Nx - 1);
    const float h2 = h * h;

    // Capture the current system wall-clock time.
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert time_t to local time struct safely.
    std::tm buf;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&buf, &in_time_t); // Windows safe variant
#else
    localtime_r(&in_time_t, &buf); // Linux safe variant
#endif

    // =========================================================================
    // 2. MULTIGRID HIERARCHY TRACKING SETUP
    // =========================================================================
    const size_t target_coarse_limit = 33;
    std::vector<float*> d_u_hierarchy;
    std::vector<float*> d_f_hierarchy;
    std::vector<float*> d_r_hierarchy;
    std::vector<size_t> Nx_lvl, Ny_lvl, Nz_lvl;
    Nx_lvl.push_back(Nx);
    Ny_lvl.push_back(Ny);
    Nz_lvl.push_back(Nz);
    std::cout << "--- Initializing Multigrid Allocation Hierarchy ---" << std::endl;
    size_t current_N = N;
    int lvl = 0;
    nvtxRangePushA("Level building (malloc only)");
    while (true) {
        size_t total_elements = Nx_lvl[lvl] * Ny_lvl[lvl] * Nz_lvl[lvl];
        size_t total_bytes    = total_elements * sizeof(float);
        float *d_u_ptr = nullptr, *d_f_ptr = nullptr, *d_r_ptr = nullptr;
        // Allocate only, no cudaMemset here:
        // d_u[0]   -> zeroed by InitializeProblem below
        // d_u[l>0] -> zeroed by the per-cycle cudaMemset inside v_cycle_hierarchical
        // d_f[*]   -> fully overwritten by InitializeProblem / Coarsen
        // d_r[*]   -> boundary zeroed below by ZeroBoundary3D; interior rewritten each cycle
        CUDA_CHECK(cudaMalloc(&d_u_ptr, total_bytes));
        CUDA_CHECK(cudaMalloc(&d_f_ptr, total_bytes));
        CUDA_CHECK(cudaMalloc(&d_r_ptr, total_bytes));
        d_u_hierarchy.push_back(d_u_ptr);
        d_f_hierarchy.push_back(d_f_ptr);
        d_r_hierarchy.push_back(d_r_ptr);
        double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        std::cout << "Level " << lvl << " Dimensions: " << Nx_lvl[lvl] << "x" << Ny_lvl[lvl] << "x" << Nz_lvl[lvl]
                  << " | Total Tier Footprint: " << (mb * 3.0) << " MB" << std::endl;
        if (current_N <= target_coarse_limit) break;
        current_N = ((current_N - 1) / 2) + 1;
        Nx_lvl.push_back(current_N);
        Ny_lvl.push_back(current_N);
        Nz_lvl.push_back(current_N);
        ++lvl;
    }
    nvtxRangePop();
    const int num_levels = d_u_hierarchy.size();
    std::cout << "Total Hierarchy Levels Generated: " << num_levels << "\n" << std::endl;

    // =========================================================================
    // 3. PROBLEM INITIALIZATION (Manufactured Solution Setup)
    // =========================================================================
    nvtxRangePushA("PROBLEM INITIALIZATION");
    const size_t finest_elements = Nx_lvl[0] * Ny_lvl[0] * Nz_lvl[0];
    const size_t finest_bytes    = finest_elements * sizeof(float);
    float* d_u_true = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_true, finest_bytes));
    dim3 init_block(32, 8, 4);
    dim3 init_grid(
        (Nx_lvl[0] + init_block.x - 1) / init_block.x,
        (Ny_lvl[0] + init_block.y - 1) / init_block.y,
        (Nz_lvl[0] + init_block.z - 1) / init_block.z
    );
    std::cout << "--- Populating Problem Initial Fields via Manufactured Solution ---" << std::endl;
    // Writes f[0], u_true[0] AND zeros u[0] (initial guess + Dirichlet shell) in one pass.
    InitializeProblem<<<init_grid, init_block>>>(
        d_f_hierarchy[0], d_u_hierarchy[0], d_u_true,
        Nx_lvl[0], Ny_lvl[0], Nz_lvl[0], h);
    CUDA_CHECK(cudaGetLastError());
    // One-time: zero the boundary of every residual array.
    {
        dim3 b(32, 8, 1);
        for (int l = 0; l < num_levels; ++l) {
            dim3 g((Nx_lvl[l] + b.x - 1) / b.x,
                   (Ny_lvl[l] + b.y - 1) / b.y,
                   (Nz_lvl[l] + b.z - 1) / b.z);
            ZeroBoundary3D<<<g, b>>>(d_r_hierarchy[l], Nx_lvl[l], Ny_lvl[l], Nz_lvl[l]);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    // Single sync after all init work is enqueued (was: sync + many implicit waits).
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // =========================================================================
    // 4. EXECUTION INITIALIZATION
    // =========================================================================
    // Allocate a tiny 8-byte bucket on the GPU to hold the sum result asynchronously
    double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    std::cout << "--- Executing Multigrid Solver ---\n" << std::endl;
    int max_v_cycles = 5;
    double h_sum = 0.0; 
    std::stringstream filename_ss;
    filename_ss << "mg_log_N" << N << "_" << std::put_time(&buf, "%Y%m%d_%H%M%S") << ".csv";
    std::string filename = filename_ss.str();
    std::ofstream log_file(filename);
    if (!log_file.is_open()) {
        std::cerr << "Error: Could not open unique log file: " << filename << std::endl;
        return -1;
    }
    std::cout << "Logging execution metrics to: " << filename << std::endl;
    log_file << "Cycle,Milliseconds,L2_Norm\n";
    // Initialize dedicated asynchronous hardware execution stream
    cudaStream_t mg_stream;
    // Allocating the stream with this flag ensures that multigrid kernels will not experience execution 
    // stalls or bubbles if other parts of the codebase launch work on the default legacy NULL stream.
    CUDA_CHECK(cudaStreamCreateWithFlags(&mg_stream, cudaStreamNonBlocking));
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));// Create start and stop event handles
    CUDA_CHECK(cudaEventCreate(&stop_event));
    // Define the functional reduction transformation operator
    DifferenceSquared transform_op;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // Create Zip Iterators to bundle the two input pointers into a single input stream
    auto zip_start = thrust::make_zip_iterator(thrust::make_tuple(d_u_hierarchy[0], d_u_true));

    // STEP A: Query the correct workspace footprint using the 9-argument modern signature
    cub::DeviceReduce::TransformReduce(// Pass a NULL pointer to query CUB for the exact memory workspace size it needs
        nullptr, temp_storage_bytes, 
        zip_start, d_sum, finest_elements, 
        cuda::std::plus{}, transform_op, 0.0, mg_stream
    );
    // STEP B: Allocate the exact temporary hardware workspace returned by the query
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // =========================================================================
    // 5. EXECUTION LOOP: Launch Multigrid V-Cycles Until Convergence
    // =========================================================================
    for (int cycle = 0; cycle < max_v_cycles; ++cycle) {
        nvtxRangePushA("V-cycle");
        // Record the start event into the stream
        CUDA_CHECK(cudaEventRecord(start_event, mg_stream));
        // 1. Fire off the primary V-cycle kernels (Non-blocking)
        v_cycle_hierarchical(
            d_u_hierarchy, d_f_hierarchy, d_r_hierarchy,
            Nx_lvl, Ny_lvl, Nz_lvl,
            0, num_levels, h, h2, 2, 2,
            mg_stream
        );
        // Fused Asynchronous Reduction: Computes subtraction, squaring, and summing. Using explicit finest level dimension calculation to guarantee memory bounds safety
        // Fused Asynchronous Reduction via Zip Iterator
        cub::DeviceReduce::TransformReduce(
            d_temp_storage, temp_storage_bytes, 
            zip_start, d_sum, finest_elements, 
            cuda::std::plus{}, transform_op, 0.0, mg_stream
        );
        // Record stop timestamp marker immediately following data reductions
        CUDA_CHECK(cudaEventRecord(stop_event, mg_stream));
        // Schedule async memory transfer back to CPU host space
        CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, mg_stream));
        // Stall only the host thread until this loop iteration's tasks finish processing
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        // Compute L2 Norm Metric
        const double h3 = static_cast<double>(h) * h * h;
        double l2_norm = std::sqrt(h_sum * h3);
        // Stream performance details straight into CSV file
        if (cycle >= 0) {
            log_file << cycle << "," << milliseconds << "," << l2_norm << "\n";
            std::cout << "V-Cycle " << cycle+1 << " Complete (" << milliseconds << " ms) | L2 Error Norm: "
                      << l2_norm << std::endl;
        }
        nvtxRangePop();
    } 
    // Close the file stream
    log_file.close();

    // ==================================================================================================================================================
    // 6. EXPLICIT SYNCHRONIZATION: synchronize the custom stream (instead of the global device) before using h_sum
    // ==================================================================================================================================================
    // Replacing the sweeping cudaDeviceSynchronize() with cudaStreamSynchronize(mg_stream) ensures the
    // host thread only waits until the exact calculations in this stream are finished
    CUDA_CHECK(cudaStreamSynchronize(mg_stream)); 
    const double h3 = static_cast<double>(h) * h * h;
    double final_l2_norm = std::sqrt(h_sum * h3);
    std::cout << "V-Cycle " << (max_v_cycles) << " Complete | L2 Error Norm vs Analytical Solution: "
              << final_l2_norm << std::endl;

    // Destroy the event handles when done
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(mg_stream));

    // ==================================================================================================================================================
    // 7. Clean up all allocated memory locations and Safely destroy the stream handle after final usage is complete
    // ==================================================================================================================================================
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_sum));

    // ==================================================================================================================================================
    // 8. MEMORY CLEANUP LIFECYCLE
    // ==================================================================================================================================================
    nvtxRangePushA("MEMORY CLEANUP");
    CUDA_CHECK(cudaFree(d_u_true));
    for (int l = 0; l < num_levels; ++l) {
        CUDA_CHECK(cudaFree(d_u_hierarchy[l]));
        CUDA_CHECK(cudaFree(d_f_hierarchy[l]));
        CUDA_CHECK(cudaFree(d_r_hierarchy[l]));
    }

    nvtxRangePop();
    std::cout << "Execution completed successfully!" << std::endl;
    return 0;
}