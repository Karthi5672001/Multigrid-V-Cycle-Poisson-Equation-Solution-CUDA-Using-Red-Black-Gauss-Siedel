#ifdef _MSC_VER
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif
//Compilation in Docker WSL in powershell:
//cd C:\Users\Karthi\CUDA
//docker build -t cuda133-cpp-dev .
//docker run --gpus all --cap-add=SYS_ADMIN -it --rm -v "${PWD}:/workspace" nvidia/cuda:13.3.0-devel-ubuntu22.04 bash
//nvcc --version
//cd workspace
//nvcc -O3 --use_fast_math --generate-code arch=compute_120,code=sm_120 --ptxas-options=-v Poisson_redesign.cu -o Poisson_solver_redesign
//nsys profile --trace=cuda,nvtx,osrt --sample=cpu --force-overwrite true -o Poisson_solver_redesign ./Poisson_solver_redesign
//ncu --metrics sm__sol_pct,dram__sol_pct ./Poisson_solver_redesign
// CHANGES IN THIS VERSION (initialisation / first-V-cycle latency):
//  * Removed all 30 unnecessary cudaMemset() calls from the hierarchy build.
//  * InitializeProblem now also writes u = 0 (initial guess + Dirichlet shell)
//    in the same pass, so d_u[0] needs no separate memset.
//  * New ZeroBoundary3D() zeroes only the 6 faces of each d_r (one-time, ~170x
//    less traffic than a full memset). d_r interior is rewritten each cycle.
//  * Removed the 4.3 GB host std::vector<float> h_err_sq.
//  * Trivial warmup kernel at startup absorbs one-time driver/JIT/context cost.
//  * NVTX marks converted to ranges so durations (not just instants) show up.
//
// Correctness contract for zeroing (why the above is safe):
//   d_u[0]   : zeroed by InitializeProblem (interior = initial guess 0, shell = Dirichlet 0)
//   d_u[l>0] : zeroed by the per-cycle cudaMemset inside v_cycle_hierarchical before first use
//   d_f[*]   : never zeroed (InitializeProblem / Coarsen overwrite every element)
//   d_r[*]   : boundary zeroed once by ZeroBoundary3D; interior overwritten by Residual each cycle
// The CUDA compiler (nvcc) is highly aggressive, but it operates under strict rules. 
/*
Why using const so often for variables:
1. Compiler optimisation
When a variable is declared without const, the compiler must assume that its value 
could potentially change at any point later in the execution flow. When you mark an 
index or parameter as const (e.g., const size_t fine_slice = fNx * fNy;), you explicitly 
promise that the value is immutable. This allows the compiler to:
Propagate values directly: It can replace the variable name with its calculated value downstream.
Prevent redundant recalculations: It knows it never has to re-evaluate that mathematical expression.
2. It Enables Direct Register Allocation
GPUs have a limited number of high-speed registers per thread. 
If a variable is mutable, the compiler may have to track its lifecycle carefully, sometimes caching it 
in local memory (which spills to slow global memory if you run out of registers).Declaring variables as 
const helps the compiler's optimization pass figure out the exact lifecycle of the value. It can fit the 
variable cleanly into a dedicated register or even optimize it out of existence by folding it into a 
single assembly instruction.
3. It Signals Safe Caching Actions (__ldg interaction)
While const on local variables inside a thread changes how registers are handled, using const on pointer 
arguments (like const float* __restrict__ fineGrid) is what allows the GPU to use the Read-Only Data Cache
(L1/Texture pipeline). It tells the hardware that no thread in the entire grid will modify this memory 
during the execution, making features like the __ldg() instruction safe to use.
4. It Prevents Critical Thread Math Errors
In complex 3D stencil calculations, thread index formulas like x, y, z, and base_idx are the foundation 
of your memory safety. If an accidental typo modifies one of these tracking variables mid-kernel, it will
cause silent data corruption or an out-of-bounds memory crash. Using const turns these accidental mistakes
into compile-time errors instead of hard-to-debug runtime bugs.
Unlike OpenMP pragmas (which split loop iterations across CPU threads), #pragma unroll is a CUDA compiler hint for nvcc.It tells the compiler: "Take this loop and copy-paste the inside code 3 times sequentially so the GPU doesn't waste time checking loop counters (dz <= 1) or incrementing variables (dz++)."This reduces instruction overhead and helps the GPU execute the loop faster.2. Why can they be removed?You can delete them without breaking anything because:The main performance gain is already elsewhere: The vast majority (over 90%) of your grid points go through the is_interior branch, which is already completely written out by hand (fully unrolled) without any loops.Modern compilers are smart: Even without the hint, modern versions of the CUDA compiler are often smart enough to automatically unrolled small loops with fixed bounds (like -1 to 1).
*/
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

/*----------------------------------------------------------------------------------------------------------------------------
    CUDA error Macro - handle runtime errors cleanly 
    do {...} while(0) - for safety, preventing macro expansion bugs you can get from if-else blocks, and thus acts like a single statement
    __LINE__ - automatically prints the exact line number in the source file where the CUDA call failed, saving time during debugging
    cudaGetErrorString() - Converts the numeric error code into readable message like cudaErrorIllegalAddress
    exit(EXIT_FAILURE) - stops the program instantly
----------------------------------------------------------------------------------------------------------------------------*/
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
// Inline helper to compute absolute distance sum clearly
__device__ __forceinline__ int abs_dx_calc(int dx, int abs_dy, int abs_dz) {
    return abs(dx) + abs_dy + abs_dz;
}

/*----------------------------------------------------------------------------------------------------------------------------
    Helper: linear index from (x,y,z)
    Gives the 1D index from 3D grid so that we can work with a flat array that is beneficial to GPU computing
    explicit casting to size_t for the prevention of 32 bit integer overflow
----------------------------------------------------------------------------------------------------------------------------*/
__device__ __inline__ size_t getLinearIdx3D(unsigned int x, 
                                            unsigned int y, 
                                            unsigned int z, 
                                            const size_t Nx, 
                                            const size_t Ny) {
    return (size_t)z * Nx * Ny + (size_t)y * Nx + x;
}


// Custom squaring difference functor compatible with zip iterators
struct DifferenceSquared {
    __host__ __device__ __forceinline__
    double operator()(const thrust::tuple<float, float>& t) const {
        // Unpack the paired values safely from the tuple
        const double a = static_cast<double>(thrust::get<0>(t));
        const double b = static_cast<double>(thrust::get<1>(t));
        
        // Cast the difference to double before squaring to prevent loss of precision 
        // during massive (>1 billion elements) reduction accumulations.
        const double diff = a - b;
        return diff * diff;
    }
};

/*----------------------------------------------------------------------------------------------------------------------------
    Initialization Kernel
    Sets up Manufactured Problem on Level 0
    ALSO zeroes the solution u (initial guess 0 + Dirichlet shell 0)
    in the same pass, so no separate cudaMemset is needed for d_u[0].
    __restrict__ - marking pointers as restricted, so compiler knows that f, u, u_true 
    do not overlap in memory allowing for optimising memory usage and maximise cache efficiency
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void InitializeProblem(float* __restrict__ f,
                                  float* __restrict__ u,
                                  float* __restrict__ u_true,
                                  const size_t Nx, 
                                  const size_t Ny, 
                                  const size_t Nz,
                                  const float h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    //prevents out-of-bounds memory accesses when the total number of threads 
    //in the grid does not perfectly match the domain dimensions
    if (x >= Nx || y >= Ny || z >= Nz) return;

    size_t i = getLinearIdx3D(x, y, z, Nx, Ny);

    float px = x * h;
    float py = y * h;
    float pz = z * h;

    constexpr float PI = 3.141592653589793f;

    float u_true_i = sinf(PI * px) * sinf(PI * py) * sinf(PI * pz);

    u_true[i] = u_true_i;
    f[i]      = 3.0f * PI * PI * u_true_i;
    u[i]      = 0.0f;   // initial guess = 0, and sets the Dirichlet boundary to 0
}

/*----------------------------------------------------------------------------------------------------------------------------
    Zero only the 6 faces (outer shell) of a 3D array.
    Used once at start-up for every residual array. The interior is
    rewritten every cycle by Residual_function, so it never needs zeroing.
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void ZeroBoundary3D(float* __restrict__ a,
                               const size_t Nx, 
                               const size_t Ny, 
                               const size_t Nz)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= Nx || y >= Ny || z >= Nz) return;

    bool onShell = (x == 0 || y == 0 || z == 0 ||
                    x == Nx - 1 || y == Ny - 1 || z == Nz - 1);
    if (!onShell) return;

    a[getLinearIdx3D(x, y, z, Nx, Ny)] = 0.0f;
}

/*----------------------------------------------------------------------------------------------------------------------------
    Red-Black Gauss-Siedel Kernel (Black)
    Allows a much easier parallel apporach for GPUs, as we can calculate based on adjacent elements of a point
    freely as none of either red or black depend on others of the same colour.
    __restrict__ - marking pointers as restricted, so compiler knows that f, u
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void multigrid_red_black_fused(
    const float * __restrict__ f, 
    float * __restrict__ u,
    const int Nx, const int Ny, const int Nz, 
    const float h2) 
{
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

    const int is_black = (x + y + z) & 1; 

    if (x < Nx && y < Ny && z < Nz) {
        s_u[stz][sty][stx] = u[i];
    } else {
        s_u[stz][sty][stx] = 0.0f;
    }

    if (tx == 0 && (x - 1) >= 0)           s_u[stz][sty][stx - 1] = u[i - 1];
    if (tx == BLOCK_X - 1 && (x + 1) < Nx) s_u[stz][sty][stx + 1] = u[i + 1];
    if (ty == 0 && (y - 1) >= 0)           s_u[stz][sty - 1][stx] = u[i - Nx];
    if (ty == BLOCK_Y - 1 && (y + 1) < Ny) s_u[stz][sty + 1][stx] = u[i + Nx];
    if (tz == 0 && (z - 1) >= 0)           s_u[stz - 1][sty][stx] = u[i - sliceSize];
    if (tz == BLOCK_Z - 1 && (z + 1) < Nz) s_u[stz + 1][sty][stx] = u[i + sliceSize];

    __syncthreads();

    const bool is_interior = (x > 0 && x < Nx - 1 && y > 0 && y < Ny - 1 && z > 0 && z < Nz - 1);

    if (is_interior && !is_black) {
        const float sum = s_u[stz][sty][stx - 1] + s_u[stz][sty][stx + 1] +
                          s_u[stz][sty - 1][stx] + s_u[stz][sty + 1][stx] +
                          s_u[stz - 1][sty][stx] + s_u[stz + 1][sty][stx];
        
        s_u[stz][sty][stx] = (sum + h2 * f[i]) * (1.0f / 6.0f);
        u[i] = s_u[stz][sty][stx];
    }

    __syncthreads();

    if (is_interior && is_black) {
        const float sum = s_u[stz][sty][stx - 1] + s_u[stz][sty][stx + 1] +
                          s_u[stz][sty - 1][stx] + s_u[stz][sty + 1][stx] +
                          s_u[stz - 1][sty][stx] + s_u[stz + 1][sty][stx];
        
        u[i] = (sum + h2 * f[i]) * (1.0f / 6.0f);
    }
}

/*----------------------------------------------------------------------------------------------------------------------------
    Residual Kernel
    Threads on the boundary planes drop out immediately. 
    They perform zero global memory reads or writes, keeping memory traffic low.
    Replacing division by h2 with multiplication by invH2 saves significant clock cycles per thread,
    since floating-point division is one of the slowest basic operations on a GPU.
    Because a thread block processes adjacent elements, loading the 6 neighboring elements through __ldg() allows threads in the same warp to share data through 
    the L1 cache seamlessly rather than pulling duplicates from high-latency global memory
----------------------------------------------------------------------------------------------------------------------------*/
__global__ void Residual_function(const float * __restrict__ f, 
                                  float * __restrict__ r, 
                                  const float * __restrict__ u,
                                  const size_t Nx, 
                                  const size_t Ny, 
                                  const size_t Nz,  
                                  const float h2){
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    // 1. Highly efficient early-exit boundary guard
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    // 2. Pre-calculate layout parameters and strides
    const size_t sliceSize = Nx * Ny;
    const size_t i = z * sliceSize + y * Nx + x;
    const float invH2 = 1.0f / h2;

    // 3. Load center variables and neighboring stencil points via L1 cache (__ldg)
    const float u_c = __ldg(&u[i]);
    const float sum = __ldg(&u[i - 1])         + __ldg(&u[i + 1]) + 
                      __ldg(&u[i - Nx])        + __ldg(&u[i + Nx]) + 
                      __ldg(&u[i - sliceSize]) + __ldg(&u[i + sliceSize]);

    // 4. Match your exact mathematical formulation using fast multiplication instead of division
    const float Laplacian = -(sum - 6.0f * u_c) * invH2;

    // 5. Stream the final residual value back to global memory
    r[i] = __ldg(&f[i]) - Laplacian;
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
                        const size_t Nx, 
                        const size_t Ny, 
                        const size_t Nz, 
                        const size_t fNx, 
                        const size_t fNy, 
                        const size_t fNz) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= Nx || y >= Ny || z >= Nz) return;//out of bounds check

    // 1. Convert to fine-grid anchors (using fast bitwise shift)
    const int fx = (int)(x << 1);
    const int fy = (int)(y << 1);
    const int fz = (int)(z << 1);
    // 2. Pre-calculate structural offsets to prevent 3D index calculations inside loops
    const size_t fine_slice = fNx * fNy;
    // 3. Separate boundary threads from internal threads to remove 'min/max' overhead entirely
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
        // --- SAFE FALLBACK PATH: BOUNDARY ONLY ---
        // Keeps code correct on physical outer grid edges
        #pragma unroll
        for (int dz = -1; dz <= 1; dz++) {
            int s_fz = max(0, min(fz + dz, (int)fNz - 1));
            size_t z_offset = (size_t)s_fz * fine_slice;
            int abs_dz = abs(dz);

            #pragma unroll
            for (int dy = -1; dy <= 1; dy++) {
                int s_fy = max(0, min(fy + dy, (int)fNy - 1));
                size_t y_offset = (size_t)s_fy * fNx;
                int abs_dy = abs(dy);

                #pragma unroll
                for (int dx = -1; dx <= 1; dx++) {
                    int s_fx = max(0, min(fx + dx, (int)fNx - 1));
                    int dist = abs_dx_calc(dx, abs_dy, abs_dz); 
                    
                    float weight = (dist == 0) ? (1.0f / 8.0f) :
                                   (dist == 1) ? (1.0f / 16.0f) :
                                   (dist == 2) ? (1.0f / 32.0f) : (1.0f / 64.0f);

                    sum += __ldg(&fineGrid[z_offset + y_offset + s_fx]) * weight;
                }
            }
        }
    }

    // 4. Stream final result to Coarse Grid Location
    const size_t coarseIdx = z * (Nx * Ny) + y * Nx + x;
    coarseGrid[coarseIdx] = sum;
}

__global__ void ProlongAndCorrect(float* __restrict__ fineSolution, 
                                  const float* __restrict__ coarseError,
                                  const size_t fNx, 
                                  const size_t fNy, 
                                  const size_t fNz, 
                                  const size_t cNx, 
                                  const size_t cNy, 
                                  const size_t cNz){
    const size_t fx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t fy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t fz = blockIdx.z * blockDim.z + threadIdx.z;
    if (fx <= 0 || fx >= fNx - 1 || fy <= 0 || fy >= fNy - 1 || fz <= 0 || fz >= fNz - 1) return;

    // 3. Compute coarse coordinates (bitwise shift right replaces division by 2)
    const int cx0 = (int)(fx >> 1);
    const int cy0 = (int)(fy >> 1);
    const int cz0 = (int)(fz >> 1);

    const int cx1 = min(cx0 + 1, (int)cNx - 1);
    const int cy1 = min(cy0 + 1, (int)cNy - 1);
    const int cz1 = min(cz0 + 1, (int)cNz - 1);

    // 4. Linear interpolation weights
    const float tx = (fx & 1) * 0.5f;
    const float ty = (fy & 1) * 0.5f;
    const float tz = (fz & 1) * 0.5f;
    
    const float o_tx = 1.0f - tx;
    const float o_ty = 1.0f - ty;
    const float o_tz = 1.0f - tz;
    
    // 5. Pre-calculate structural stride constants for 3D indexing
    const size_t coarse_slice = cNx * cNy;

    // 6. Read coarse error via L1 Read-Only Cache Directive (__ldg)
    // This dramatically optimizes the redundant overlapping lookups between adjacent threads.
    float e000 = __ldg(&coarseError[cz0 * coarse_slice + cy0 * cNx + cx0]);
    float e100 = __ldg(&coarseError[cz0 * coarse_slice + cy0 * cNx + cx1]);
    float e010 = __ldg(&coarseError[cz0 * coarse_slice + cy1 * cNx + cx0]);
    float e110 = __ldg(&coarseError[cz0 * coarse_slice + cy1 * cNx + cx1]);
    float e001 = __ldg(&coarseError[cz1 * coarse_slice + cy0 * cNx + cx0]);
    float e101 = __ldg(&coarseError[cz1 * coarse_slice + cy0 * cNx + cx1]);
    float e011 = __ldg(&coarseError[cz1 * coarse_slice + cy1 * cNx + cx0]);
    float e111 = __ldg(&coarseError[cz1 * coarse_slice + cy1 * cNx + cx1]);

    // 7. Trilinear interpolation synthesis
    float interpolated_error = o_tz * (o_ty * (o_tx * e000 + tx * e100) + ty * (o_tx * e010 + tx * e110)) +
                               tz   * (o_ty * (o_tx * e001 + tx * e101) + ty * (o_tx * e011 + tx * e111));

    // 8. Update fine grid array
    const size_t fineIdx = fz * (fNx * fNy) + fy * fNx + fx;
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
void v_cycle_hierarchical(std::vector<float*>& d_u_hierarchy,
                          std::vector<float*>& d_f_hierarchy,
                          std::vector<float*>& d_r_hierarchy,
                          const std::vector<size_t>& Nx_lvl,
                          const std::vector<size_t>& Ny_lvl, 
                          const std::vector<size_t>& Nz_lvl,
                          int current_lvl, 
                          int total_levels, 
                          float h, 
                          float h2, 
                          int nu1 = 2, 
                          int nu2 = 2,
                          cudaStream_t stream = 0) // Added stream parameter defaulting to default stream
{
    size_t Nx = Nx_lvl[current_lvl];
    size_t Ny = Ny_lvl[current_lvl];
    size_t Nz = Nz_lvl[current_lvl];

    // Fused Kernel optimal block configuration (16x8x4 = 512 threads)
    dim3 block_fused(16, 8, 4);
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

    // 1. PRE-SMOOTHING (Pass stream to kernel configuration fourth parameter)
    for (int i = 0; i < nu1; ++i) {
        multigrid_red_black_fused<<<grid_fused, block_fused, 0, stream>>>(d_f, d_u, Nx, Ny, Nz, h2);
    }


    // 2. CORNERSTONE LEVEL CHECK
    if (current_lvl == total_levels - 1) {
        for (int i = 0; i < 10; ++i) {
            multigrid_red_black_fused<<<grid_fused, block_fused, 0, stream>>>(d_f, d_u, Nx, Ny, Nz, h2);
        }
        return;
    }

    // 3. COMPUTE RESIDUAL & COARSEN DOWNWARD
    Residual_function<<<grid_legacy, block_legacy, 0, stream>>>(d_f, d_r, d_u, Nx, Ny, Nz, h2);

    size_t cNx = Nx_lvl[current_lvl + 1];
    size_t cNy = Ny_lvl[current_lvl + 1];
    size_t cNz = Nz_lvl[current_lvl + 1];

    dim3 coarse_grid((cNx + block_legacy.x - 1) / block_legacy.x,
                     (cNy + block_legacy.y - 1) / block_legacy.y,
                     (cNz + block_legacy.z - 1) / block_legacy.z);

    float* d_f_coarse = d_f_hierarchy[current_lvl + 1];
    float* d_u_coarse = d_u_hierarchy[current_lvl + 1];

    // Reset the coarse error variable to 0 for this V-cycle 
    size_t coarse_bytes = cNx * cNy * cNz * sizeof(float);
    /*
    Switch to asynchronous memset to prevent pipeline bubbles on the stream
    Regular cudaMemset behaves like a synchronization barrier and stops the queue 
    cudaMemsetAsync correctly places the zero-out task inside the execution stream
    */
    cudaMemsetAsync(d_u_coarse, 0, coarse_bytes, stream);
    // calls the optimized branch-separated Coarsen kernel
    Coarsen<<<coarse_grid, block_legacy, 0, stream>>>(d_r, d_f_coarse, cNx, cNy, cNz, Nx, Ny, Nz);

    float h_coarse = h * 2.0f;
    float h2_coarse = h_coarse * h_coarse;

    // 4. RECURSIVE RECONSTRUCTION (Forward the active stream down)
    v_cycle_hierarchical(d_u_hierarchy, d_f_hierarchy, d_r_hierarchy,
                         Nx_lvl, Ny_lvl, Nz_lvl,
                         current_lvl + 1, total_levels,
                         h_coarse, h2_coarse, nu1, nu2, stream);

    // 5. PROLONGATION & ERROR CORRECTION
    ProlongAndCorrect<<<grid_legacy, block_legacy, 0, stream>>>(d_u, d_u_coarse, Nx, Ny, Nz, cNx, cNy, cNz);

    // 6. POST-SMOOTHING
    for (int i = 0; i < nu2; ++i) {
        multigrid_red_black_fused<<<grid_fused, block_fused, 0, stream>>>(d_f, d_u, Nx, Ny, Nz, h2);
    }
}

// One-time warmup: absorbs driver init / JIT / context / first-launch overhead
// so it is not charged to an initialisation timing. 
__global__ void warmup() {}

int main() {
    // Absorb one-time warmup cost before any measurement.
    warmup<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    // 1. BASE VARIABLES
    const size_t N = 1025;
    const size_t Nx = N, Ny = Nx, Nz = N;
    constexpr float L = 1.0f;
    const float h  = L / (Nx - 1);
    const float h2 = h * h;
    // Capture the current system wall-clock time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    // Convert time_t to local time struct safely
    std::tm buf;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&buf, &in_time_t); // Windows safe variant
#else
    localtime_r(&in_time_t, &buf); // Linux/macOS safe variant
#endif
    // 2. MULTIGRID HIERARCHY TRACKING SETUP
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
        // Allocate only. Do NOT cudaMemset here:
        //   d_u[0]   -> zeroed by InitializeProblem below
        //   d_u[l>0] -> zeroed by the per-cycle cudaMemset inside v_cycle_hierarchical
        //   d_f[*]   -> fully overwritten by InitializeProblem / Coarsen
        //   d_r[*]   -> boundary zeroed below by ZeroBoundary3D; interior rewritten each cycle
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
    // 3. PROBLEM INITIALIZATION (Manufactured Solution Setup)
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
    // 4. EXECUTION INITIALIZATION
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
    //Allocating the stream with this flag ensures that multigrid kernels will not experience execution 
    //stalls or bubbles if other parts of the codebase launch work on the default legacy NULL stream.
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
    cub::DeviceReduce::TransformReduce(
        nullptr, temp_storage_bytes, 
        zip_start, d_sum, finest_elements, 
        cuda::std::plus{}, transform_op, 0.0, mg_stream
    );// Pass a NULL pointer to query CUB for the exact memory workspace size it needs
    // STEP B: Allocate the exact temporary hardware workspace returned by the query
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // --- MAIN SOLVER LOOP ---
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
        if (cycle > 0) {
            log_file << cycle << "," << milliseconds << "," << l2_norm << "\n";
            std::cout << "V-Cycle " << cycle << " Complete (" << milliseconds << " ms) | L2 Error Norm: "
                      << l2_norm << std::endl;
        }
        nvtxRangePop();
    } 
    // Close the file stream
    log_file.close();
    // 5. Explicitly synchronize the custom stream (instead of the global device) before using h_sum
    // Replacing the sweeping cudaDeviceSynchronize() with cudaStreamSynchronize(mg_stream) ensures the
    // host thread only waits until the exact calculations in this stream are finished
    CUDA_CHECK(cudaStreamSynchronize(mg_stream)); 
    const double h3 = static_cast<double>(h) * h * h;
    double final_l2_norm = std::sqrt(h_sum * h3);
    std::cout << "V-Cycle " << (max_v_cycles - 1) << " Complete | L2 Error Norm vs Analytical Solution: "
              << final_l2_norm << std::endl;
    // Destroy the event handles when done
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaStreamDestroy(mg_stream));
    // 6. Clean up all allocated memory locations and Safely destroy the stream handle after final usage is complete
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_sum));
    // 7. MEMORY CLEANUP LIFECYCLE (From your original code)
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