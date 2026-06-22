#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdint> // Required for int64_t
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
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
constexpr float inv6 = 1.0f/6.0f;
/*--------------------------------------------------------------
  Helper: linear index from (x,y,z)
  --------------------------------------------------------------*/
__device__ __inline__ size_t getLinearIdx3D(int x, int y, int z, const size_t Nx, const size_t Ny) {
    // Keeps the 2 sequential hardware MAD instructions using pure integer math
    return (size_t)z * Nx * Ny + (size_t)y * Nx + x;
}
/*--------------------------------------------------------------
  Red‑Black Gauss‑Seidel sweep (red points)
  --------------------------------------------------------------*/
__global__ void red_black_gauss_siedel_red(const float * __restrict__ f, float * __restrict__ u, const int Nx, const int Ny, const int Nz, const float h, const float h2){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    if (((x + y + z) & 1) != 0) return; 
    
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx_c = z * sliceSize + y * Nx + x;
    float rhs = f[i];
    float sum = 0.0f;
    /*
    if(x>0) sum += u[idx_c-1];
    if(x<Nx-1) sum += u[idx_c+1];
    if(y>0) sum += u[idx_c-Nx];
    if(y<Ny-1) sum += u[idx_c+Nx];
    if(z>0) sum += u[idx_c-sliceSize];
    if(z<Nz-1) sum += u[idx_c+sliceSize];
    */
    //-∇²u, array is zero padded 
    sum = (u[idx_c-1] + u[idx_c+1] + u[idx_c-Nx] + u[idx_c+Nx] + u[idx_c-sliceSize] + u[idx_c+sliceSize] - 6.0f*u[i]) / h2;
    u[i] = inv6 * (sum + h2 * rhs);
}
/*--------------------------------------------------------------
  Red‑Black Gauss‑Seidel sweep (black points)
  --------------------------------------------------------------*/
__global__ void red_black_gauss_siedel_black(const float * __restrict__ f, float * __restrict__ u, const int Nx, const int Ny, const int Nz, const float h, const float h2){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    if (((x + y + z) & 1) == 0) return; 
    
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx_c = z * sliceSize + y * Nx + x;
    float rhs = f[i];
    float sum = 0.0f;
    /*
    if(x>0) sum += u[idx_c-1];
    if(x<Nx-1) sum += u[idx_c+1];
    if(y>0) sum += u[idx_c-Nx];
    if(y<Ny-1) sum += u[idx_c+Nx];
    if(z>0) sum += u[idx_c-sliceSize];
    if(z<Nz-1) sum += u[idx_c+sliceSize];
    */
    //-∇²u
    sum = (u[idx_c-1] + u[idx_c+1] + u[idx_c-Nx] + u[idx_c+Nx] + u[idx_c-sliceSize] + u[idx_c+sliceSize] - 6.0f*u[i]) / h2;
    u[i] = inv6 * (sum + h2 * rhs);
}
/*--------------------------------------------------------------
  Residual kernel: r = f - A u
  --------------------------------------------------------------*/
__global__ void Residual_function(const float * __restrict__ f, float * __restrict__ r, float * __restrict__ u, const int Nx, const int Ny, const int Nz, const float h, const float h2){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    //We ensure that we stay within the 0-padding
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx_c = z * sliceSize + y * Nx + x;
    //Make array for Laplacian (-∇²u) 0-padded, to ensure that no checks are necessary
    float Laplacian = (u[idx_c-1] + u[idx_c+1] + u[idx_c-Nx] + u[idx_c+Nx] + u[idx_c-sliceSize] + u[idx_c+sliceSize] - 6.0f*u[i]) / h2;
    r[i] = f[i] - Laplacian;
}
__global__ void Reduction(){

}
__global__ void Prolongate_function(){
  
}
/*--------------------------------------------------------------
  Host driver: one V‑cycle (pre‑smooth, residual, restrict,
  coarse solve, prolongate, post‑smooth)
  --------------------------------------------------------------*/
void v_cycle(float *d_f, float *d_u, float *d_r, int Nx, int Ny, int Nz, const float h, const float h2, int v1=2, int nu2=2){
    const size_t numCells = Nx*Ny*Nz;
    const size_t NX = Nx + 2, NY = NX, NZ = NX;
    dim3 block(8, 8, 8);
    dim3 grid((Nx+block.x-1)/block.x, (Ny+block.y-1)/block.y, (Nz+block.z-1)/block.z);
    const size_t bytes = (size_t)NX * NY * NZ * sizeof(float);
    cudaStream_t stream; cudaStreamCreate(&stream);
    //Memory Allocation
    float *d_phi_old, *d_phi_new, *d_rhs;
    CUDA_CHECK(cudaMalloc(&d_phi_old, bytes));
    CUDA_CHECK(cudaMalloc(&d_phi_new, bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs, bytes));
    // Zero out the arrays completely. This sets the padding borders to 0.
    CUDA_CHECK(cudaMemset(d_phi_old, 0, bytes));
    CUDA_CHECK(cudaMemset(d_phi_new, 0, bytes));
    CUDA_CHECK(cudaMemset(d_rhs, 0, bytes));
    // 1. Pre-smoothing
    // Gauss-Siedel
    for (size_t i=0;i<nu2;++i){
      red_black_gauss_siedel_red<<<grid,block,0,stream>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
      red_black_gauss_siedel_black<<<grid,block,0,stream>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
    }
    // Residual
    Residual_function<<<grid,block,0,stream>>>(d_f, d_u, d_r, Nx, Ny, Nz, h, h2);
    // 2. Restrict the residual

    // 3. Recursive call on the coarse grid
    // 4. Prolongate and correct
    // 5. Post-smoothing
    for (size_t i=0;i<nu2;++i){
      red_black_gauss_siedel_red<<<grid,block,0,stream>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
      red_black_gauss_siedel_black<<<grid,block,0,stream>>>(d_f, d_u, Nx, Ny, Nz, h, h2);
    }
    //Memory Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_phi_old));
    CUDA_CHECK(cudaFree(d_phi_new));
    CUDA_CHECK(cudaFree(d_rhs));
}
int main(){
    // We are running on an RTX 5090, with 21,760 CUDA cores across its 170 Streaming Multiprocessors (SMs)
    // To achieve max utilization of the core, we use at least 511 grid
    // A full V-cycle hierarchy (tracking u, f, r across all levels) will consume roughly 5 to 6 GB of total VRAM
    // 513 → 257 → 129 → 65 → 33
    const size_t N = 513; // power of two, change for scaling study
    const size_t Nx = N, Ny = Nx, Nz = Nx;
    int64_t numCells = static_cast<int64_t>(Nx) * Ny * Nz; 
    constexpr  float L = 1.0f; //f suffix explicitly tells the compiler to treat the number as a float rather than a default double
    const float h = L / (Nx-1);
    const float h2 = h*h;

    std::cout << "Execution completed successfully!" << std::endl;

    return 0;
}

/*
Kernal Fine-to-Coarse:
performs ν₁ pre-smoothing iterations /
(fused Red-Black sweeps) in shared memory, calculates the residual (r = f - Ax) /
restricts it down to the coarser grid (\(I_h^{2h}r\)) 
writes only the restricted coarse values back to global memory. 
*/
__global__ void FinetoCoarse(float *d_f, float *d_u, float *d_r, float *f, float *r, float *u, int Nx, int Ny, int Nz, const float h, const float h2, int v1){
  for (int i=0;i<v1;++i){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    if (((x + y + z) & 1) != 0) return; 
    
    size_t i_1 = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize1 = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx_c = z * sliceSize1 + y * Nx + x;
    float rhs = f[i_1];
    float sum = 0.0f;

    sum = (u[idx_c-1] + u[idx_c+1] + u[idx_c-Nx] + u[idx_c+Nx] + u[idx_c-sliceSize1] + u[idx_c+sliceSize1] - 6.0f*u[i_1]) / h2;
    u[i_1] = inv6 * (sum + h2 * rhs);
  }
  for (int j=0;j<v1;++j){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    if (((x + y + z) & 1) == 0) return; 
    
    size_t i_2 = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize2 = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx_c = z * sliceSize2 + y * Nx + x;
    float rhs = f[i_2];
    float sum = 0.0f;

    sum = (u[idx_c-1] + u[idx_c+1] + u[idx_c-Nx] + u[idx_c+Nx] + u[idx_c-sliceSize2] + u[idx_c+sliceSize2] - 6.0f*u[i_2]) / h2;
    u[i_2] = inv6 * (sum + h2 * rhs);
  }
  for (int k=0;k<v1;++k){
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x <= 0 || x >= Nx - 1 || y <= 0 || y >= Ny - 1 || z <= 0 || z >= Nz - 1) return;
    
    size_t i = getLinearIdx3D(x,y,z,Nx,Ny);
    size_t sliceSize = static_cast<size_t>(Nx) * static_cast<size_t>(Ny);
    size_t idx_c = z * sliceSize + y * Nx + x;
    //Make array for Laplacian (-∇²u) 0-padded, to ensure that no checks are necessary
    float Laplacian = (u[idx_c-1] + u[idx_c+1] + u[idx_c-Nx] + u[idx_c+Nx] + u[idx_c-sliceSize] + u[idx_c+sliceSize] - 6.0f*u[i]) / h2;
    r[i] = f[i] - Laplacian;
  }
}
__global__ void FinetoCoarseCooperative(float *f, float *u, float *r, int Nx, int Ny, int Nz, const float h2, int v1) {
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
    for (int iter = 0; iter < v1; ++iter) {
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
/*
Kernel Coarse-to-Fine:
Create a fused kernel that reads the corrected coarse-grid solution.
prolongates it to the fine grid (\(I_{2h}^{h}e\)).
adds it to the existing fine solution.
immediately performs ν₂ post-smoothing iterations entirely within shared memory before a single global memory write.
*/
__global__ void CoarsetoFine(float *d_f, float *d_u, float *d_r, float *f, float *u, float *u_r, float *u_d, int Nx, int Ny, int Nz, const float h, const float h2, int nu2){
}
/*
Test Case 1: Triple Sine Solution (Dirichlet Boundary Conditions)
This is the most common test case for 3D Geometric Multigrid. It features zero Dirichlet boundary conditions on all sides of the unit cube, 
making it ideal for checking interior relaxation stencils and smoothing performance.
Exact Solution (\(u_{exact}\)):\(u(x,y,z)=\sin (\pi x)\sin (\pi y)\sin (\pi z)\)
Forcing Function (\(f\)):\(f(x,y,z)=-3\pi ^{2}\sin (\pi x)\sin (\pi y)\sin (\pi z)\)
Boundary Conditions: \(u = 0\) on all boundaries of the domain \(\partial\Omega\).

Test Case 2: Polynomial Solution (Non-Zero Boundary Conditions)
Polynomial fields test the capability of your restriction, prolongation, and smoothing functions to capture structural profiles. 
Because the boundary values are non-zero, this case validates that your boundary conditions are properly accounted for in the residual equations 
across grid levels.
Exact Solution (\(u_{exact}\)):\(u(x,y,z)=x^{3}+y^{3}+z^{3}\)
Forcing Function (\(f\)):\(f(x,y,z)=6x+6y+6z\)
Boundary Conditions: Evaluate \(u_{exact}\) directly at the boundaries:At \(x=0\): \(u(0, y, z) = y^3 + z^3\)
At \(x=1\): \(u(1, y, z) = 1 + y^3 + z^3\)(Repeat analogously for \(y\) and \(z\) faces)

Test Case 3: Mixed High-Frequency Trigonometric Field
Multigrid algorithms are designed to eliminate high-frequency errors on finer grids and low-frequency errors on coarser grids. 
This mixed-frequency equation tests the multigrid's efficiency across a broader wave spectrum.
Exact Solution (\(u_{exact}\)):\(u(x,y,z)=\sin (2\pi x)\cos (\pi y)\sin (3\pi z)\)
Forcing Function (\(f\)):\(f(x,y,z)=-14\pi ^{2}\sin (2\pi x)\cos (\pi y)\sin (3\pi z)\)
Boundary Conditions: Evaluated explicitly from \(u_{exact}\):\(u = 0\) at \(x=0, x=1, z=0, z=1\)
At \(y=0\): \(u(x, 0, z) = \sin(2\pi x) \sin(3\pi z)\)At \(y=1\): \(u(x, 1, z) = -\sin(2\pi x) \sin(3\pi z)\)

Test Case 4: Asymmetric Exponential Solution
This case tests the multigrid solver against steep gradients and smooth non-symmetric solutions, ensuring that cell stencils do not suffer from directional bias.
Exact Solution (\(u_{exact}\)):\(u(x,y,z)=e^{x}e^{2y}e^{3z}=e^{x+2y+3z}\)
Forcing Function (\(f\)):\(f(x,y,z)=14e^{x+2y+3z}\)
Boundary Conditions: Set Dirichlet boundaries matching \(u(x, y, z) = e^{x + 2y + 3z}\) at all 6 outer boundaries.
*/