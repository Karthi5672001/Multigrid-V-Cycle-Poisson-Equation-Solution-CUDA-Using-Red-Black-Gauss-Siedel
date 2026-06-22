// poisson_mg.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

constexpr int RADIUS = 1;   // 7‑point stencil
constexpr float inv6 = 1.0f / 6.0f;   // Jacobi weight (for simplicity)

/*--------------------------------------------------------------
  Helper: linear index from (x,y,z)
  --------------------------------------------------------------*/
__device__ __host__ __inline__ size_t idx(int x, int y, int z, int Nx, int Ny)
{
    return z * Ny * Nx + y * Nx + x;
}

/*--------------------------------------------------------------
  Red‑Black Gauss‑Seidel sweep (red points)
  --------------------------------------------------------------*/
__global__ void rbgs_red_kernel(const float * __restrict__ f, float * __restrict__ u, const int Nx, const int Ny, const int Nz, const float h2)               // h^2
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;int y = blockIdx.y * blockDim.y + threadIdx.y;int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;
    // Red points: (x+y+z) % 2 == 0
    if (((x + y + z) & 1) != 0) return;

    size_t i = idx(x, y, z, Nx, Ny);float rhs = f[i];

    // 6‑point stencil contributions (neighbors already updated if they are black)
    float sum = 0.0f;if (x > 0)     sum += u[idx(x-1, y, z, Nx, Ny)];if (x < Nx-1)  sum += u[idx(x+1, y, z, Nx, Ny)];if (y > 0)     sum += u[idx(x, y-1, z, Nx, Ny)];if (y < Ny-1)  sum += u[idx(x, y+1, z, Nx, Ny)];if (z > 0)     sum += u[idx(x, y, z-1, Nx, Ny)];if (z < Nz-1)  sum += u[idx(x, y, z+1, Nx, Ny)];
    u[i] = inv6 * (sum + h2 * rhs);
}

/*--------------------------------------------------------------
  Black points (same kernel, just invert parity)
  --------------------------------------------------------------*/
__global__ void rbgs_black_kernel(const float * __restrict__ f, float * __restrict__ u, const int Nx, const int Ny, const int Nz, const float h2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;int y = blockIdx.y * blockDim.y + threadIdx.y;int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    // Black points: (x+y+z) % 2 == 1
    if (((x + y + z) & 1) == 0) return;

    size_t i = idx(x, y, z, Nx, Ny);
    float rhs = f[i];
    float sum = 0.0f;
    if (x > 0)     sum += u[idx(x-1, y, z, Nx, Ny)];if (x < Nx-1)  sum += u[idx(x+1, y, z, Nx, Ny)];if (y > 0)     sum += u[idx(x, y-1, z, Nx, Ny)];if (y < Ny-1)  sum += u[idx(x, y+1, z, Nx, Ny)];if (z > 0)     sum += u[idx(x, y, z-1, Nx, Ny)];if (z < Nz-1)  sum += u[idx(x, y, z+1, Nx, Ny)];
    u[i] = inv6 * (sum + h2 * rhs);
}

/*--------------------------------------------------------------
  Residual kernel: r = f - A u
  --------------------------------------------------------------*/
__global__ void residual_kernel(const float * __restrict__ f, const float * __restrict__ u, float * __restrict__ r, const int Nx, const int Ny, const int Nz, const float h2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;int y = blockIdx.y * blockDim.y + threadIdx.y;int z = blockIdx.z * blockDim.z + threadIdx.z;if (x >= Nx || y >= Ny || z >= Nz) return;

    size_t i = idx(x, y, z, Nx, Ny);
    float Laplacian = 
        ( (x>0)?u[idx(x-1,y,z,Nx,Ny)]:0.0f + (x<Nx-1)?u[idx(x+1,y,z,Nx,Ny)]:0.0f + (y>0)?u[idx(x,y-1,z,Nx,Ny)]:0.0f + (y<Ny-1)?u[idx(x,y+1,z,Nx,Ny)]:0.0f + (z>0)?u[idx(x,y,z-1,Nx,Ny)]:0.0f + (z<Nz-1)?u[idx(x,y,z+1,Nx,Ny)]:0.0f - 6.0f*u[i]) / h2;
    r[i] = f[i] - Laplacian;
}

/*--------------------------------------------------------------
  L2 norm reduction (simple thrust wrapper)
  --------------------------------------------------------------*/
float compute_l2_norm(const float *d_vec, int numElements, cudaStream_t stream=0)
{
    thrust::device_ptr<float> d_ptr(const_cast<float*>(d_vec));
    float sq_sum = thrust::reduce(thrust::cuda::par.on(stream), d_ptr, d_ptr+numElements, 0.0f, thrust::plus<float>());
    return sqrtf(sq_sum / static_cast<float>(numElements));
}

/*--------------------------------------------------------------
  Host driver: one V‑cycle (pre‑smooth, residual, restrict,
  coarse solve, prolongate, post‑smooth)
  --------------------------------------------------------------*/
void v_cycle(float *d_u, float *d_f, float *d_r, int Nx, int Ny, int Nz, float h, int nu1=2, int nu2=2)
{
    const int numCells = Nx*Ny*Nz;
    const float h2 = h*h;
    dim3 block(8,8,8);
    dim3 grid((Nx+block.x-1)/block.x, (Ny+block.y-1)/block.y, (Nz+block.z-1)/block.z);

    cudaStream_t stream; cudaStreamCreate(&stream);

    // ---- pre‑smooth (red/black) ----
    for (size_t i=0;i<nu1;++i){
        rbgs_red_kernel<<<grid,block,0,stream>>>(d_f, d_u, Nx,Ny,Nz,h2);
        rbgs_black_kernel<<<grid,block,0,stream>>>(d_f, d_u, Nx,Ny,Nz,h2);
    }

    // ---- residual ----
    residual_kernel<<<grid,block,0,stream>>>(d_f, d_u, d_r, Nx,Ny,Nz,h2);

    // ---- restriction (full‑weighting) ----
    // (omitted for brevity – similar stencil with averaging)
    // ...

    // ---- coarse grid solve (recursively call v_cycle on half resolution) ----

    // ---- prolongation (linear interpolation) ----

    // ---- post‑smooth ----
    for (size_t i=0;i<nu2;++i){
        rbgs_red_kernel<<<grid,block,0,stream>>>(d_f, d_u, Nx,Ny,Nz,h2);
        rbgs_black_kernel<<<grid,block,0,stream>>>(d_f, d_u, Nx,Ny,Nz,h2);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

/*--------------------------------------------------------------
  Simple main to test on a manufactured solution
  --------------------------------------------------------------*/
int main()
{
    const int N = 64;               // power of two, change for scaling study
    const int Nx = Ny = Nz = N;
    const float L = 1.0f;
    const float h = L / (Nx-1);
    const int numCells = Nx*Ny*Nz;

    // allocate device vectors
    thrust::device_vector<float> d_u(numCells, 0.0f);
    thrust::device_vector<float> d_f(numCells);
    thrust::device_vector<float> d_r(numCells);

    // fill RHS for manufactured solution u = sin(pi x) sin(pi y) sin(pi z)
    // -∇²u = (π²+π²+π²) u = 3π² u
    const float pi = 3.141592653589793f;
    const float factor = 3.0f * fabsf(pi*pi);
    auto f_raw = thrust::raw_pointer_cast(d_f.data());
    auto u_raw = thrust::raw_pointer_cast(d_u.data());

    // launch a simple kernel to fill f
    dim3 block(256);
    dim3 grid((numCells+block.x-1)/block.x);
    auto init_f = [] __device__ (size_t idx, float h, int Nx,int Ny,int Nz, float *f){
        int z = idx/(Ny*Nx);
        int y = (idx%(Ny*Nx))/Nx;
        int x = idx%Nx;
        float xx = x*h, yy = y*h, zz = z*h;
        float uex = sinf(M_PI*xx) * sinf(M_PI*yy) * sinf(M_PI*zz);
        f[idx] = factor * uex;        // because -∇² uex = factor * uex
    };
    // (you can write a tiny kernel or use thrust::transform)
    // For brevity we skip the kernel launch here; assume f is set.

    // Run a few V‑cycles and monitor residual
    for (size_t it=0; it<20; ++it){
        v_cycle(u_raw, f_raw, thrust::raw_pointer_cast(d_r.data()),
                Nx,Ny,Nz,h,2,2);
        float res_norm = compute_l2_norm(d_r.data().get(), numCells);
        printf("Iter %2d  ||r||₂ = %e\n", it, res_norm);
        if (res_norm < 1e-8f) break;
    }

    // Optional: compute error vs exact solution
    // ...

    return 0;
}
