# Multigrid-V-Cycle-Poisson-Equation-Solution-CUDA-Using-Red-Black-Gauss-Siedel\
Docker instructions:\
In Powershell type:\
docker build -t cuda133-cpp-dev .\
docker run --gpus all -it -v ${PWD}:/workspace cuda133-cpp-dev bash\
In container type:\
nvcc --version //check the version and ensure not broken\
nvcc -O3 --use_fast_math --generate-code arch=compute_120,code=sm_120 --ptxas-options=-v Poisson_basic.cu -o Poisson_solver\

Implementing a Poisson equation solver that takes a multigrid approach using Red-Black Gauss-Siedel to calculate a solution for a given test equation.

Poisson Equation:\
∇² u = f\
Where:\
∇² (del squared or the Laplacian) represents the divergence of the gradient of a function\
u is the unknown scalar field or potential you are solving for.\
f is the known source term, such as charge density or mass density.\
Vector Mechanics: In Cartesian 3D space, the Laplacian is the sum of the second partial derivatives: \(\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\)\
Laplace's Equation Relationship: If the region of space has zero source (f = 0), the Poisson equation simplifies to Laplace's equation (∇² u = 0),\
Intuitive Meaning: Imagine u is the physical displacement of a rubber sheet, and f is the force pushing down on it at specific points. The equation mathematically balances how much the field curves based on the applied force.\

Multigrid method:\
a multigrid method (MG method) is an algorithm for solving differential equations using a hierarchy of discretizations.
An example of a class of techniques called multiresolution methods, very useful in problems exhibiting multiple scales of behavior.
The main idea of multigrid is to accelerate the convergence of a basic iterative method (known as relaxation, which generally reduces short-wavelength error) by a global correction of the fine grid solution approximation from time to time, accomplished by solving a coarse problem. 
coarse problem, while cheaper to solve, is similar to the fine grid problem in that it also has short- and long-wavelength errors.
This recursive process is repeated until a grid is reached where the cost of direct solution there is negligible compared to the cost of one relaxation sweep on the fine grid.

V-Cycle MATLAB:\
function phi = V_Cycle(phi,f,h)\
// Recursive V-Cycle Multigrid for solving the Poisson equation (\nabla^2 phi = f) on a uniform grid of spacing h\

// Pre-Smoothing\
    phi = smoothing(phi,f,h);\

// Compute Residual Errors\
    r = residual(phi,f,h);\

// Restriction\
    rhs = restriction(r);\
    eps = zeros(size(rhs));\

// stop recursion at smallest grid size, otherwise continue recursion\
    if smallest_grid_size_is_achieved\
        eps = coarse_level_solve(eps,rhs,2*h);\
    else\
        eps = V_Cycle(eps,rhs,2*h);\
    end\

// Prolongation and Correction\
    phi = phi + prolongation(eps);\

// Post-Smoothing\
    phi = smoothing(phi,f,h);\
end\

I will be using CUDA C++, as GPUs are well suited to solving these problems, due to their large memory bandwidth and thousands of compute cores. I will stick with FP32, due to significant fall-off in performance on the GPU I am using. 104TFLOPS vs 1.67TFLOP for FP32 vs FP64, but may experiment with different levels of precision for each grid cycle.

Plan:\
Want to take a fused kernel approach to multi-grid v-cycle, as launching multiple kernels add a lot of latency.\
Fine-to-Coarse:\
Single fused Kernel performs v_1 pre smoothing iterations in shared memory, and calculates the residual (r=f- Ax), restricts to coarse grid ((I_h)^2h * r) and only writes the coarse back to global memory. Thus avoids writing the fine grid to memory.\
Coarse-to-Fine:\
Create a fused kernel that reads the corrected coarse-grid solution, prolongates to fine grid ((I_h)^2h * e), adds it to the existing fine solution, and immediately  performs v_2 post-smoothing iterations entirely within shared memory before a single global memory write.\
Reads the Coarse\

Extension:
Temporal Blocking - For even greater efficiency, multiple full iterations can be fused into a single kernel using temporal blocking.
1. A larger spatial tile is loaded into shared memory
2. Threads iterate through multiple red-black cycles purely inside cache and shared memory, safely advancing solution through multiple steps.
3. The tile shrinks slighltly each step to accomodate advancing data dependancies (dependancy pyramid) before final write-back.
Advantages:
Reduced Memory Banwidth Pressure - Converts a heavily memory-bound algorithm into a compute-bound task by replacing slow global memory round trips with fast shared memory registers.
Eliminated launch overhead - Removes the CPU overhead associated with dispatching thousands of separate kernel launches during long running iterations
Cache-locality - Keeps adjacent stencil data inside L1/Shared memory subsystem, ensuring minimal cache thrashing.

Initial conclusions on successful compilation and execution:\
Program is executed extremely quickly, completing in 8.49 seconds. Suggesting that I had much more headroom for a more complex example. This is a memory bound problem; the actual computation is finished very quickly.\
In this example, we ran in block sizes of (8,8,8), which was a good starting point, but may not be optimal for this device. Additionally, our compiler flags haven't been optimised for this application, and I will read up on what are optimal for the type of program I am running
Nsight Event View:\
Name	                        Start	    Duration	TID\
InitializeProblem	            4.08954s	2.584 ms	3264\
Lazy Function Loading	        4.09206s	36.541 μs	3264\
cudaDeviceSynchronize	        4.09213s	6.134 ms	3264\
cuLibraryGetKernel	            4.09829s	1.410 μs	3264\
cuKernelGetName	                4.09829s	430 ns	    3264\
red_black_gauss_siedel_red	    4.09829s	52.982 μs	3264\
Lazy Function Loading	        4.09829s	32.041 μs	3264\
cuLibraryGetKernel	            4.09835s	430 ns	    3264\
cuKernelGetName	                4.09835s	140 ns	    3264\
red_black_gauss_siedel_black	4.09835s	23.151 μs	3264\
Lazy Function Loading	        4.09835s	16.580 μs	3264\
cuKernelGetName	                4.09837s	90 ns	    3264\
red_black_gauss_siedel_red	    4.09837s	4.730 μs	3264\
cuKernelGetName	                4.09838s	80 ns	    3264\
red_black_gauss_siedel_black	4.09838s	4.770 μs	3264\
cuLibraryGetKernel	            4.09838s	390 ns	    3264\
cuKernelGetName	                4.09838s	80 ns	    3264\
Residual_function	            4.09838s	16.980 μs	3264\
Lazy Function Loading	        4.09838s	11.490 μs	3264\
cudaMemset	                    4.0984s	    14.001 μs	3264\
cuLibraryGetKernel	            4.09841s	140 ns	    3264\
cuKernelGetName	                4.09841s	140 ns	    3264\
Coarsen	                        4.09841s	17.520 μs	3264\
Lazy Function Loading	        4.09841s	12.820 μs	3264\
cuKernelGetName	                4.09843s	100 ns	    3264\
red_black_gauss_siedel_red	    4.09843s	5.401 μs	3264\
cuKernelGetName	                4.09844s	230 ns	    3264\
red_black_gauss_siedel_black	4.09844s	4.220 μs	3264\
cuKernelGetName	                4.09844s	80 ns	    3264\
red_black_gauss_siedel_red	    4.09844s	4.970 μs	3264\
cuKernelGetName	                4.09845s	80 ns	    3264\
red_black_gauss_siedel_black	4.09845s	4.500 μs	3264\
cuKernelGetName	                4.09845s	140 ns	    3264\
Residual_function	            4.09845s	4.850 μs	3264\
cudaMemset	                    4.09846s	5.790 μs	3264\
cuKernelGetName	                4.09846s	140 ns	    3264\
Coarsen	                        4.09846s	4.730 μs	3264\

From this, we see that the the actual computation is on the order of microseconds/ milliseconds. Further examination of the Event View and sorting by duration shows that our longest events are cudaMemcpy, cudaDeviceSynchronize and cudaMalloc, with no significant relation to the iteration step. That is to say, it is not related to the changing size of the grid as it varies over number of iterations, as it is a fairly constant time consumption.\
Sorting by Time duration:\
Name	                Start	    Duration	TID\
cudaMemcpy	            5.94264s	249.006 ms	3264\
cudaMemcpy	            7.65956s	241.925 ms	3264\
cudaMemcpy	            6.80614s	239.777 ms	3264\
cudaMemcpy	            5.08573s	239.636 ms	3264\
cudaMemcpy	            4.22976s	239.377 ms	3264\
cudaMalloc	            0.677199s	199.553 ms	3264\
cudaDeviceSynchronize	4.95407s	131.601 ms	3264\
cudaDeviceSynchronize	5.81142s	131.085 ms	3264\
cudaDeviceSynchronize	4.09882s	130.627 ms	3264\
cudaDeviceSynchronize	6.67553s	130.548 ms	3264\
cudaDeviceSynchronize	7.5292s	    130.301 ms	3264\
cudaMalloc	            0.943477s	24.055 ms	3264\
cudaMalloc	            0.900136s	23.678 ms	3264\
cudaMalloc	            0.876753s	23.381 ms	3264\
cudaMalloc	            0.967533s	22.944 ms	3264\
cudaFree	            8.38281s	20.967 ms	3264\
cudaFree	            8.40378s	19.845 ms	3264\
cudaFree	            8.46232s	19.545 ms	3264\
cudaFree	            8.42363s	19.538 ms	3264\
cudaFree	            8.44317s	19.151 ms	3264\
cudaDeviceSynchronize	4.09213s	6.134 ms	3264\
cudaMalloc	            0.932645s	5.828 ms	3264\
cudaMalloc	            0.925102s	3.790 ms	3264\
cudaMalloc	            0.928893s	3.751 ms	3264\
cudaFree	            8.48187s	2.719 ms	3264\
InitializeProblem	    4.08954s	2.584 ms	3264\
cudaFree	            8.48459s	2.556 ms	3264\
cudaFree	            8.48715s	2.503 ms	3264\
cudaMalloc	            0.939862s	1.138 ms	3264\
cuLibraryLoadData	    4.08824s	971.458 μs	3264\
cudaMemset	            0.92429s	735.456 μs	3264\
cudaMalloc	            0.939202s	658.799 μs	3264\
cudaMalloc	            0.93856s	641.164 μs	3264\
cudaMalloc	            0.941751s	516.220 μs	3264\
cudaFree	            8.49056s	478.054 μs	3264\
cudaFree	            8.4901s	    461.454 μs	3264\
cudaFree	            8.48965s	447.412 μs	3264\
cudaMalloc	            0.941076s	350.690 μs	3264\
cudaMalloc	            0.942343s	341.602 μs	3264\
cudaMalloc	            0.942686s	336.848 μs	3264\
cudaMalloc	            0.941428s	321.419 μs	3264\
cuLibraryGetKernel	    4.08921s	316.364 μs	3264\

Changing block size from (8,8,8) to (32, 8, 1) gave very minimal difference. Not suprising, given that this is a highly memory bound application. If I want to see significant improvements, I need to consider efficient loading and unloading of data and variables, which in some cases, could include serious redesign of the code.
