# Multigrid-V-Cycle-Poisson-Equation-Solution-CUDA-Using-Red-Black-Gauss-Siedel
Implementing a Poisson equation solver that takes a multigrid approach using Red-Black Gauss-Siedel to calculate a solution for a given test equation.

Poisson Equation:
∇² u = f
Where:
∇² (del squared or the Laplacian) represents the divergence of the gradient of a function
u is the unknown scalar field or potential you are solving for.
f is the known source term, such as charge density or mass density.
Vector Mechanics: In Cartesian 3D space, the Laplacian is the sum of the second partial derivatives: \(\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\)
Laplace's Equation Relationship: If the region of space has zero source (f = 0), the Poisson equation simplifies to Laplace's equation (∇² u = 0),
Intuitive Meaning: Imagine u is the physical displacement of a rubber sheet, and f is the force pushing down on it at specific points. The equation mathematically balances how much the field curves based on the applied force.

Multigrid method:
a multigrid method (MG method) is an algorithm for solving differential equations using a hierarchy of discretizations.
an example of a class of techniques called multiresolution methods, very useful in problems exhibiting multiple scales of behavior.
The main idea of multigrid is to accelerate the convergence of a basic iterative method (known as relaxation, which generally reduces short-wavelength error) by a global correction of the fine grid solution approximation from time to time, accomplished by solving a coarse problem. 
coarse problem, while cheaper to solve, is similar to the fine grid problem in that it also has short- and long-wavelength errors.
This recursive process is repeated until a grid is reached where the cost of direct solution there is negligible compared to the cost of one relaxation sweep on the fine grid.

V-Cycle MATLAB:
function phi = V_Cycle(phi,f,h)
    % Recursive V-Cycle Multigrid for solving the Poisson equation (\nabla^2 phi = f) on a uniform grid of spacing h

    % Pre-Smoothing
    phi = smoothing(phi,f,h);

    % Compute Residual Errors
    r = residual(phi,f,h);

    % Restriction
    rhs = restriction(r);

    eps = zeros(size(rhs));

    % stop recursion at smallest grid size, otherwise continue recursion
    if smallest_grid_size_is_achieved
        eps = coarse_level_solve(eps,rhs,2*h);
    else
        eps = V_Cycle(eps,rhs,2*h);
    end

    % Prolongation and Correction
    phi = phi + prolongation(eps);

    % Post-Smoothing
    phi = smoothing(phi,f,h);
end

I will be using CUDA C++, as GPUs are well suited to solving these problems, due to their large memory bandwidth and thousands of compute cores. I will stick with FP32, due to significant fall-off in performance on the GPU I am using. 104TFLOPS vs 1.67TFLOP for FP32 vs FP64, but may experiment with different levels of precision for each grid cycle.

Plan:
Want to take a fused kernel approach to multi-grid v-cycle, as launching multiple kernels add a lot of latency. 
Fine-to-Coarse:
Single fused Kernel performs v_1 pre smoothing iterations in shared memory, and calculates the residual (r=f- Ax), restricts to coarse grid ((I_h)^2h * r) and only writes the coarse back to global memory. Thus avoids writing the fine grid to memory.
Coarse-to-Fine:
Create a fused kernel that reads the corrected coarse-grid solution, prolongates to fine grid ((I_h)^2h * e), adds it to the existing fine solution, and immediately  performs v_2 post-smoothing iterations entirely within shared memory before a single global memory write.
Reads the Coarse

Extension:
Temporal Blocking - For even greater efficiency, multiple full iterations can be fused into a single kernel using temporal blocking.
1. A larger spatial tile is loaded into shared memory
2. Threads iterate through multiple red-black cycles purely inside cache and shared memory, safely advancing solution through multiple steps.
3. The tile shrinks slighltly each step to accomodate advancing data dependancies (dependancy pyramid) before final write-back.
Advantages:
Reduced Memory Banwidth Pressure - Converts a heavily memory-bound algorithm into a compute-bound task by replacing slow global memory round trips with fast shared memory registers.
Eliminated launch overhead - Removes the CPU overhead associated with dispatching thousands of separate kernel launches during long running iterations
Cache-locality - Keeps adjacent stencil data inside L1/Shared memory subsystem, ensuring minimal cache thrashing.
