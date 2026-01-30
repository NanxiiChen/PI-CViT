# Shallow water equation

\[
\frac{\partial h}{\partial t} + H \left( \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right) = 0, \\
\frac{\partial u}{\partial t} - fv + g \frac{\partial h}{\partial x} = 0, \\
\frac{\partial v}{\partial t} + fu + g \frac{\partial h}{\partial y} = 0.
\]

Or in vector form:
\[
\frac{\partial h}{\partial t} + H \nabla \cdot \mathbf{u} = 0, \\
\frac{\partial \mathbf u }{\partial t} + f \mathbf{k} \times \mathbf{u} + g \nabla h = 0.
\]
subject to periodic boundary conditions and initial conditions:
\[
h(x,y,0) = h_0(x,y), \\
\mathbf{u}(x,y,0) = \mathbf{0} ,\\
x, y \in [0, 1), t\in [0, 1].  
\]
Where:
- \( h \) is the perturbation of the free surface height
- \( H \) is the mean fluid depth
- \( \mathbf{u} = (u, v) \) is the horizontal velocity
- \( f \) is the Coriolis parameter, we set \( f = 1 \) here
- \( g \) is the gravitational acceleration, we set \( g = 1 \)
- \( h_0(x, y) \) is a randomly generated initial condition using Gaussian random fields with \(l = 0.1\)
- \( \mathbf{k} \) is the unit vector in the vertical direction with \( \mathbf{k} = (0, 0, 1) \)
