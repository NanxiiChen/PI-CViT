# Lid-driven-cavity Flow

Steady-state solution of the incompressible Navier-Stokes equations for a lid-driven-cavity flow.
\[
\nabla \cdot \mathbf{u} = 0, \\
\mathbf{u} \cdot \nabla \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u},
\]
subject to no-slip boundary conditions on the left, bottom, and right walls:
\[
    \mathbf{u} = 0 \quad \text{on the left, bottom, and right walls}
\]
and a moving lid on the top wall:
\[
    \mathbf{u} = (U, 0) \quad \text{on the top wall}
\]

The streamfunction-vorticity formulation is used to eliminate the pressure term:
\[
\nabla^2 \psi = -\omega, \\
\mathbf{u} = \left( \frac{\partial \psi}{\partial y}, -\frac{\partial \psi}{\partial x} \right), \\
\mathbf{u} \cdot \nabla \omega = \nu \nabla^2 \omega,
\]
where \(\psi\) is the streamfunction and \(\omega\) is the vorticity. The boundary conditions for the streamfunction are:
\[
    \psi = 0 \quad \text{on all walls}
\]
