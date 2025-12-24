# Ice Melting Simulation Example

This is an example simulating Ice melting using Legacy FEniCS (2019.1.0).

## Governing Equations

The governing equation is:

$$
\frac{\partial \phi}{\partial t} = M\left(
    \Delta \phi - \frac{F'(\phi)}{\epsilon^2}
\right) - \lambda \frac{\sqrt{2F(\phi)}}{\epsilon}
$$

where $\phi$ is the phase field variable, and the double-well potential is $F(\phi)=0.25(\phi^2-1)^2$.
For a test function $v$, the variational form is:

$$
\int_\Omega\left(\frac{\partial \phi}{\partial t}v\right) dx + M\int_\Omega\nabla\phi\cdot\nabla v dx + \frac{M}{\epsilon^2} \int_\Omega\left( \phi^3-\phi \right)vdx + \lambda \int_\Omega\frac{\sqrt{2F(\phi)}}{\epsilon}vdx = 0
$$

## Simulation Setup

- **Computational Domain**: $\Omega = [-50, 50] \times [-50, 50]$ (2D).
- **Time Stepping**: Initial time step $\Delta t = 0.005$, Total time $T=3.0$.

### Initial Condition

The initial condition is a randomized rotated ellipse defined by semi-axes $a, b$ and rotation angle $\theta$:

$$
\phi(x, y, t=0) =\tanh\left(
    \frac{d(x, y)}{\sqrt{2}\epsilon}
\right)
$$

The approximate signed distance $d(x, y)$ is calculated via the following steps:

1.  **Coordinate Rotation**: Transform global coordinates $(x, y)$ to the ellipse's local frame $(x', y')$:
    $$
    x' = x \cos\theta + y \sin\theta, \quad y' = -x \sin\theta + y \cos\theta
    $$
2.  **Normalized Radius**:
    $$
    r_{norm} = \sqrt{\left(\frac{x'}{a}\right)^2 + \left(\frac{y'}{b}\right)^2}
    $$
3.  **Distance Approximation**:
    $$
    d(x, y) \approx S \cdot (1 - r_{norm}), \quad \text{where } S = \frac{2ab}{a+b}
    $$
    The scaling factor $S$ (harmonic mean of the axes) is used to approximate the distance near the interface, ensuring $d > 0$ inside and $d < 0$ outside.

The parameters are sampled uniformly for each simulation in the batch:
- Semi-axes: $a, b \sim U[20, 40]$
- Rotation: $\theta \sim U[0, \pi]$

### Boundary Conditions

Neumann boundary conditions are applied:
$$
\frac{\partial \phi}{\partial \mathbf{n}} = 0
$$

### Parameters

- $\lambda = 5$: Coupling coefficient
- $N=63$: Mesh resolution (resulting in $(N+1) \times (N+1)$ grid points)
- $h=\dfrac{100}{N}$: Grid spacing
- $\epsilon=\dfrac{6h}{2\sqrt{2}\tanh^{-1}(0.9)}$: Interface thickness
- $M = 0.1$: Mobility coefficient
- $a, b \sim U[20, 40]$: Random semi-axes
- $\theta \sim U[0, \pi]$: Random rotation angle

## Data Generation

The script runs a batch of simulations (default $B=10$) with different random initial parameters and saves the results to `./results`.
