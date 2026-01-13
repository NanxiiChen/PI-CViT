"""
使用伪谱方法(FFT)求解2D Burgers方程
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


class BurgersSolver2D:
    """
    2D 矢量Burgers方程求解器
    
    方程形式：
    u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
    v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
    
    使用伪谱方法在频域计算空间导数
    """
    
    def __init__(self, Nx, Ny, Lx=1.0, Ly=1.0, nu=0.01):
        """
        初始化求解器
        
        Parameters:
        -----------
        Nx, Ny : int
            网格点数
        Lx, Ly : float
            空间域大小
        nu : float
            粘性系数
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu = nu
        
        # 空间网格
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.y = np.linspace(0, Ly, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        
        # 频域波数
        self.kx = 2 * np.pi * fftfreq(Nx, Lx/Nx)
        self.ky = 2 * np.pi * fftfreq(Ny, Ly/Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='xy')
        
        # 拉普拉斯算子 (频域)
        self.laplacian = -(self.KX**2 + self.KY**2)
        
        # 用于去混叠的2/3规则
        self.dealias = self.get_dealias_filter()
    
    def get_dealias_filter(self):
        """
        创建去混叠滤波器 (2/3规则)
        """
        dealias = np.ones((self.Nx, self.Ny), dtype=bool)
        dealias[np.abs(self.KX) > (2/3) * np.max(np.abs(self.kx))] = False
        dealias[np.abs(self.KY) > (2/3) * np.max(np.abs(self.ky))] = False
        return dealias
    
    def spatial_derivative(self, u_hat, direction):
        """
        在频域计算空间导数
        
        Parameters:
        -----------
        u_hat : complex array
            场在频域的表示
        direction : str
            'x' 或 'y'
        
        Returns:
        --------
        du : real array
            导数在空间域的表示
        """
        if direction == 'x':
            du_hat = 1j * self.KX * u_hat
        elif direction == 'y':
            du_hat = 1j * self.KY * u_hat
        else:
            raise ValueError("direction must be 'x' or 'y'")
        
        # 去混叠
        # du_hat[~self.dealias] = 0
        
        return np.real(ifft2(du_hat))
    
        
    def compute_rhs(self, u, v):
        """
        计算右端项
        
        Parameters:
        -----------
        u, v : array, shape (Nx, Ny)
            速度场的两个分量
        
        Returns:
        --------
        rhs_u, rhs_v : array
            u和v的时间导数
        """
        # 转换到频域
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        # 计算空间导数
        u_x = self.spatial_derivative(u_hat, 'x')
        u_y = self.spatial_derivative(u_hat, 'y')
        v_x = self.spatial_derivative(v_hat, 'x')
        v_y = self.spatial_derivative(v_hat, 'y')
        
        # 非线性项 (矢量形式)
        # u方程: -u*u_x - v*u_y
        nl_u = -(u * u_x + v * u_y)
        # v方程: -u*v_x - v*v_y
        nl_v = -(u * v_x + v * v_y)
        
        # 转换到频域
        nl_u_hat = fft2(nl_u)
        nl_v_hat = fft2(nl_v)
        
        # 去混叠
        nl_u_hat[~self.dealias] = 0
        nl_v_hat[~self.dealias] = 0
        
        # 粘性项 (频域)
        visc_u_hat = self.nu * self.laplacian * u_hat
        visc_v_hat = self.nu * self.laplacian * v_hat
        
        # 总的右端项 (频域)
        rhs_u_hat = nl_u_hat + visc_u_hat
        rhs_v_hat = nl_v_hat + visc_v_hat
        
        # 转换回空间域
        rhs_u = np.real(ifft2(rhs_u_hat))
        rhs_v = np.real(ifft2(rhs_v_hat))
        
        return rhs_u, rhs_v
    
    def rk4_step(self, u, v, dt):
        """
        四阶Runge-Kutta时间步进
        
        Parameters:
        -----------
        u, v : array
            当前时刻的速度场两个分量
        dt : float
            时间步长
        
        Returns:
        --------
        u_new, v_new : array
            下一时刻的速度场两个分量
        """
        # k1
        k1_u, k1_v = self.compute_rhs(u, v)
        
        # k2
        k2_u, k2_v = self.compute_rhs(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v)
        
        # k3
        k3_u, k3_v = self.compute_rhs(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v)
        
        # k4
        k4_u, k4_v = self.compute_rhs(u + dt*k3_u, v + dt*k3_v)
        
        # 更新
        u_new = u + (dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_new = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return u_new, v_new
    
    def solve(self, u0, v0, t_end, dt, save_interval):
        """
        求解Burgers方程
        
        Parameters:
        -----------
        u0, v0 : array, shape (Nx, Ny)
            初始条件（两个分量）
        t_end : float
            结束时间
        dt : float
            时间步长
        save_interval : float
            保存间隔
        
        Returns:
        --------
        solution : dict
            包含时间序列和解
        """
        # 计算保存时刻
        save_times = np.arange(0, t_end + save_interval/2, save_interval)
        n_saves = len(save_times)
        
        # 初始化存储
        u_history = np.zeros((n_saves, self.Nx, self.Ny))
        v_history = np.zeros((n_saves, self.Nx, self.Ny))
        
        # 保存初始条件
        u_history[0] = u0.copy()
        v_history[0] = v0.copy()
        
        # 初始化
        u = u0.copy()
        v = v0.copy()
        t = 0
        save_idx = 1
        next_save_time = save_interval
        
        # 时间积分
        n_steps = int(t_end / dt)
        for step in range(n_steps):
            # RK4步进
            u, v = self.rk4_step(u, v, dt)
            t += dt
            
            # 检查是否需要保存
            if save_idx < n_saves and t >= next_save_time - dt/2:
                u_history[save_idx] = u.copy()
                v_history[save_idx] = v.copy()
                save_idx += 1
                next_save_time += save_interval
                
                print(f"  Time: {t:.3f}/{t_end:.3f} (step {step+1}/{n_steps})")
        
        return {
            'times': save_times,
            'u': u_history,
            'v': v_history,
            'x': self.x,
            'y': self.y
        }


def solve_multiple_ics(initial_conditions, Nx, Ny, nu, t_end, dt, save_interval):
    """
    对多个初始条件求解矢量Burgers方程
    
    Parameters:
    -----------
    initial_conditions : array, shape (N, 2, Nx, Ny)
        N个初始条件（每个有u和v两个通道）
    Nx, Ny : int
        网格点数
    nu : float
        粘性系数
    t_end : float
        结束时间
    dt : float
        时间步长
    save_interval : float
        保存间隔
    
    Returns:
    --------
    solutions : array, shape (N, T, 2, Nx, Ny)
        求解结果（2个通道：u和v）
    times : array
        保存的时间点
    """
    N = initial_conditions.shape[0]
    
    # 计算保存时刻数
    save_times = np.arange(0, t_end + save_interval/2, save_interval)
    T = len(save_times)
    
    # 创建求解器
    solver = BurgersSolver2D(Nx, Ny, nu=nu)
    
    # 初始化存储 (channels=2)
    solutions = np.zeros((N, T, 2, Nx, Ny))
    
    # 对每个初始条件求解
    for i in range(N):
        print(f"\n求解初始条件 {i+1}/{N}...")
        
        # 使用初始条件
        u0 = initial_conditions[i, 0]
        v0 = initial_conditions[i, 1]
        
        # 求解
        result = solver.solve(u0, v0, t_end, dt, save_interval)
        
        # 保存结果 (通道0=u, 通道1=v)
        solutions[i, :, 0, :, :] = result['u']
        solutions[i, :, 1, :, :] = result['v']
    
    return solutions, save_times


def generate_periodic_ics(Nx, Ny, length_scale=0.1, amplitude=0.01, seed=None,
                          Lx=1.0, Ly=1.0):
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2

    # Spectral density of RBF kernel
    spectrum = np.exp(-0.5 * length_scale**2 * K2)

    noise = np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)
    u_hat = noise * np.sqrt(spectrum)

    field = np.fft.ifft2(u_hat).real
    field = (field - field.mean()) / field.std()
    return amplitude * field


if __name__ == "__main__":
    
    # 参数设置
    N = 8  # 样本数
    Nx, Ny = 64, 64  # 网格大小
    nu = 0.01  # 粘性系数
    t_end = 1.0  # 结束时间
    dt = 0.001  # 时间步长
    save_interval = 0.1  # 保存间隔
    length_scale = 0.1  # RBF kernel长度尺度
    amplitude = 0.2  # 初始条件振幅
    seed = 543  # 初始条件随机种子
    
    print("="*60)
    print("2D Burgers方程求解器")
    print("="*60)
    print(f"参数配置:")
    print(f"  样本数: N = {N}")
    print(f"  网格: {Nx} x {Ny}")
    print(f"  粘性系数: nu = {nu}")
    print(f"  时间范围: [0, {t_end}]")
    print(f"  时间步长: dt = {dt}")
    print(f"  保存间隔: {save_interval}s")
    print(f"  RBF长度尺度: {length_scale}")
    print(f"  初始条件振幅: {amplitude}")
    print("="*60)
    
    # 生成初始条件 (两个通道)
    print("\n步骤1: 生成随机初始条件...")
    initial_conditions = np.zeros((N, 2, Nx, Ny))
    for i in range(N):
        # 为u分量生成初始条件
        field_u = generate_periodic_ics(Nx, Ny, length_scale, amplitude, seed=seed+i*2)
        initial_conditions[i, 0] = field_u
        # 为v分量生成初始条件
        field_v = generate_periodic_ics(Nx, Ny, length_scale, amplitude, seed=seed+i*2+1)
        initial_conditions[i, 1] = field_v
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    print(f"初始条件形状: {initial_conditions.shape}")
    print(f"初始条件值范围: [{np.min(initial_conditions):.6f}, {np.max(initial_conditions):.6f}]")
    
    # 求解方程
    print("\n步骤2: 求解Burgers方程...")
    solutions, times = solve_multiple_ics(
        initial_conditions=initial_conditions,
        Nx=Nx,
        Ny=Ny,
        nu=nu,
        t_end=t_end,
        dt=dt,
        save_interval=save_interval
    )
    
    print(f"\n求解完成！")
    print(f"解的形状: {solutions.shape}")
    print(f"  N (样本数): {solutions.shape[0]}")
    print(f"  T (时间点数): {solutions.shape[1]}")
    print(f"  Channels (通道数): {solutions.shape[2]} (0=u, 1=v)")
    print(f"  Nx: {solutions.shape[3]}")
    print(f"  Ny: {solutions.shape[4]}")
    print(f"保存的时间点: {times}")
    
    # 保存结果
    print("\n步骤3: 保存结果...")
    np.savez('./data/burgers/burgers_solutions.npz',
             solutions=solutions,
             times=times,
             x=x,
             y=y,
             nu=nu,
             dt=dt)
    print("结果已保存到 burgers_solutions.npz")
    
    # 统计信息
    print("\n解的统计信息:")
    print(f"  最小值: {np.min(solutions):.4f}")
    print(f"  最大值: {np.max(solutions):.4f}")
    print(f"  均值: {np.mean(solutions):.4f}")
    print(f"  标准差: {np.std(solutions):.4f}")
    
    # 可选：可视化
    try:
        import matplotlib.pyplot as plt
        
        # 可视化第一个样本在不同时刻的演化 (u分量)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # 选择要显示的时间点
        time_indices = [0, 2, 4, 6, 8, 10]  # t=0, 0.2, 0.4, 0.6, 0.8, 1.0
        
        for i, t_idx in enumerate(time_indices):
            u = solutions[0, t_idx, 0, :, :]  # u分量
            im = axes[i].contourf(x, y, u.T, levels=20, cmap='RdBu_r')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_title(f't = {times[t_idx]:.1f}s (u)')
            axes[i].axis('equal')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('tmp/burgers_evolution_u.png', dpi=150)
        print("\nu分量演化过程可视化已保存到 burgers_evolution_u.png")
        
        # 可视化v分量
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, t_idx in enumerate(time_indices):
            v = solutions[0, t_idx, 1, :, :]  # v分量
            im = axes[i].contourf(x, y, v.T, levels=20, cmap='RdBu_r')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_title(f't = {times[t_idx]:.1f}s (v)')
            axes[i].axis('equal')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('tmp/burgers_evolution_v.png', dpi=150)
        print("v分量演化过程可视化已保存到 burgers_evolution_v.png")
        
        # 可视化涡量场
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        kx = 2 * np.pi * fftfreq(Nx, 1/Nx)
        ky = 2 * np.pi * fftfreq(Ny, 1/Ny)
        KX, KY = np.meshgrid(kx, ky, indexing='xy')
        
        for i, t_idx in enumerate(time_indices):
            u = solutions[0, t_idx, 0, :, :]
            v = solutions[0, t_idx, 1, :, :]
            u_hat = fft2(u)
            v_hat = fft2(v)
            u_x_hat = 1j * KX * u_hat
            u_y_hat = 1j * KY * u_hat
            v_x_hat = 1j * KX * v_hat
            v_y_hat = 1j * KY * v_hat
            u_x = np.real(ifft2(u_x_hat))
            u_y = np.real(ifft2(u_y_hat))
            v_x = np.real(ifft2(v_x_hat))
            v_y = np.real(ifft2(v_y_hat))
            omega = v_x - u_y  # 涡量
            
            im = axes[i].contourf(x, y, omega.T, levels=20, cmap='RdBu_r')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_title(f't = {times[t_idx]:.1f}s (vorticity)')
            axes[i].axis('equal')
            plt.colorbar(im, ax=axes[i])
        plt.tight_layout()
        plt.savefig('tmp/burgers_evolution_vorticity.png', dpi=150)
        
    except ImportError:
        print("\nmatplotlib未安装，跳过可视化")