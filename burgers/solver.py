"""
使用伪谱方法(FFT)求解2D Burgers方程
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


class BurgersSolver2D:
    """
    2D 标量Burgers方程求解器
    
    方程形式：
    u_t + u*u_x + u*u_y = nu*(u_xx + u_yy)
    
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
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # 频域波数
        self.kx = 2 * np.pi * fftfreq(Nx, Lx/Nx)
        self.ky = 2 * np.pi * fftfreq(Ny, Ly/Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        
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
        du_hat[~self.dealias] = 0
        
        return np.real(ifft2(du_hat))
    
    def compute_rhs(self, u):
        """
        计算右端项
        
        Parameters:
        -----------
        u : array, shape (Nx, Ny)
            速度场
        
        Returns:
        --------
        rhs_u : array
            u的时间导数
        """
        # 转换到频域
        u_hat = fft2(u)
        
        # 计算空间导数
        u_x = self.spatial_derivative(u_hat, 'x')
        u_y = self.spatial_derivative(u_hat, 'y')
        
        # 非线性项: -u*u_x - u*u_y
        nl_u = -(u * u_x + u * u_y)
        
        # 转换到频域
        nl_u_hat = fft2(nl_u)
        
        # 去混叠
        nl_u_hat[~self.dealias] = 0
        
        # 粘性项 (频域)
        visc_u_hat = self.nu * self.laplacian * u_hat
        
        # 总的右端项 (频域)
        rhs_u_hat = nl_u_hat + visc_u_hat
        
        # 转换回空间域
        rhs_u = np.real(ifft2(rhs_u_hat))
        
        return rhs_u
    
    def rk4_step(self, u, dt):
        """
        四阶Runge-Kutta时间步进
        
        Parameters:
        -----------
        u : array
            当前时刻的速度场
        dt : float
            时间步长
        
        Returns:
        --------
        u_new : array
            下一时刻的速度场
        """
        # k1
        k1_u = self.compute_rhs(u)
        
        # k2
        k2_u = self.compute_rhs(u + 0.5*dt*k1_u)
        
        # k3
        k3_u = self.compute_rhs(u + 0.5*dt*k2_u)
        
        # k4
        k4_u = self.compute_rhs(u + dt*k3_u)
        
        # 更新
        u_new = u + (dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        
        return u_new
    
    def solve(self, u0, t_end, dt, save_interval):
        """
        求解Burgers方程
        
        Parameters:
        -----------
        u0 : array, shape (Nx, Ny)
            初始条件
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
        
        # 保存初始条件
        u_history[0] = u0.copy()
        
        # 初始化
        u = u0.copy()
        t = 0
        save_idx = 1
        next_save_time = save_interval
        
        # 时间积分
        n_steps = int(t_end / dt)
        for step in range(n_steps):
            # RK4步进
            u = self.rk4_step(u, dt)
            t += dt
            
            # 检查是否需要保存
            if save_idx < n_saves and t >= next_save_time - dt/2:
                u_history[save_idx] = u.copy()
                save_idx += 1
                next_save_time += save_interval
                
                print(f"  Time: {t:.3f}/{t_end:.3f} (step {step+1}/{n_steps})")
        
        return {
            'times': save_times,
            'u': u_history,
            'x': self.x,
            'y': self.y
        }


def solve_multiple_ics(initial_conditions, Nx, Ny, nu, t_end, dt, save_interval):
    """
    对多个初始条件求解标量Burgers方程
    
    Parameters:
    -----------
    initial_conditions : array, shape (N, Nx, Ny)
        N个初始条件
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
    solutions : array, shape (N, T, 1, Nx, Ny)
        求解结果
    times : array
        保存的时间点
    """
    N = initial_conditions.shape[0]
    
    # 计算保存时刻数
    save_times = np.arange(0, t_end + save_interval/2, save_interval)
    T = len(save_times)
    
    # 创建求解器
    solver = BurgersSolver2D(Nx, Ny, nu=nu)
    
    # 初始化存储
    solutions = np.zeros((N, T, 1, Nx, Ny))
    
    # 对每个初始条件求解
    for i in range(N):
        print(f"\n求解初始条件 {i+1}/{N}...")
        
        # 使用初始条件
        u0 = initial_conditions[i]
        
        # 求解
        result = solver.solve(u0, t_end, dt, save_interval)
        
        # 保存结果
        solutions[i, :, 0, :, :] = result['u']
    
    return solutions, save_times


def generate_periodic_grf(Nx, Ny, length_scale=1, amplitude=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
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
    length_scale = 5  # RBF kernel长度尺度
    amplitude = 0.1  # 初始条件振幅
    seed = 1234  # 初始条件随机种子
    
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
    
    # 生成初始条件
    print("\n步骤1: 生成随机初始条件...")
    # initial_conditions, x, y = generate_initial_conditions(
    #     N=N, 
    #     Nx=Nx, 
    #     Ny=Ny, 
    #     length_scale=length_scale,
    #     amplitude=amplitude,
    #     seed=42
    # )
    initial_conditions = np.zeros((N, Nx, Ny))
    for i in range(N):
        field = generate_periodic_grf(Nx, Ny, length_scale, amplitude, seed=seed+i)
        initial_conditions[i] = field
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
    print(f"  Channel: {solutions.shape[2]}")
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
        
        # 可视化第一个样本在不同时刻的演化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # 选择要显示的时间点
        time_indices = [0, 2, 4, 6, 8, 10]  # t=0, 0.2, 0.4, 0.6, 0.8, 1.0
        
        for i, t_idx in enumerate(time_indices):
            u = solutions[0, t_idx, 0, :, :]
            im = axes[i].contourf(x, y, u.T, levels=20, cmap='RdBu_r')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_title(f't = {times[t_idx]:.1f}s')
            axes[i].axis('equal')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('tmp/burgers_evolution.png', dpi=150)
        print("\n演化过程可视化已保存到 burgers_evolution.png")
        
    except ImportError:
        print("\nmatplotlib未安装，跳过可视化")
