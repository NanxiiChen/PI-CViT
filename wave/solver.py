"""
使用伪谱方法(FFT)求解2D波动方程（非常数波速）
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


class WaveSolver2D:
    """
    2D 波动方程求解器（非常数波速）
    
    方程形式：
    u_tt = c(x,y)^2 * (u_xx + u_yy)
    
    转换为一阶系统：
    u_t = v
    v_t = c^2 * (u_xx + u_yy)
    
    使用伪谱方法在频域计算空间导数
    """
    
    def __init__(self, Nx, Ny, c_field, Lx=1.0, Ly=1.0):
        """
        初始化求解器
        
        Parameters:
        -----------
        Nx, Ny : int
            网格点数
        c_field : array, shape (Nx, Ny)
            波速场 c(x,y)
        Lx, Ly : float
            空间域大小
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.c_field = c_field
        self.c2_field = c_field**2  # 预计算 c^2
        
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
    
    def compute_laplacian(self, u):
        """
        在频域计算拉普拉斯算子
        
        Parameters:
        -----------
        u : array, shape (Nx, Ny)
            场在空间域的表示
        
        Returns:
        --------
        laplacian_u : array
            拉普拉斯算子在空间域的表示
        """
        u_hat = fft2(u)
        laplacian_u_hat = self.laplacian * u_hat
        
        # 去混叠
        laplacian_u_hat[~self.dealias] = 0
        
        return np.real(ifft2(laplacian_u_hat))
    
    def compute_rhs(self, u, v):
        """
        计算右端项
        
        Parameters:
        -----------
        u : array, shape (Nx, Ny)
            位移场
        v : array, shape (Nx, Ny)
            速度场
        
        Returns:
        --------
        rhs_u, rhs_v : array
            u和v的时间导数
        """
        # u的时间导数
        rhs_u = v
        
        # v的时间导数: c^2 * laplacian(u)
        laplacian_u = self.compute_laplacian(u)
        rhs_v = self.c2_field * laplacian_u
        
        return rhs_u, rhs_v
    
    def rk4_step(self, u, v, dt):
        """
        四阶Runge-Kutta时间步进
        
        Parameters:
        -----------
        u : array
            当前时刻的位移场
        v : array
            当前时刻的速度场
        dt : float
            时间步长
        
        Returns:
        --------
        u_new, v_new : array
            下一时刻的位移场和速度场
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
        求解波动方程
        
        Parameters:
        -----------
        u0 : array, shape (Nx, Ny)
            初始位移场
        v0 : array, shape (Nx, Ny)
            初始速度场 (通常为0)
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


def solve_multiple_ics(initial_conditions, c_fields, Nx, Ny, t_end, dt, save_interval):
    """
    对多个初始条件求解波动方程
    
    Parameters:
    -----------
    initial_conditions : array, shape (N, Nx, Ny)
        N个初始位移场
    c_fields : array, shape (N, Nx, Ny)
        N个波速场
    Nx, Ny : int
        网格点数
    t_end : float
        结束时间
    dt : float
        时间步长
    save_interval : float
        保存间隔
    
    Returns:
    --------
    solutions : array, shape (N, T, Nx, Ny)
        求解结果（位移场）
    times : array
        保存的时间点
    """
    N = initial_conditions.shape[0]
    
    # 计算保存时刻数
    save_times = np.arange(0, t_end + save_interval/2, save_interval)
    T = len(save_times)
    
    # 初始化存储
    solutions = np.zeros((N, T, Nx, Ny))
    
    # 对每个初始条件求解
    for i in range(N):
        print(f"\n求解初始条件 {i+1}/{N}...")
        
        # 使用初始条件
        u0 = initial_conditions[i]
        v0 = np.zeros((Nx, Ny))  # 初始速度为0
        c_field = c_fields[i]
        
        # 创建求解器
        solver = WaveSolver2D(Nx, Ny, c_field)
        
        # 求解
        result = solver.solve(u0, v0, t_end, dt, save_interval)
        
        # 保存结果
        solutions[i, :, :, :] = result['u']
    
    return solutions, save_times


def generate_periodic_field(Nx, Ny, length_scale=0.1, amplitude=1.0, seed=None,
                            Lx=1.0, Ly=1.0):
    """
    生成周期性随机场（使用RBF核的谱方法）
    
    Parameters:
    -----------
    Nx, Ny : int
        网格点数
    length_scale : float
        RBF核的长度尺度
    amplitude : float
        场的振幅
    seed : int
        随机种子
    Lx, Ly : float
        空间域大小
    
    Returns:
    --------
    field : array, shape (Nx, Ny)
        生成的随机场
    """
    if seed is not None:
        np.random.seed(seed)
    
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # RBF核的谱密度
    spectrum = np.exp(-0.5 * length_scale**2 * K2)

    # 生成随机噪声
    noise = np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)
    field_hat = noise * np.sqrt(spectrum)

    # 转换到空间域
    field = np.fft.ifft2(field_hat).real
    field = (field - field.mean()) / field.std()
    
    return amplitude * field


if __name__ == "__main__":
    
    # 参数设置
    N = 8  # 样本数
    Nx, Ny = 64, 64  # 网格大小
    t_end = 1.0  # 结束时间
    dt = 0.0001  # 时间步长
    save_interval = 0.1  # 保存间隔
    u0_length_scale = 0.1  # 初始条件长度尺度
    c_length_scale = 0.5  # 波速场长度尺度
    u0_amplitude = 0.2  # 初始位移振幅
    c_mean = 1.0  # 波速平均值
    c_std = 0.2  # 波速标准差
    seed = 123  # 随机种子
    
    print("="*60)
    print("2D 波动方程求解器（非常数波速）")
    print("="*60)
    print(f"参数配置:")
    print(f"  样本数: N = {N}")
    print(f"  网格: {Nx} x {Ny}")
    print(f"  时间范围: [0, {t_end}]")
    print(f"  时间步长: dt = {dt}")
    print(f"  保存间隔: {save_interval}s")
    print(f"  初始条件长度尺度: {u0_length_scale}")
    print(f"  波速场长度尺度: {c_length_scale}")
    print(f"  初始位移振幅: {u0_amplitude}")
    print(f"  波速: mean={c_mean}, std={c_std}")
    print("="*60)
    
    # 生成初始条件和波速场
    print("\n步骤1: 生成随机初始条件和波速场...")
    initial_conditions = np.zeros((N, Nx, Ny))
    c_fields = np.zeros((N, Nx, Ny))

    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    for i in range(N):
        # 生成初始位移场 u0
        u0 = generate_periodic_field(Nx, Ny, u0_length_scale, u0_amplitude, 
                                     seed=seed+i*2)
        initial_conditions[i] = u0
        # 生成波速场 c(x,y)
        c_field = 1 + 0.5 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        c_fields[i] = c_field

    
    print(f"初始条件形状: {initial_conditions.shape}")
    print(f"初始条件值范围: [{np.min(initial_conditions):.6f}, {np.max(initial_conditions):.6f}]")
    print(f"波速场形状: {c_fields.shape}")
    print(f"波速值范围: [{np.min(c_fields):.6f}, {np.max(c_fields):.6f}]")
    
    # 求解方程
    print("\n步骤2: 求解波动方程...")
    solutions, times = solve_multiple_ics(
        initial_conditions=initial_conditions,
        c_fields=c_fields,
        Nx=Nx,
        Ny=Ny,
        t_end=t_end,
        dt=dt,
        save_interval=save_interval
    )
    # solutions 形状: (N, T, Nx, Ny)
    # reshape as (N, T, C=1, Nx, Ny)
    solutions = np.expand_dims(solutions, axis=2)
    
    print(f"\n求解完成！")
    print(f"解的形状: {solutions.shape}")
    print(f"  N (样本数): {solutions.shape[0]}")
    print(f"  T (时间点数): {solutions.shape[1]}")
    print(f"  Nx: {solutions.shape[2]}")
    print(f"  Ny: {solutions.shape[3]}")
    print(f"保存的时间点: {times}")
    
    # 保存结果
    print("\n步骤3: 保存结果...")
    np.savez('./data/wave/wave_solutions.npz',
             solutions=np.transpose(solutions, (0, 1, 2, 4, 3)),  # 转置为 (N, T, C, Ny, Nx)
             times=times,
             x=x,
             y=y,
             c_fields=np.transpose(c_fields, (0, 2, 1)),  # 转置为 (N, Ny, Nx)
             initial_conditions=np.transpose(initial_conditions, (0, 2, 1)),  # 转置为 (N, Ny, Nx)
             dt=dt)
    print("结果已保存到 wave_solutions.npz")
    
    # 统计信息
    print("\n解的统计信息:")
    print(f"  最小值: {np.min(solutions):.4f}")
    print(f"  最大值: {np.max(solutions):.4f}")
    print(f"  均值: {np.mean(solutions):.4f}")
    print(f"  标准差: {np.std(solutions):.4f}")
    
    # 可视化
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
        plt.savefig('tmp/wave_evolution.png', dpi=150)
        print("\n波动演化过程可视化已保存到 wave_evolution.png")
        
        # 可视化波速场
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(8, N)):
            im = axes[i].contourf(x, y, c_fields[i].T, levels=20, cmap='viridis')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_title(f'Wave speed field (sample {i+1})')
            axes[i].axis('equal')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('tmp/wave_speed_fields.png', dpi=150)
        print("波速场可视化已保存到 wave_speed_fields.png")
        
    except ImportError:
        print("\nmatplotlib未安装，跳过可视化")