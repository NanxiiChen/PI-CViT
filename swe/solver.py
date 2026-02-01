"""
使用伪谱方法(FFT)求解线性浅水方程 (Linear Shallow Water Equations)
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import os

class SWESolver2D:
    """
    2D 线性浅水方程求解器
    
    方程形式：
    h_t = - H * (u_x + v_y)
    u_t = f*v - g*h_x
    v_t = -f*u - g*h_y
    """
    
    def __init__(self, Nx, Ny, Lx=1.0, Ly=1.0, H=1.0, f=1.0, g=1.0):
        """
        初始化求解器
        """
        self.Nx = Nx
        self.Ny = Ny
        self.H = H
        self.f = f
        self.g = g
        
        # 空间网格
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.y = np.linspace(0, Ly, Ny, endpoint=False)
        
        # 频域波数
        self.kx = 2 * np.pi * fftfreq(Nx, Lx/Nx)
        self.ky = 2 * np.pi * fftfreq(Ny, Ly/Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        
    def compute_rhs(self, h, u, v):
        """
        计算右端项
        """
        # 转换到频域
        h_hat = fft2(h)
        u_hat = fft2(u)
        v_hat = fft2(v)
        
        # 空间导数 (频域)
        h_x_hat = 1j * self.KX * h_hat
        h_y_hat = 1j * self.KY * h_hat
        u_x_hat = 1j * self.KX * u_hat
        v_y_hat = 1j * self.KY * v_hat
        
        # 计算方程右端项 (Linear Shallow Water Equations)
        # h_t = -H * div(u)
        rhs_h_hat = -self.H * (u_x_hat + v_y_hat)
        
        # u_t = f*v - g*h_x
        rhs_u_hat = self.f * v_hat - self.g * h_x_hat
        
        # v_t = -f*u - g*h_y
        rhs_v_hat = -self.f * u_hat - self.g * h_y_hat
        
        # 转换回空间域 (返回实部)
        rhs_h = np.real(ifft2(rhs_h_hat))
        rhs_u = np.real(ifft2(rhs_u_hat))
        rhs_v = np.real(ifft2(rhs_v_hat))
        
        return rhs_h, rhs_u, rhs_v
    
    def rk4_step(self, h, u, v, dt):
        """
        三变量的四阶Runge-Kutta时间步进
        """
        # k1
        k1_h, k1_u, k1_v = self.compute_rhs(h, u, v)
        
        # k2
        k2_h, k2_u, k2_v = self.compute_rhs(
            h + 0.5*dt*k1_h, 
            u + 0.5*dt*k1_u, 
            v + 0.5*dt*k1_v
        )
        
        # k3
        k3_h, k3_u, k3_v = self.compute_rhs(
            h + 0.5*dt*k2_h, 
            u + 0.5*dt*k2_u, 
            v + 0.5*dt*k2_v
        )
        
        # k4
        k4_h, k4_u, k4_v = self.compute_rhs(
            h + dt*k3_h, 
            u + dt*k3_u, 
            v + dt*k3_v
        )
        
        # 更新
        h_new = h + (dt/6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        u_new = u + (dt/6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_new = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return h_new, u_new, v_new

    def solve(self, h0, u0, v0, t_end, dt, save_interval):
        """
        求解循环
        """
        # 计算保存时刻
        save_times = np.arange(0, t_end + save_interval/2, save_interval)
        n_saves = len(save_times)
        
        # 初始化存储
        h_history = np.zeros((n_saves, self.Nx, self.Ny))
        u_history = np.zeros((n_saves, self.Nx, self.Ny))
        v_history = np.zeros((n_saves, self.Nx, self.Ny))
        
        # 保存初始条件
        h_history[0] = h0.copy()
        u_history[0] = u0.copy()
        v_history[0] = v0.copy()
        
        # 初始化
        h, u, v = h0.copy(), u0.copy(), v0.copy()
        t = 0
        save_idx = 1
        next_save_time = save_interval
        
        # 时间积分
        n_steps = int(t_end / dt)
        print(f"  Total steps: {n_steps}")
        
        for step in range(n_steps):
            h, u, v = self.rk4_step(h, u, v, dt)
            t += dt
            
            # 检查是否需要保存
            if save_idx < n_saves and t >= next_save_time - dt/2:
                h_history[save_idx] = h.copy()
                u_history[save_idx] = u.copy()
                v_history[save_idx] = v.copy()
                save_idx += 1
                next_save_time += save_interval
        
        return {
            'times': save_times,
            'h': h_history,
            'u': u_history,
            'v': v_history,
            'x': self.x,
            'y': self.y
        }

def solve_multiple_ics(initial_h, Nx, Ny, params, t_end, dt, save_interval):
    """
    对多个初始条件求解
    initial_h: (N, Nx, Ny)
    """
    N = initial_h.shape[0]
    save_times = np.arange(0, t_end + save_interval/2, save_interval)
    T = len(save_times)
    
    solver = SWESolver2D(Nx, Ny, Lx=params['Lx'], Ly=params['Ly'], 
                         H=params['H'], f=params['f'], g=params['g'])
    
    # 存储: (N, T, 3, Nx, Ny), 通道顺序 h, u, v
    solutions = np.zeros((N, T, 3, Nx, Ny))
    
    # 初始 u, v 全为0
    u0 = np.zeros((Nx, Ny))
    v0 = np.zeros((Nx, Ny))
    
    for i in range(N):
        print(f"求解样本 {i+1}/{N}...")
        h0 = initial_h[i]
        
        result = solver.solve(h0, u0, v0, t_end, dt, save_interval)
        
        solutions[i, :, 0, :, :] = result['h']
        solutions[i, :, 1, :, :] = result['u']
        solutions[i, :, 2, :, :] = result['v']
        
    return solutions, save_times

def generate_periodic_ics_h(Nx, Ny, length_scale=0.1, amplitude=1.0, seed=None, Lx=1.0, Ly=1.0):
    """
    生成周期性高斯随机场作为初始高度 h
    """
    if seed is not None:
        np.random.seed(seed)
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # RBF kernel spectrum
    spectrum = np.exp(-0.5 * length_scale**2 * K2)
    noise = np.random.randn(Nx, Ny) + 1j * np.random.randn(Nx, Ny)
    u_hat = noise * np.sqrt(spectrum)

    field = np.fft.ifft2(u_hat).real
    field = (field - field.mean()) / field.std() # Normalize
    return amplitude * field

if __name__ == "__main__":
    # 参数配置
    N_samples = 8
    Nx, Ny = 64, 64
    t_end = 1.0
    dt = 0.001
    save_interval = 0.1
    
    # 物理参数
    swe_params = {
        'Lx': 1.0, 'Ly': 1.0,
        'H': 1.0,  # 平均水深
        'f': 10.0,  # 科氏参数
        'g': 1.0   # 重力加速度
    }
    
    # 初始条件参数
    ic_params = {
        'length_scale': 0.1,
        'amplitude': 0.5, # 扰动幅度
        'seed': 0
    }

    print("="*60)
    print("Linear Shallow Water Equation Solver")
    print(f"Grid: {Nx}x{Ny}, Samples: {N_samples}")
    print(f"Physics: {swe_params}")
    print("="*60)

    # 1. 生成初始条件 h0
    print("\nGenerating Initial Conditions...")
    initial_h = np.zeros((N_samples, Nx, Ny))
    for i in range(N_samples):
        initial_h[i] = generate_periodic_ics_h(Nx, Ny, 
                                               length_scale=ic_params['length_scale'], 
                                               amplitude=ic_params['amplitude'], 
                                               seed=ic_params['seed']+i)
    
    # 2. 求解
    print("\nSolving Equations...")
    solutions, times = solve_multiple_ics(initial_h, Nx, Ny, swe_params, t_end, dt, save_interval)
    
    # --- diagnostics: mean(h) 和 总能量 随时间 ---
    # 选第一个样本
    h_hist = solutions[0, :, 0, :, :]  # (T, Nx, Ny)
    u_hist = solutions[0, :, 1, :, :]
    v_hist = solutions[0, :, 2, :, :]
    T = h_hist.shape[0]

    mean_h = h_hist.mean(axis=(1, 2))
    energy = 0.5 * (swe_params['g'] * (h_hist**2).mean(axis=(1,2)) + swe_params['H'] * ((u_hist**2 + v_hist**2).mean(axis=(1,2))))

    # 绝对质量变化（更稳健）
    abs_mean_range = mean_h.max() - mean_h.min()
    abs_mean_change = abs_mean_range
    print(f"Diagnostics (sample 0): mean(h) abs change = {abs_mean_change:.3e}")

    # 检查零波数(k=0)模（质量守恒对应 k=0 模保持常数）
    h_hat_all = np.fft.fft2(h_hist, axes=(1,2))          # shape (T, Nx, Ny) complex
    h0_mode = h_hat_all[:, 0, 0]
    h0_abs_change = np.abs(h0_mode).max() - np.abs(h0_mode).min()
    h0_mean_abs = np.abs(h0_mode).mean()
    if h0_mean_abs < 1e-12:
        h0_rel_change = np.nan
    else:
        h0_rel_change = h0_abs_change / h0_mean_abs
    print(f"k=0 mode: abs change = {h0_abs_change:.3e}, rel change = {h0_rel_change if not np.isnan(h0_rel_change) else 'nan'}")

    rel_energy_change = (energy.max() - energy.min()) / (np.abs(energy).mean() + 1e-16)
    print(f"Diagnostics (sample 0): energy change rel = {rel_energy_change:.3e}")
    
    # 3. 保存
    print("\nSaving Results...")
    os.makedirs('./data/swe', exist_ok=True)
    save_path = './data/swe/swe_solutions.npz'
    
    # 转置为 (N, T, Channels, Ny, Nx) 以适应常规习惯或后续处理
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    
    np.savez(save_path,
             solutions=np.transpose(solutions, (0, 1, 2, 4, 3)), 
             times=times,
             x=x, y=y,
             **swe_params)
    print(f"Saved to {save_path}")
    print(f"Shape: {solutions.shape} (N, T, C, Nx, Ny), Channels: 0=h, 1=u, 2=v")

    # 4. 简单可视化 (Optional)
    try:
        import matplotlib.pyplot as plt
        os.makedirs('tmp', exist_ok=True)
        
        # 画第一个样本的 h 场演化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        plot_indices = np.linspace(0, len(times)-1, 6, dtype=int)
        
        for i, t_idx in enumerate(plot_indices):
            h_field = solutions[0, t_idx, 0, :, :]   # 改为 sample 0
            im = axes[i].contourf(x, y, h_field.T, levels=20, cmap='RdBu_r')
            axes[i].set_title(f'h (t={times[t_idx]:.2f})')
            plt.colorbar(im, ax=axes[i])
            axes[i].set_aspect('equal')
            
        plt.tight_layout()
        plt.savefig('tmp/swe_evolution_h.png')
        print("Visualization saved to tmp/swe_evolution_h.png")
        
    except ImportError:
        print("Matplotlib not found, skipping visualization.")
