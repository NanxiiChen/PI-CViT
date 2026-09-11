import os

# 必须在导入 numpy/fenics 之前设置：每个 MPI 进程只用 1 个 BLAS/OpenMP 线程，
# 避免 8 个进程各自再开多线程互相争抢核心（oversubscription 导致严重变慢）
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time
import fenics as fn
import numpy as np
from mpi4py import MPI

start_time = time.time()

# 参数设置
lambda_ = 5
N = 63
h = 100 / N
epsilon = 6 * h / (2 * np.sqrt(2) * np.arctanh(0.9))
M = 0.1
# R0 = 35 # 不再使用固定半径

# 批量运行设置
B = 8  # 工况数 (Batch size)。每个时间步都保存后文件很大，故减少工况数
seed = 0  # 第 b 个工况使用 seed = seed + b + 1（与串行版本一致，保证参数可复现）
total_time = 3.0
init_dt = 0.001
save_every = init_dt  # 保存间隔与求解器步长一致：每个时间步都保存，不再降采样

a_range = (20, 40)
b_range = (20, 40)
theta_range = (0, np.pi)

# 预先计算时间步信息
# 注意用 round：3.0/0.001 在浮点下是 2999.9999...，直接 int() 会丢掉最后一帧
expected_steps = int(round(total_time / save_every)) + 1
times = np.linspace(0, total_time, expected_steps)

# MPI 设置：mpirun -np 8 python fem.py
# 各进程独立求解分配给自己的工况（一人一个，高度并行）；
# 网格用 comm_self，即每个进程持有完整网格做串行求解，不做跨进程区域分解
world = MPI.COMM_WORLD
rank, nprocs = world.rank, world.size

# 创建结果目录（全时间步数据，保存到新目录避免覆盖旧数据）
save_dir = "/root/autodl-tmp/data/10testdata_finedt"
if rank == 0:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Times shape: {times.shape}, running with {nprocs} MPI process(es)", flush=True)
world.Barrier()

# 创建网格和函数空间（comm_self：完整网格，独立串行求解）
mesh = fn.RectangleMesh(fn.MPI.comm_self, fn.Point(-50, -50), fn.Point(50, 50), N, N)

ele = fn.FiniteElement('CG', mesh.ufl_cell(), 1)
V = fn.FunctionSpace(mesh, ele)
mesh_points = V.tabulate_dof_coordinates()

# 计算重排索引以匹配 (H, W) 网格结构
# 按照 y (主键) 然后 x (次键) 排序，以便 reshape 成图像格式
x_coords = mesh_points[:, 0]
y_coords = mesh_points[:, 1]
sort_idx = np.lexsort((x_coords, y_coords))
H = N + 1
W = N + 1

# 降低日志级别，避免 8 进程 x 3000 步的 Newton 迭代信息刷屏
fn.set_log_level(fn.LogLevel.WARNING)

# 网格坐标只需 rank 0 保存一份
if rank == 0:
    print(f"Mesh shape: {mesh_points.shape}, Grid shape: ({H}, {W})", flush=True)
    # 调整为 (2, H, W) 格式，其中第0维是x，第1维是y
    np.save(f"{save_dir}/mesh_points.npy", mesh_points[sort_idx].reshape(H, W, 2).transpose(2, 0, 1))

phi = fn.Function(V)
phi_n = fn.Function(V)
v = fn.TestFunction(V)
u = fn.TrialFunction(V)

class InitCondition(fn.UserExpression):
    def __init__(self, a, b, theta, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.theta = theta
        self.epsilon = epsilon

    def eval_cell(self, values, x, ufl_cell):
        # 坐标旋转
        x_rot = x[0] * np.cos(self.theta) + x[1] * np.sin(self.theta)
        y_rot = -x[0] * np.sin(self.theta) + x[1] * np.cos(self.theta)

        # 椭圆方程项: sqrt((x'/a)^2 + (y'/b)^2)
        term = np.sqrt((x_rot/self.a)**2 + (y_rot/self.b)**2)

        # 缩放因子 (使用轴的调和平均数作为特征长度，保持界面宽度一致)
        # 数学原理：调和平均数偏向较小的轴，防止在短轴处界面过陡导致网格无法解析(数值稳定性)
        scale = 2 * self.a * self.b / (self.a + self.b)

        # 近似符号距离: scale * (1 - term)
        # 内部 term < 1 -> dist > 0; 外部 term > 1 -> dist < 0
        dist = scale * (1 - term)

        values[0] = np.tanh(dist / (np.sqrt(2) * self.epsilon))

    def value_shape(self):
        return ()

# 定义变分形式
dx = fn.dx()
dt = fn.Constant(init_dt)  # 时间步长

E = (
    (phi - phi_n) / dt * v * dx
    + M * fn.dot(fn.grad(phi), fn.grad(v)) * dx
    + M / epsilon**2 * (phi**3 - phi) * v * dx
    + lambda_ * fn.sqrt(2 * 0.25 * (phi**2 - 1)**2) / epsilon * v * dx
)

J = fn.derivative(E, phi, u)
problem = fn.NonlinearVariationalProblem(E, phi, J=J)
solver = fn.NonlinearVariationalSolver(problem)
solver.parameters['newton_solver']['absolute_tolerance'] = 1E-8
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']["convergence_criterion"] = "incremental"
solver.parameters['newton_solver']["relative_tolerance"] = 1e-8
solver.parameters['newton_solver']["maximum_iterations"] = 10

# 数据存储
# rank 0 预分配磁盘 memmap，其余进程随后以 r+ 打开；
# 各进程只写自己的 batch 切片（互不重叠），每算完一个 batch 立即落盘，
# 峰值内存约 100MB/进程，且崩溃只损失正在算的 batch
solutions_path = f"{save_dir}/solutions.npy"
if rank == 0:
    solutions_mmap = np.lib.format.open_memmap(
        solutions_path, mode="w+", dtype=np.float64,
        shape=(B, expected_steps, 1, H, W),
    )
world.Barrier()
if rank != 0:
    solutions_mmap = np.lib.format.open_memmap(solutions_path, mode="r+")

# 每个进程写自己的日志文件，避免多进程写同一文件时行交错
log_path = f"{save_dir}/time_logs_r{rank}.txt"
with open(log_path, "w") as f:
    f.write("batch,sim_time,compute_time\n")

# 分配工况：进程 rank 依次负责 b = rank, rank + nprocs, ...
# （B=8、8 进程时正好一人一个；直接 python fem.py 则退化为串行）
my_params = []
for b in range(rank, B, nprocs):
    np.random.seed(seed + b + 1)  # 与串行版本相同：第 b 个工况用 seed b+1
    print(f"[rank {rank}] === Starting Batch {b+1}/{B} ===", flush=True)

    # 随机生成参数
    a_val = np.random.uniform(*a_range)
    b_val = np.random.uniform(*b_range)
    if a_val < b_val:
        a_val, b_val = b_val, a_val  # 确保 a 是长轴
    theta_val = np.random.uniform(*theta_range)

    params = [a_val, b_val, theta_val]
    my_params.append((b, params))
    print(f"[rank {rank}] Params: a={a_val:.2f}, b={b_val:.2f}, theta={theta_val:.2f}", flush=True)

    # 初始化场
    phi_init = InitCondition(a=a_val, b=b_val, theta=theta_val, epsilon=epsilon, degree=2)
    phi_n.interpolate(phi_init)
    phi.interpolate(phi_init)

    now = 0.0
    dt.assign(init_dt)

    # 当前 batch 的解存储
    batch_sol = []

    # 保存初始状态 (t=0)
    # 获取向量 -> 重排 -> Reshape (1, H, W)
    sol_vec = phi.vector().get_local()[sort_idx].reshape(1, H, W)
    batch_sol.append(sol_vec)

    next_save = save_every

    while now < total_time:
        try:
            info = solver.solve()
        except RuntimeError:
            print(f"[rank {rank}] Newton solver did not converge", flush=True)
            now -= dt.values()
            dt.assign(dt.values() / 2)
            # print('Decreasing time step to', dt.values()[0])
            if dt.values() < 1E-8:
                break
            continue

        now += dt.values()[0]
        phi_n.assign(phi)

        # 定时保存（save_every = init_dt，即每个时间步都保存）
        if now >= next_save - 1e-6:
            sol_vec = phi.vector().get_local()[sort_idx].reshape(1, H, W)
            batch_sol.append(sol_vec)
            next_save += save_every

            # 每步都写日志会产生约 3 万行，改为每 100 次保存记录一次
            if len(batch_sol) % 100 == 0:
                checkpoint_time = time.time()
                with open(log_path, "a") as f:
                    f.write(f"{b},{now},{checkpoint_time - start_time}\n")

    # 确保每个 batch 的时间步数一致 (补齐或截断)
    if len(batch_sol) > expected_steps:
        batch_sol = batch_sol[:expected_steps]
    elif len(batch_sol) < expected_steps:
        # 如果因为步长太小没跑完，复制最后一帧补齐
        last_frame = batch_sol[-1]
        for _ in range(expected_steps - len(batch_sol)):
            batch_sol.append(last_frame)

    # 本 batch 立即写入磁盘 memmap 中自己的切片（随后 batch_sol 可被回收）
    solutions_mmap[b] = np.array(batch_sol)
    solutions_mmap.flush()
    del batch_sol
    print(f"[rank {rank}] Batch {b+1}/{B} done (t={now:.3f}), "
          f"elapsed {time.time() - start_time:.1f}s", flush=True)

# 汇总所有工况参数到 rank 0，按 batch 序号排序后统一保存
gathered = world.gather(my_params, root=0)
if rank == 0:
    all_params = [p for chunk in gathered for p in chunk]
    all_params.sort(key=lambda bp: bp[0])  # (b, params) 按 b 排序
    final_params = np.array([p for _, p in all_params])

    print(f"Final Solutions shape: {solutions_mmap.shape}")
    print(f"Final Params shape: {final_params.shape}")

    np.save(f"{save_dir}/initial_params.npy", final_params)
    np.save(f"{save_dir}/times.npy", times)

    print('Simulation finished')
    print('Total Time elapsed:', time.time() - start_time)

del solutions_mmap
world.Barrier()
