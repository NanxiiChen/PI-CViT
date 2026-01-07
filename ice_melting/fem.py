import time
import os
import fenics as fn
import numpy as np

start_time = time.time()


# 参数设置
lambda_ = 5
N = 63
h = 100 / N
epsilon = 6 * h / (2 * np.sqrt(2) * np.arctanh(0.9))
M = 0.1
# R0 = 35 # 不再使用固定半径

# 批量运行设置
B = 8  # 工况数 (Batch size)
seed = 0
total_time = 3.0
save_every = 0.1
init_dt = 0.001

a_range = (20, 40)
b_range = (20, 40)
theta_range = (0, np.pi)

# 预先计算时间步信息
expected_steps = int(total_time / save_every) + 1
times = np.linspace(0, total_time, expected_steps)
print(f"Times shape: {times.shape}")

# 创建结果目录
save_dir = "./data/ice_melting/ellipse"
os.makedirs(save_dir, exist_ok=True)

# 创建网格和函数空间
# 更改为二维矩形网格
mesh = fn.RectangleMesh(fn.Point(-50, -50), fn.Point(50, 50), N, N)

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

print(f"Mesh shape: {mesh_points.shape}, Grid shape: ({H}, {W})")
# 保存排序后的网格点
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
all_solutions = []
all_params = []

with open("time_logs.txt", "w") as f:
    f.write("batch,sim_time,compute_time\n")

# 批量循环
for b in range(B):
    seed = seed + 1
    np.random.seed(seed)
    print(f"=== Starting Batch {b+1}/{B} ===")
    
    # 随机生成参数   
    a_val = np.random.uniform(*a_range)
    b_val = np.random.uniform(*b_range)
    if a_val < b_val:
        a_val, b_val = b_val, a_val  # 确保 a 是长轴
    theta_val = np.random.uniform(*theta_range)
    # for a standard circle
    # a_val = 30
    # b_val = 30
    # theta_val = 0.0
    
    params = [a_val, b_val, theta_val]
    all_params.append(params)
    print(f"Params: a={a_val:.2f}, b={b_val:.2f}, theta={theta_val:.2f}")
    
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
            print('Newton solver did not converge')
            now -= dt.values()
            dt.assign(dt.values() / 2)
            # print('Decreasing time step to', dt.values()[0])
            if dt.values() < 1E-8:
                break
            continue
        
        now += dt.values()[0]
        phi_n.assign(phi)
        
        # 定时保存
        if now >= next_save - 1e-6:
            # print(f"  Saving at t={now:.4f}")
            sol_vec = phi.vector().get_local()[sort_idx].reshape(1, H, W)
            batch_sol.append(sol_vec)
            next_save += save_every
            
            checkpoint_time = time.time()
            with open("time_logs.txt", "a") as f:
                f.write(f"{b},{now},{checkpoint_time - start_time}\n")

    # 确保每个 batch 的时间步数一致 (补齐或截断)
    if len(batch_sol) > expected_steps:
        batch_sol = batch_sol[:expected_steps]
    elif len(batch_sol) < expected_steps:
        # 如果因为步长太小没跑完，复制最后一帧补齐
        last_frame = batch_sol[-1]
        for _ in range(expected_steps - len(batch_sol)):
            batch_sol.append(last_frame)
            
    all_solutions.append(np.array(batch_sol))

# 转换为最终的 numpy 数组
# Shape: (B, T, C, H, W)
final_solutions = np.array(all_solutions)
final_params = np.array(all_params)

print(f"Final Solutions shape: {final_solutions.shape}")
print(f"Final Params shape: {final_params.shape}")

np.save(f"{save_dir}/solutions.npy", final_solutions)
np.save(f"{save_dir}/initial_params.npy", final_params)
np.save(f"{save_dir}/times.npy", times)

print('Simulation finished')

end_time = time.time()
print('Total Time elapsed:', end_time - start_time)
