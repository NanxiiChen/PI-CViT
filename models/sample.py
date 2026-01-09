import jax
from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt


def mesh_flat(*args):
    return [coord.reshape(-1, 1) for coord in jnp.meshgrid(*args)]


def lhs_sampling(mins, maxs, num, key):
    dim = len(mins)
    
    sub_key, *keys = random.split(key, dim + 1)
    u = (jnp.arange(0, num) + random.uniform(sub_key, (num,))) / num
    # u = (jnp.arange(0, num) + 0.5) / num

    result = jnp.zeros((num, dim))

    for i in range(dim):
        perm = random.permutation(keys[i], u)
        result = result.at[:, i].set(mins[i] + perm * (maxs[i] - mins[i]))

    return result

     
def shifted_grid(mins, maxs, nums, key, eps=1e-3):
    dim = len(mins)
    mins = jnp.array(mins)
    maxs = jnp.array(maxs)
    nums = jnp.array(nums)

    grids = []
    distances = (maxs - mins) / (nums - 1)

    keys = random.split(key, dim)
    shifts = jnp.array(
        [
            random.uniform(keys[i], shape=(), minval=-distances[i], maxval=distances[i])
            for i in range(dim)
        ]
    )

    # create shifted grids
    for i in range(dim):
        grid_i = jnp.linspace(mins[i], maxs[i], nums[i]) + shifts[i]
        grid_i = jnp.clip(grid_i, mins[i] + eps, maxs[i] - eps)
        grids.append(grid_i)

    data = jnp.stack(mesh_flat(*grids), axis=-1).reshape(-1, dim)
    data = data[random.permutation(key, data.shape[0])]
    return data   

if __name__ == "__main__":
    key = random.PRNGKey(0)

    # 定义采样范围和数量
    mins = jnp.array([-1.0,1.0])
    maxs = jnp.array([-1.0,1.0])
    num_samples = 10

    # 生成拉丁超立方体采样点
    samples = lhs_sampling(mins, maxs, num_samples, key)
    print(samples)

    # 可视化采样点
    plt.scatter(samples[:, 0], samples[:, 1], c='blue', marker='o')
    plt.title('Latin Hypercube Sampling')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(mins[0], maxs[0])
    plt.ylim(mins[1], maxs[1])
    plt.grid()
    plt.savefig('tmp/lhssample.png')
    plt.show()