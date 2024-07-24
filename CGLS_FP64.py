import numpy as np
import torch
import matplotlib.pyplot as plt

# 设置随机数种子
np.random.seed(42)
torch.manual_seed(42)

def generate_diagonal_matrix(condition_number, distribution_type, size=1024):
    if distribution_type == 1:
        # 聚类，大部分元素接近小值，少数元素接近条件数
        diag = np.ones(size)
        diag[-1] = condition_number
    elif distribution_type == 2: 
        # 大部分元素接近条件数，少数元素接近小值
        diag = np.ones(size) * condition_number
        diag[-1] = 1
    elif distribution_type == 3:  # 对角线元素呈等比关系
        diag = np.geomspace(1, condition_number, size)
    elif distribution_type == 4:  # 对角线元素呈等差关系
        diag = np.linspace(1, condition_number, size)
    elif distribution_type == 5:  # randn 正态分布
        diag = np.random.randn(size)
    elif distribution_type == 6:  # rand 均匀分布
        diag = np.random.rand(size)
    else:
        raise ValueError("Invalid distribution type")

    # 根据给定的条件数调整对角矩阵的对角线元素
    diag *= np.sqrt(condition_number / np.max(diag**2))

    # 生成对角矩阵，并明确指定元素为双精度浮点数
    diagonal_matrix = np.diag(diag).astype(np.float64)

    # 生成正交矩阵
    ortho_matrix, _ = np.linalg.qr(np.random.randn(size, size))

    # 计算矩阵乘积
    result_matrix = np.dot(np.dot(ortho_matrix, diagonal_matrix), ortho_matrix.T)

    return diagonal_matrix, result_matrix

# 修改cgls_qr函数，在每一轮迭代中记录 norms/norms0 的值
def cgls_qr_with_norms_tracking(A, b, shift=0, tol=1e-6, max_iters=None, prnt=False, x0=None): #shift 正则化参数
    # 判断 A 的类型
    if isinstance(A, torch.Tensor):
        explicitA = True
    elif callable(A):
        explicitA = False
    else:
        raise ValueError("Invalid type for A. It must be either a torch.Tensor or a callable function.")
    
    m, n = A.size()
    x = torch.zeros(n, 1, dtype=torch.float64)
    r = b - A.mm(x) # 计算初始残差向量
    s = A.t().mm(r) # 计算正交化残差向量
    p = s.clone() # 初始化搜索方向向量 
    norms0 = torch.norm(r) # 计算初始残差的二范数 
    gamma = norms0**2

    norms_ratio_list = [] # 用于存储每一轮迭代中的 norms/norms0 值

    if max_iters is None:
        max_iters = min(m, n, 20)

    if x0 is not None:
        x = x0


    print(norms0)
    norms_ratio_list.append(1)


    for k in range(1, max_iters + 1):
        # q, _ = torch.triangular_solve(A.mm(p), A) # A * q = A * p
        if explicitA:
            q = A.mm(p)
            s = A.t().mm(r) - shift * x
        else:
            q = A(p, 1)
            s = A(r, 2) - shift * x
        
        delta = torch.norm(q)**2 + shift * torch.norm(p)**2 # 计算参数 delta，用于计算步长 alpha
        alpha = gamma / delta # 计算步长 alpha
        x = x + alpha * p # 更新解向量
        r = r - alpha * q # 更新残差向量
        # s, _ = torch.triangular_solve(A.t().mm(r), A.t()) # 计算更新后的正交化残差向量

        norms = torch.norm(r) # 计算更新后的正交化残差的二范数
        norms_ratio = norms / norms0
        norms_ratio_list.append(norms_ratio.item())
        # gamma1 = gamma
        # gamma = norms**2 # 更新参数 gamma
        gamma = torch.norm(A.t().mm(r))**2
        gamma1 = torch.norm(s)**2 # 更新参数 gamma
        beta = gamma1 / gamma
        p = s + beta * p

        if norms < tol * norms0: # 如果满足收敛条件则跳出循环
            break

    print(torch.norm(A.mm(x) - b))
    return x, norms_ratio_list

def validate_matrix_generation(condition_number, distribution_type, max_iters=200, shift=0, tol= 1e-6, prnt=False, x0=None):
    diag_matrix, result_matrix = generate_diagonal_matrix(condition_number, distribution_type)
    A = torch.tensor(result_matrix, dtype=torch.float64)
    b = torch.randn(A.size(0), 1, dtype=torch.float64)
    
    # # 使用矩阵 A 和向量 b 的伪逆来计算初始解向量
    # x0 = torch.tensor(np.linalg.pinv(A.numpy()).dot(b.numpy()), dtype=torch.float64)
    x0 = torch.randn(A.size(1), 1, dtype=torch.float64) # 统一设成0

    x, norms_ratio_list = cgls_qr_with_norms_tracking(A, b, shift=shift, tol=tol, max_iters=max_iters, prnt=prnt, x0=x0)
    print(A.mm(x) - b)
    
    plt.plot(range(1, len(norms_ratio_list) + 1), norms_ratio_list)
    print(norms_ratio_list)
    plt.xlabel('Iterations')
    plt.ylabel('Norms/norms0')
    plt.title(f'Condition Number: {condition_number}, Distribution Type: {distribution_type}')
    plt.xticks(range(0, len(norms_ratio_list), 5))
    # plt.yticks(np.arange(0.99998, 1.0, 0.00001))
    # plt.ticklabel_format(style='plain', axis='y')  # 设置y轴刻度格式为普通格式，避免科学计数法
    ymin, ymax = 0.9998, 1.0
    plt.yticks(np.arange(ymin, ymax + 0.00002, 0.00002))  # 确保y轴刻度覆盖所需范围
    formatter = plt.FuncFormatter(lambda x, _: f"{x:.5f}")
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()

# 启动验证函数
validate_matrix_generation(100000, 1)



# # 打印对角矩阵和结果矩阵
# diag_matrix, result_matrix = generate_diagonal_matrix(100, 1)


# print("Generated Diagonal Matrix:")
# print(diag_matrix)


# print("\nResult Matrix:")
# print(result_matrix)

# # 对结果矩阵做奇异值分解
# U, S, V = np.linalg.svd(result_matrix)

# # 打印奇异值
# print("\nSingular Values (from SVD):")
# print(S)

# # 对角线元素
# print("\nDiagonal Matrix Values (Singular Values):")
# print(np.diag(diag_matrix))