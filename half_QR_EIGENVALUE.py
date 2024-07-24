import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_diagonal_matrix(condition_number, distribution_type, size=10):
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

def mixPrecisionQR(A):
    A = A.float()  # 将输入转换为float32
    Q, R = torch.linalg.qr(A)  # 在float32下执行QR分解
    
    # 将Q转换为float16（半精度）
    Q_fp16 = Q.half()
    
    # 将R转换为float16（半精度）
    R_fp16 = R.half()
    
    return Q_fp16, R_fp16

def main():
    SIZE = 10
    
    # 生成一个具有指定条件数的对角矩阵
    _, A_ = generate_diagonal_matrix(condition_number=1000, distribution_type=6, size=SIZE)
    A = torch.from_numpy(A_)  # 转换为torch张量
    A = A.to('cuda')  # 移动到GPU
    A = A.float()  # 确保是float32类型
    
    Q, R = mixPrecisionQR(A)  # 执行混合精度QR分解
    
    # 重构矩阵 A
    A_reconstructed = Q.matmul(R)
    
    # 将矩阵转换回 float32 以计算特征值
    A = A.float()
    A_reconstructed = A_reconstructed.float()
    
    # 计算特征值
    eigvals_A = torch.linalg.eigvalsh(A).cpu().numpy()
    eigvals_A_reconstructed = torch.linalg.eigvalsh(A_reconstructed).cpu().numpy()
    
    # 打印特征值
    print("Original Matrix Eigenvalues:\n", eigvals_A)
    print("Reconstructed Matrix Eigenvalues:\n", eigvals_A_reconstructed)
    
    # 比较特征值分布
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Matrix Eigenvalues")
    plt.hist(eigvals_A, bins=SIZE, alpha=0.7)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Matrix Eigenvalues")
    plt.hist(eigvals_A_reconstructed, bins=SIZE, alpha=0.7)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
