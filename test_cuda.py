import torch
import sys

def test_cuda():
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        
        # 创建一个测试张量并移动到GPU
        x = torch.rand(5, 5)
        print("\nCPU张量:")
        print(x)
        print(f"设备: {x.device}")
        
        x_cuda = x.cuda()
        print("\nGPU张量:")
        print(x_cuda)
        print(f"设备: {x_cuda.device}")
        
        # 进行一些基本运算
        y = torch.rand(5, 5).cuda()
        z = x_cuda + y
        print("\nGPU上的加法运算结果:")
        print(z)
        
        # 测试矩阵乘法
        matrix_mult = torch.matmul(x_cuda, y)
        print("\nGPU上的矩阵乘法结果:")
        print(matrix_mult)
    else:
        print("警告: CUDA不可用，请检查您的CUDA安装和PyTorch配置。")

if __name__ == "__main__":
    test_cuda() 