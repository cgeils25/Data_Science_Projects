"""
Experimenting with  pytorch's implementation using metal performance shaders (MPS), which is like cuda for apple silicon
"""

import torch
import time
import numpy as np

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("MPS is available and PyTorch is built with MPS support")
    device_mps = torch.device("mps")
    device_cpu = torch.device("cpu")
elif not torch.backends.mps.is_available():
    print("MPS is not available. Stopping program")
    quit()
elif not torch.backends.mps.is_built():
    print("PyTorch is not built with MPS support. Stopping program")
    quit()


def time_function(func, *args):
    """
    Times the function and returns the time it took to run and the result of the function
    """
    
    start = time.time()
    result = func(*args)
    end = time.time()
    return end - start, result

# inverse matrix 
np_array = np.random.randn(1_000, 1_000)  # numpy array
t_cpu = torch.randn(1_000, 1_000, device=device_cpu)  # tensor on cpu
t_mps = torch.randn(1_000, 1_000, device=device_mps)  # tensor on mps

def hundred_inv(matrix):
    if isinstance(matrix, torch.Tensor):
        for i in range(100):
            matrix = torch.inverse(matrix)
        return matrix
    if isinstance(matrix, np.ndarray):
        for i in range(100):
            matrix = np.linalg.inv(matrix)
        return matrix
    print('not a valid type')

def hundred_matmul(matrix):
    if isinstance(matrix, torch.Tensor):
        for i in range(100):
            matrix = torch.matmul(matrix, matrix)
            matrix = matrix / matrix
        return matrix
    if isinstance(matrix, np.ndarray):
        for i in range(100):
            matrix = np.matmul(matrix, matrix)
            matrix = matrix / matrix
        return matrix
    print('not a valid type')

def main():
    # inverse matrix experiment
    print("Inverse matrix experiment")
    total_time, result = time_function(np.linalg.inv, np_array)
    print(f"Time taken for numpy: {total_time}")

    total_time, result = time_function(torch.inverse, t_cpu)
    print(f"Time taken for torch cpu: {total_time}")

    total_time, result = time_function(torch.inverse, t_mps)
    print(f"Time taken for torch mps: {total_time}")

    # 100 x inverse matrix experiment 
    print("100 x Inverse matrix experiment")
    total_time, result = time_function(hundred_inv, np_array)
    print(f"Time taken for numpy: {total_time}")

    total_time, result = time_function(hundred_inv, t_cpu)
    print(f"Time taken for torch cpu: {total_time}")

    total_time, result = time_function(hundred_inv, t_mps)
    print(f"Time taken for torch mps: {total_time}")

    # 100x matrix multiplication experiment
    print("100 x Matrix multiplication experiment")
    total_time, result = time_function(hundred_matmul, np_array)
    print(f"Time taken for numpy: {total_time}")

    total_time, result = time_function(hundred_matmul, t_cpu)
    print(f"Time taken for torch cpu: {total_time}")

    total_time, result = time_function(hundred_matmul, t_mps)
    print(f"Time taken for torch mps: {total_time}")
    

if __name__ == "__main__":
    main()