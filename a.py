import multiprocessing
import time
import math

def is_prime(n):
    """判断一个数字是否为素数"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def find_primes_in_range(start, end):
    """在指定范围内查找所有素数"""
    return [n for n in range(start, end) if is_prime(n)]

def cpu_benchmark():
    """CPU性能测试"""
    print(f"利用所有可用核心进行性能测试...")
    num_cores = multiprocessing.cpu_count()
    print(f"检测到 {num_cores} 个核心")

    # 定义测试范围和任务分配
    start = 10**6
    end = start + 10**5
    step = (end - start) // num_cores

    ranges = [(start + i * step, start + (i + 1) * step) for i in range(num_cores)]

    # 开始性能测试
    start_time = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(find_primes_in_range, ranges)
    end_time = time.time()

    # 合并结果
    primes = [p for result in results for p in result]
    print(f"找到素数数量: {len(primes)}")
    print(f"性能测试完成，耗时 {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    cpu_benchmark()