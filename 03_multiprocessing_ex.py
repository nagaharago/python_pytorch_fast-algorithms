import psutil


def print_cpu_usage():
    # 全体のCPU使用率
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # 各CPUの使用率
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
        print(f"CPU{i} Usage: {percentage}%")


def process_data(data_chunk):
    result = []
    for i in data_chunk:
        for j in data_chunk:
            result.append(i * j)
    return result


data = list(range(1000))

# シングルプロセスによる実装
# print("=== Before Processing ===")
# print_cpu_usage()

# processed_data = process_data(data)

# print("\n=== After Processing ===")
# print_cpu_usage()

# multiprocessingによる実装
import multiprocessing

# データをCPUの数だけに分割
chunks = [
    data[i :: multiprocessing.cpu_count()] for i in range(multiprocessing.cpu_count())
]

print("=== Before Processing ===")
print_cpu_usage()

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
processed_data = pool.map(process_data, chunks)
pool.close()
pool.join()

print("\n=== After Processing ===")
print_cpu_usage()

# データを1つのリストに戻す
processed_data = [item for sublist in processed_data for item in sublist]

print(f"Processed Data: {processed_data[:10]}...")  # 最初の10数字のみを表示
